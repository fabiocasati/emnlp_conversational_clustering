"""Generate native-format fixtures to exercise the full analysis pipeline.

Creates, under a target directory:
- enterprise_ticket_dataset_fixture.csv   (ticket_text + all 24 POV columns)
- batch_run_results/fixture_run/          (results.jsonl + per-run clustering.json
                                           + selected_topic.json, native layout)

Planted structure the pipeline must recover:
- ticket_text vocabulary correlates with `issue_type` (so TF-IDF baselines
  favor issue_type in Table R1);
- most label columns are drawn independently (near-orthogonal), except the
  granularity families, which are nested refinements (correlated by design);
- run partitions are noisy copies of their target POV's reference partition
  over a random item subset, so same-POV pairs score higher than
  different-POV pairs and the permutation test must reject.

Usage: uv run python oneshot_baselines/make_fixtures.py <out_dir> [n_items]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from loaders import POV_COLUMNS, TEXT_COLUMN

ISSUE_VOCAB = {
    "authentication_failure": ["login", "password", "sso", "credentials", "locked", "mfa"],
    "billing_dispute": ["invoice", "billing", "refund", "charge", "subscription", "quote"],
    "application_crash": ["crash", "exception", "stacktrace", "freeze", "corrupt", "kernel"],
    "performance_degradation": ["slow", "latency", "timeout", "lag", "throughput", "spike"],
    "access_request": ["permission", "role", "grant", "access", "provision", "entitlement"],
}
FILLER = ["please", "help", "ticket", "issue", "team", "urgent", "system", "user", "error", "since"]


def _column_values(col: str, n: int, rng: np.random.Generator, issue_types: np.ndarray) -> list[str]:
    if col == "issue_type":
        return list(issue_types)
    if col.endswith("_coarse") or col.endswith("_normal") or col.endswith("_fine"):
        raise AssertionError("granularity columns are built separately")
    if col == "approval_required":
        return [str(bool(x)) for x in rng.integers(0, 2, n)]
    k = int(rng.integers(3, 9))
    return [f"{col}_v{val}" for val in rng.integers(0, k, n)]


def _granularity_family(prefix: str, n: int, rng: np.random.Generator) -> dict[str, list[str]]:
    """Nested coarse -> normal -> fine labels (refinements of one another)."""
    coarse = rng.integers(0, 3, n)
    normal = coarse * 3 + rng.integers(0, 3, n)
    fine = normal * 2 + rng.integers(0, 2, n)
    return {
        f"{prefix}_coarse": [f"{prefix[:3]}C{v}" for v in coarse],
        f"{prefix}_normal": [f"{prefix[:3]}N{v}" for v in normal],
        f"{prefix}_fine": [f"{prefix[:3]}F{v}" for v in fine],
    }


def make_dataset_csv(out_csv: Path, n: int, rng: np.random.Generator) -> None:
    issue_names = list(ISSUE_VOCAB)
    issue_types = np.array([issue_names[i] for i in rng.integers(0, len(issue_names), n)])

    columns: dict[str, list[str]] = {}
    for fam in ("intent_label", "organizational_label"):
        columns.update(_granularity_family(fam, n, rng))
    for col in POV_COLUMNS:
        if col not in columns:
            columns[col] = _column_values(col, n, rng, issue_types)

    texts = []
    seen = set()
    for i in range(n):
        vocab = ISSUE_VOCAB[issue_types[i]]
        words = rng.choice(vocab, size=10).tolist() + rng.choice(FILLER, size=5).tolist()
        rng.shuffle(words)
        text = " ".join(words) + f" ref{i:05d}"  # suffix guarantees uniqueness
        if text in seen:
            raise AssertionError("duplicate fixture text")
        seen.add(text)
        texts.append(text)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([TEXT_COLUMN, *POV_COLUMNS])
        for i in range(n):
            writer.writerow([texts[i], *(columns[c][i] for c in POV_COLUMNS)])


def make_batch_run_tree(
    batch_dir: Path,
    dataset_csv: Path,
    rng: np.random.Generator,
    povs: tuple[str, ...] = ("issue_type", "department", "intent_label_normal"),
    runs_per_pov: int = 5,
    coverage: float = 0.6,
    noise: float = 0.15,
) -> None:
    with dataset_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r[TEXT_COLUMN] for r in rows]

    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    for cfg_idx, pov in enumerate(povs):
        config_slug = f"{cfg_idx:02d}-fixtcfg{cfg_idx}"
        run_ids = []
        ref_values = [r[pov] for r in rows]
        distinct = sorted(set(ref_values))
        for run_idx in range(runs_per_pov):
            run_id = f"{run_idx:03d}-fixtrun{cfg_idx}{run_idx}"
            run_ids.append(run_id)
            run_dir = batch_dir / config_slug / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            subset = rng.choice(len(texts), size=int(coverage * len(texts)), replace=False)
            clusters: dict[str, list[str]] = {}
            for i in subset:
                if rng.random() < noise:
                    value = distinct[rng.integers(0, len(distinct))]
                else:
                    value = ref_values[i]
                clusters.setdefault(value, []).append(texts[i])
            payload = {
                "clusters": [
                    {"name": f"{pov}:{value}", "members": members}
                    for value, members in sorted(clusters.items())
                ]
            }
            (run_dir / "clustering.json").write_text(json.dumps(payload, indent=2))
            (run_dir / "selected_topic.json").write_text(
                json.dumps(
                    {
                        "topic_id": pov,
                        "label": pov.replace("_", " ").title(),
                        "guidance": f"Group tickets by {pov}.",
                        "generated_at_turn": 3,
                    }
                )
            )
        manifest_lines.append(
            json.dumps(
                {
                    "run_ids": run_ids,
                    "config_slug": config_slug,
                    "timestamp": "20260710T000000Z",
                    "config": {
                        "name": "fixture_run",
                        "dataset_name": "enterprise_ticket",
                        "clustering_agent_backend": "instruction_kmeans_v1",
                        "metadata": {},
                    },
                    "results": [
                        {"turns_run": 7, "stopped_reason": "clustering_saved"}
                        for _ in run_ids
                    ],
                }
            )
        )
    (batch_dir / "results.jsonl").write_text("\n".join(manifest_lines) + "\n")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: make_fixtures.py <out_dir> [n_items]")
    out_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    dataset_csv = out_dir / "enterprise_ticket_dataset_fixture.csv"
    make_dataset_csv(dataset_csv, n, rng)
    make_batch_run_tree(out_dir / "batch_run_results" / "fixture_run", dataset_csv, rng)
    print(f"fixtures written under {out_dir}")


if __name__ == "__main__":
    main()
