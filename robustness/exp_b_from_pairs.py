"""Experiment B computed from the paper's own pairwise robustness artifact.

Input: evaluation_report_details/robustness_expectations.jsonl produced by the
repo's robustness_expectations evaluator over the 146-conversation batch. Each
record carries the pair's ARI, the resolved theme of each side, and the
evaluator's expected_pair_type (invariant = same resolved theme).

Analyses:
1. Primary groups = the paper's own pair labels (invariant/responsive):
   distribution summaries + Invariance/Responsiveness Success γ/δ sweep.
2. Permutation test at the run level. Themes were resolved per pair, so a run
   can flicker between themes across pairs (LLM judge nondeterminism); for a
   run-level test each run gets its majority-vote theme, and the resulting
   grouping is reported alongside the paper-label grouping for transparency.
3. Cross-validation: recompute ARI with sklearn from the saved clustering.json
   files for a sample of pairs and compare to the artifact's values.

Usage:
  uv run python robustness/exp_b_from_pairs.py \
      --pairs-jsonl data/enterprise_ticket_run_evaluations/enterprise_ticket_robustness/evaluation_report_details/robustness_expectations.jsonl \
      --batch-dir data/final_enterprise_ticket_run/enterprise_ticket_run \
      --out-dir robustness/results
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from loaders import normalize_item_text
from metrics_core import (
    PairwiseGroups,
    ari,
    group_summary,
    invariance_responsiveness_sweep,
    permutation_test_from_pair_values,
)
from report import table_r2


def load_pairs(jsonl_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (valid pair records, failed/unlabeled records)."""
    pairs, failed = [], []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        d = rec.get("details", {})
        pair = {
            "left": f"{rec['left_config_slug']}/{rec['left_run_id']}",
            "right": f"{rec['right_config_slug']}/{rec['right_run_id']}",
            "ari": rec.get("metrics", {}).get("adjusted_rand_index"),
            "type": d.get("expected_pair_type"),
            "left_theme": d.get("left_resolved_theme_id"),
            "right_theme": d.get("right_resolved_theme_id"),
            "left_mode": d.get("left_topic_resolution_mode"),
            "right_mode": d.get("right_topic_resolution_mode"),
            "status": rec.get("status"),
            "left_clustering_path": rec.get("left_clustering_path"),
            "right_clustering_path": rec.get("right_clustering_path"),
        }
        if pair["type"] in ("invariant", "responsive") and pair["ari"] is not None:
            pairs.append(pair)
        else:
            failed.append(pair)
    if not pairs:
        raise ValueError(f"{jsonl_path}: no valid pair records")
    return pairs, failed


def majority_themes(pairs: list[dict]) -> tuple[dict[str, str], dict]:
    votes: dict[str, Counter] = defaultdict(Counter)
    modes: dict[str, Counter] = defaultdict(Counter)
    for p in pairs:
        votes[p["left"]][p["left_theme"]] += 1
        votes[p["right"]][p["right_theme"]] += 1
        modes[p["left"]][p["left_mode"]] += 1
        modes[p["right"]][p["right_mode"]] += 1
    majority = {run: v.most_common(1)[0][0] for run, v in votes.items()}
    shares = np.array([v.most_common(1)[0][1] / sum(v.values()) for v in votes.values()])
    diagnostics = {
        "n_runs": len(votes),
        "runs_with_unstable_theme": int(sum(1 for v in votes.values() if len(v) > 1)),
        "majority_share_mean": float(shares.mean()),
        "majority_share_min": float(shares.min()),
        "resolution_mode_census": dict(
            Counter(m.most_common(1)[0][0] for m in modes.values())
        ),
        "runs_per_majority_theme": dict(Counter(majority.values()).most_common()),
    }
    return majority, diagnostics


def crosscheck_ari(pairs: list[dict], repo_root: Path, sample_size: int, seed: int = 0) -> dict:
    """Recompute ARI with sklearn from clustering.json for a sample of pairs."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=min(sample_size, len(pairs)), replace=False)

    def load_assignment(rel_path: str) -> dict[str, int]:
        path = repo_root / rel_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            normalize_item_text(m): ci
            for ci, cl in enumerate(payload["clusters"])
            for m in cl["members"]
        }

    diffs = []
    for i in idx:
        p = pairs[int(i)]
        a = load_assignment(p["left_clustering_path"])
        b = load_assignment(p["right_clustering_path"])
        common = sorted(a.keys() & b.keys())
        if len(common) < 2:
            raise ValueError(f"Pair {p['left']} × {p['right']}: no overlap")
        recomputed = ari([a[x] for x in common], [b[x] for x in common])
        diffs.append(abs(recomputed - p["ari"]))
    return {
        "n_checked": len(diffs),
        "max_abs_diff": float(max(diffs)),
        "mean_abs_diff": float(np.mean(diffs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-jsonl", required=True)
    ap.add_argument("--batch-dir", help="extracted batch dir, used to remap clustering paths for the cross-check")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-permutations", type=int, default=10_000)
    ap.add_argument("--crosscheck-sample", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, failed = load_pairs(Path(args.pairs_jsonl))
    print(f"pairs: {len(pairs)} valid, {len(failed)} failed/unlabeled")
    runs = sorted({p["left"] for p in pairs} | {p["right"] for p in pairs})
    print(f"runs appearing in pairs: {len(runs)}")

    # 1) Primary analysis: the paper's own pair labels.
    same = np.array([p["ari"] for p in pairs if p["type"] == "invariant"])
    diff = np.array([p["ari"] for p in pairs if p["type"] == "responsive"])
    groups = PairwiseGroups(same_pov=same, diff_pov=diff)
    summary = group_summary(groups)
    sweep = invariance_responsiveness_sweep(groups)

    # 2) Run-level permutation test on majority-vote themes.
    majority, diag = majority_themes(pairs)
    pair_values = {(p["left"], p["right"]): p["ari"] for p in pairs}
    perm = permutation_test_from_pair_values(
        pair_values, majority, n_permutations=args.n_permutations
    )
    maj_same = np.array([p["ari"] for p in pairs if majority[p["left"]] == majority[p["right"]]])
    maj_diff = np.array([p["ari"] for p in pairs if majority[p["left"]] != majority[p["right"]]])
    label_agreement = float(
        np.mean(
            [
                (majority[p["left"]] == majority[p["right"]]) == (p["type"] == "invariant")
                for p in pairs
            ]
        )
    )

    # 3) Cross-validate the artifact's ARI values against sklearn.
    crosscheck = None
    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        remapped = []
        for p in pairs:
            q = dict(p)
            for side in ("left", "right"):
                rel = Path(p[f"{side}_clustering_path"])
                # artifact paths look like data/batch_run_results/enterprise_ticket_run/<slug>/<run>/clustering.json
                q[f"{side}_clustering_path"] = str(batch_dir / Path(*rel.parts[-3:]))
            remapped.append(q)
        crosscheck = crosscheck_ari(remapped, Path("."), args.crosscheck_sample)
        print("ARI cross-check vs sklearn:", crosscheck)
        if crosscheck["max_abs_diff"] > 1e-6:
            raise AssertionError(
                f"Artifact ARI disagrees with sklearn recomputation: {crosscheck}"
            )

    md = table_r2(summary, perm, sweep)
    (out_dir / "table_r2.md").write_text(md + "\n", encoding="utf-8")
    details = {
        "source": str(args.pairs_jsonl),
        "n_pairs_valid": len(pairs),
        "n_pairs_failed": len(failed),
        "n_runs": len(runs),
        "paper_label_groups": summary,
        "majority_theme_groups": {
            "same_pov": {"n_pairs": int(maj_same.size), "mean": float(maj_same.mean()), "sd": float(maj_same.std(ddof=1))},
            "diff_pov": {"n_pairs": int(maj_diff.size), "mean": float(maj_diff.mean()), "sd": float(maj_diff.std(ddof=1))},
            "pair_label_agreement_with_paper": label_agreement,
        },
        "theme_resolution_diagnostics": diag,
        "permutation_test_majority_themes": perm,
        "ari_crosscheck": crosscheck,
        "sweep": sweep,
    }
    (out_dir / "exp_b_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    with (out_dir / "exp_b_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "invariance_success", "responsiveness_success"])
        writer.writeheader()
        writer.writerows(sweep)

    print(f"\nwrote {out_dir}/table_r2.md, exp_b_details.json, exp_b_sweep.csv\n")
    print(md)
    print(
        f"\nmajority-theme grouping: same n={maj_same.size} mean={maj_same.mean():.4f} | "
        f"diff n={maj_diff.size} mean={maj_diff.mean():.4f} | "
        f"agreement with paper labels {label_agreement:.4f}"
    )


if __name__ == "__main__":
    main()
