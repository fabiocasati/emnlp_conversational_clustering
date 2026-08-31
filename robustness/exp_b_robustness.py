"""Experiment B — same-POV vs. different-POV robustness disambiguation (Table R2).

Input is either the native batch-run directory or the long-format CSV export:
  uv run python robustness/exp_b_robustness.py \
      --batch-dir data/batch_run_results/enterprise_ticket_run --out-dir robustness/results
  uv run python robustness/exp_b_robustness.py \
      --run-partitions run_partitions.csv --out-dir robustness/results

Outputs: same/diff-POV ARI distributions, run-level permutation test, and the
Invariance/Responsiveness Success γ/δ sweep (paper App. B.1.4 definitions).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from loaders import load_batch_run_partitions, load_run_partitions_csv
from metrics_core import (
    group_summary,
    invariance_responsiveness_sweep,
    pairwise_groups_dicts,
    permutation_test_mean_diff_dicts,
)
from report import table_r2


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--batch-dir")
    src.add_argument("--run-partitions")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-overlap", type=int, default=20)
    ap.add_argument("--n-permutations", type=int, default=10_000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_dir:
        runs = load_batch_run_partitions(args.batch_dir)
    else:
        runs = load_run_partitions_csv(args.run_partitions)

    assignments, povs = runs.resolved()
    print(
        f"runs: {len(runs.assignments)} with partitions, {len(assignments)} with resolved POV "
        f"({len(runs.skipped_no_clustering)} without clustering, {len(runs.skipped_no_pov)} without POV)"
    )
    pov_counts: dict[str, int] = {}
    for p in povs.values():
        pov_counts[p] = pov_counts.get(p, 0) + 1
    print("runs per POV:", dict(sorted(pov_counts.items(), key=lambda kv: -kv[1])))
    if len(pov_counts) < 2:
        raise SystemExit("Need at least two distinct POVs for same-vs-different comparison")
    if all(c < 2 for c in pov_counts.values()):
        raise SystemExit("No POV has >= 2 runs; there are no same-POV pairs")

    groups, skipped_pairs = pairwise_groups_dicts(assignments, povs, min_overlap=args.min_overlap)
    summary = group_summary(groups)
    perm = permutation_test_mean_diff_dicts(
        assignments, povs, n_permutations=args.n_permutations, min_overlap=args.min_overlap
    )
    sweep = invariance_responsiveness_sweep(groups)

    md = table_r2(summary, perm, sweep)
    (out_dir / "table_r2.md").write_text(md + "\n", encoding="utf-8")
    (out_dir / "exp_b_details.json").write_text(
        json.dumps(
            {
                "group_summary": summary,
                "permutation_test": perm,
                "sweep": sweep,
                "runs_per_pov": pov_counts,
                "skipped_low_overlap_pairs": len(skipped_pairs),
                "runs_without_clustering": runs.skipped_no_clustering,
                "runs_without_pov": runs.skipped_no_pov,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_dir / "exp_b_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "invariance_success", "responsiveness_success"])
        writer.writeheader()
        writer.writerows(sweep)

    print(f"\nwrote {out_dir}/table_r2.md, exp_b_details.json, exp_b_sweep.csv\n")
    print(md)


if __name__ == "__main__":
    main()
