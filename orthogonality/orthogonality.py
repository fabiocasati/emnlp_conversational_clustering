"""Bonus — 24x24 pairwise-ARI matrix over the reference partitions.

Usage:
  uv run python orthogonality/orthogonality.py --dataset <dataset.csv> --out-dir orthogonality/results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from loaders import load_dataset_csv
from metrics_core import offdiag_stats, pairwise_ari_matrix
from report import orthogonality_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset_csv(args.dataset)
    names, mat = pairwise_ari_matrix(ds.references)
    stats = offdiag_stats(mat)

    with (out_dir / "orthogonality_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pov", *names])
        for i, name in enumerate(names):
            writer.writerow([name, *(f"{v:.6f}" for v in mat[i])])

    md = orthogonality_report(names, mat, stats)
    (out_dir / "orthogonality_report.md").write_text(md + "\n", encoding="utf-8")
    print(f"wrote {out_dir}/orthogonality_matrix.csv and orthogonality_report.md\n")
    print(md)


if __name__ == "__main__":
    main()
