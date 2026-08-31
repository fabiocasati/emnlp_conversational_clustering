"""Experiment A — one-shot baselines vs. all 24 reference partitions (Table R1).

Usage:
  uv run python oneshot_baselines/exp_a_baselines.py --dataset <dataset.csv> \
      [--embeddings <emb.npy> --embeddings-name qwen3_base] \
      [--embeddings <emb2.npy> --embeddings-name qwen3_pov_issue_type] \
      --out-dir oneshot_baselines/results

Embeddings .npy files must be row-aligned with the dataset CSV row order
(the repo embeds items in dataset order; verify with embedding_order.csv if
one is provided). Repeat --embeddings/--embeddings-name pairs for variants.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baselines import load_embedding_matrix, run_baselines, tfidf_svd_matrix
from loaders import load_dataset_csv
from metrics_core import score_against_references
from report import table_r1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embeddings", action="append", default=[])
    ap.add_argument("--embeddings-name", action="append", default=[])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if len(args.embeddings) != len(args.embeddings_name):
        raise SystemExit("--embeddings and --embeddings-name must be paired")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset_csv(args.dataset)
    print(f"dataset: {len(ds.texts)} items, {len(ds.references)} reference partitions")

    representations = {"tfidf": tfidf_svd_matrix(ds.texts)}
    for path, name in zip(args.embeddings, args.embeddings_name):
        representations[name] = load_embedding_matrix(path, expected_rows=len(ds.texts))

    all_scores = []
    for rep_name, matrix in representations.items():
        print(f"running baselines on '{rep_name}' {matrix.shape} ...")
        for pipeline, labels in run_baselines(matrix, rep_name).items():
            all_scores.append(score_against_references(pipeline, labels, ds.references))

    rows = [s.summary_row() for s in all_scores]
    md = table_r1(rows)
    (out_dir / "table_r1.md").write_text(md + "\n", encoding="utf-8")

    # Full per-POV ARI/NMI grid for the record / appendix.
    with (out_dir / "exp_a_full_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "pov", "ari", "nmi"])
        for s in all_scores:
            for pov in s.ari_by_pov:
                writer.writerow([s.pipeline, pov, f"{s.ari_by_pov[pov]:.6f}", f"{s.nmi_by_pov[pov]:.6f}"])

    print(f"\nwrote {out_dir}/table_r1.md and exp_a_full_scores.csv\n")
    print(md)


if __name__ == "__main__":
    main()
