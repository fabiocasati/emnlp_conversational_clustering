"""Generate Qwen3-Embedding-8b embeddings for Table R1 via the repo's own stack.

Variants:
- base: raw ticket texts (what the backend's cache pre-warm embeds)
- pov_<theme>: texts wrapped in the backend's exact guided-embedding format
  (format_instruction_embedding_input, instruction_kmeans_v1/tools.py:87-88),
  using the official theme guidance from the benchmark spec as the instruction —
  i.e. "stated POV label, zero conversation".

Model: qwen/qwen3-embedding-8b via OpenRouter (the repo's alternate config for
the same model the runs used locally; provider noted in outputs).

Usage (from repo root, needs OPENROUTER_API_KEY in .env):
  uv run python oneshot_baselines/gen_embeddings.py \
      --dataset data/enteprise_dataset_1000/enterprise_ticket_dataset_1000.csv \
      --spec data/final_enterprise_ticket_run/enterprise_ticket_run/_benchmark_specs/enterprise_ticket__all.benchmark.json \
      --povs department issue_type system_name \
      --out-dir oneshot_baselines/embeddings
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loaders import item_id_from_text, load_dataset_csv  # noqa: E402

from conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.config import (  # noqa: E402
    default_instruction_kmeans_v1_openrouter_embedding_model_config,
)
from conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools import (  # noqa: E402
    format_instruction_embedding_input,
)
from conversational_clustering.data_processing.embeddings import (  # noqa: E402
    generate_embeddings_for_texts,
)


class _BatchStall(RuntimeError):
    pass


def _alarm_handler(*_):
    raise _BatchStall("batch did not complete within the stall limit")


def embed_and_save(
    texts: list[str], out_path: Path, cfg, batch_size: int, stall_limit_s: int = 180
) -> None:
    if out_path.exists():
        arr = np.load(out_path)
        if arr.shape[0] == len(texts):
            print(f"{out_path.name}: exists with {arr.shape}, skipping", flush=True)
            return
        raise AssertionError(f"{out_path.name} exists with wrong shape {arr.shape}; delete to regenerate")
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm_handler)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        signal.alarm(stall_limit_s)
        try:
            vectors.extend(
                generate_embeddings_for_texts(texts=chunk, model_config=cfg, batch_size=batch_size)
            )
        finally:
            signal.alarm(0)
        print(
            f"{out_path.name}: {min(start + batch_size, len(texts))}/{len(texts)} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.shape[0] != len(texts):
        raise AssertionError(f"{out_path.name}: got {arr.shape[0]} vectors for {len(texts)} texts")
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise AssertionError(f"{out_path.name}: suspicious shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise AssertionError(f"{out_path.name}: non-finite values in embeddings")
    np.save(out_path, arr)
    print(f"{out_path.name}: {arr.shape} in {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--povs", nargs="+", default=["department", "issue_type", "system_name"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    load_dotenv()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset_csv(args.dataset)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    guidance = {t["theme_id"]: t["guidance"] for t in spec["themes"]}
    missing = [p for p in args.povs if p not in guidance]
    if missing:
        raise SystemExit(f"POVs without spec guidance: {missing}; available: {sorted(guidance)}")

    with (out_dir / "embedding_order.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "item_id"])
        for i, t in enumerate(ds.texts):
            writer.writerow([i, item_id_from_text(t)])

    cfg = default_instruction_kmeans_v1_openrouter_embedding_model_config()
    print(f"model={cfg.model} backend={cfg.backend}", flush=True)

    embed_and_save(ds.texts, out_dir / "emb_qwen3_base.npy", cfg, args.batch_size)
    for pov in args.povs:
        instruction = " ".join(guidance[pov].split())
        wrapped = [
            format_instruction_embedding_input(embedding_instruction=instruction, item_text=t)
            for t in ds.texts
        ]
        embed_and_save(wrapped, out_dir / f"emb_qwen3_pov_{pov}.npy", cfg, args.batch_size)
        (out_dir / f"instruction_{pov}.txt").write_text(instruction + "\n", encoding="utf-8")

    print("done", flush=True)


if __name__ == "__main__":
    main()
