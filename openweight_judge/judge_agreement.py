"""Open-weight judge agreement analysis (paper App. B.1.6).

Compares the final-clustering-quality evaluation judged by the closed model
(gpt-5.4-mini, the paper's judge) against the same saved artifacts re-judged
by the pinned open-weight model (qwen/qwen3-235b-a22b-2507, temperature 0).

Inputs (details JSONL written by the repo's evaluator):
  closed: openweight_judge/inputs/closed_judge_details/
  open:   openweight_judge/results/

Pairing is exact: (config_slug, run_id) for partitions; cluster_name within a
partition for per-cluster judgments; (left, right) cluster names for
distinctiveness pairs. Only artifacts judged successfully by BOTH judges are
compared; the counts of everything excluded are printed (fail loud).

Outputs: openweight_judge/results/judge_agreement.json + stdout table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

REPO = Path(__file__).resolve().parents[1]
CLOSED_DIR = REPO / "openweight_judge/inputs/closed_judge_details"
OPEN_DIR = REPO / "openweight_judge/results/evaluation_report_details"
OUT_PATH = REPO / "openweight_judge/results/judge_agreement.json"

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


def load_details(path: Path) -> tuple[dict[tuple[str, str], dict], int]:
    """Return (successful records keyed by artifact, count of failed artifacts).

    Retry passes append a fresh row for the same artifact, so the LAST row per
    key wins. Rows with status == "error" (empty metrics) are failures.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    records: dict[tuple[str, str], dict] = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        records[(d["config_slug"], d["run_id"])] = d  # last row per key wins
    ok = {k: d for k, d in records.items() if d.get("status") != "error" and d["metrics"]}
    return ok, len(records) - len(ok)


def paired_bootstrap_ci(diffs: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    closed_csq, closed_csq_failed = load_details(CLOSED_DIR / "cluster_semantic_quality.jsonl")
    open_csq, open_csq_failed = load_details(OPEN_DIR / "cluster_semantic_quality.jsonl")
    closed_tcq, closed_tcq_failed = load_details(CLOSED_DIR / "topic_conditioned_quality.jsonl")
    open_tcq, open_tcq_failed = load_details(OPEN_DIR / "topic_conditioned_quality.jsonl")

    out: dict = {"inputs": {
        "closed_csq_ok": len(closed_csq), "closed_csq_failed": closed_csq_failed,
        "open_csq_ok": len(open_csq), "open_csq_failed": open_csq_failed,
        "closed_tcq_ok": len(closed_tcq), "closed_tcq_failed": closed_tcq_failed,
        "open_tcq_ok": len(open_tcq), "open_tcq_failed": open_tcq_failed,
    }}

    # ---------- cluster_semantic_quality ----------
    common = sorted(set(closed_csq) & set(open_csq))
    if not common:
        raise SystemExit("no paired cluster_semantic_quality artifacts")
    out["csq_paired_partitions"] = len(common)
    out["csq_only_closed"] = len(set(closed_csq) - set(open_csq))
    out["csq_only_open"] = len(set(open_csq) - set(closed_csq))

    coh_c, coh_o, sup_c, sup_o = [], [], [], []
    dis_c, dis_o = [], []
    cl_scores_c, cl_scores_o, cl_sup_c, cl_sup_o = [], [], [], []
    pair_flags_c, pair_flags_o = [], []
    unmatched_clusters = 0
    for key in common:
        c, o = closed_csq[key], open_csq[key]
        coh_c.append(c["metrics"]["average_cluster_coherence"])
        coh_o.append(o["metrics"]["average_cluster_coherence"])
        sup_c.append(c["metrics"]["supported_cluster_rate"])
        sup_o.append(o["metrics"]["supported_cluster_rate"])
        dis_c.append(c["metrics"]["cluster_distinctiveness"])
        dis_o.append(o["metrics"]["cluster_distinctiveness"])
        cj_c = {j["cluster_name"]: j for j in c["details"]["cluster_judgments"]}
        cj_o = {j["cluster_name"]: j for j in o["details"]["cluster_judgments"]}
        names = set(cj_c) & set(cj_o)
        unmatched_clusters += len(set(cj_c) ^ set(cj_o))
        for name in names:
            cl_scores_c.append(cj_c[name]["score"])
            cl_scores_o.append(cj_o[name]["score"])
            cl_sup_c.append(bool(cj_c[name]["supported_by_guidance"]))
            cl_sup_o.append(bool(cj_o[name]["supported_by_guidance"]))
        dp_c = {(p["left_cluster_name"], p["right_cluster_name"]): p
                for p in c["details"]["distinctiveness_pairs"]}
        dp_o = {(p["left_cluster_name"], p["right_cluster_name"]): p
                for p in o["details"]["distinctiveness_pairs"]}
        for pk in set(dp_c) & set(dp_o):
            pair_flags_c.append(bool(dp_c[pk]["overlapping_or_redundant"]))
            pair_flags_o.append(bool(dp_o[pk]["overlapping_or_redundant"]))
    if unmatched_clusters:
        print(f"WARNING: {unmatched_clusters} cluster names unmatched across judges", file=sys.stderr)

    coh_c_arr, coh_o_arr = np.array(coh_c), np.array(coh_o)
    sup_c_arr, sup_o_arr = np.array(sup_c), np.array(sup_o)

    out["coherence"] = {
        "closed_mean": float(coh_c_arr.mean()), "open_mean": float(coh_o_arr.mean()),
        "pearson_r": float(pearsonr(coh_c_arr, coh_o_arr).statistic),
        "spearman_rho": float(spearmanr(coh_c_arr, coh_o_arr).statistic),
        "mean_abs_diff": float(np.abs(coh_c_arr - coh_o_arr).mean()),
        "paired_diff_mean": float((coh_o_arr - coh_c_arr).mean()),
        "paired_diff_ci95": paired_bootstrap_ci(coh_o_arr - coh_c_arr, BOOTSTRAP_N, BOOTSTRAP_SEED),
        "cluster_level_n": len(cl_scores_c),
        "cluster_level_spearman": float(spearmanr(cl_scores_c, cl_scores_o).statistic),
    }
    out["supported_cluster_rate"] = {
        "closed_macro": float(sup_c_arr.mean()), "open_macro": float(sup_o_arr.mean()),
        "paired_diff_mean": float((sup_o_arr - sup_c_arr).mean()),
        "paired_diff_ci95": paired_bootstrap_ci(sup_o_arr - sup_c_arr, BOOTSTRAP_N, BOOTSTRAP_SEED),
        "cluster_level_n": len(cl_sup_c),
        "cluster_level_kappa": float(cohen_kappa_score(cl_sup_c, cl_sup_o)),
        "cluster_level_raw_agreement": float(np.mean(np.array(cl_sup_c) == np.array(cl_sup_o))),
    }
    dis_c_arr, dis_o_arr = np.array(dis_c), np.array(dis_o)
    out["distinctiveness_score"] = {
        "closed_mean": float(dis_c_arr.mean()), "open_mean": float(dis_o_arr.mean()),
        "pearson_r": float(pearsonr(dis_c_arr, dis_o_arr).statistic),
        "spearman_rho": float(spearmanr(dis_c_arr, dis_o_arr).statistic),
        "paired_diff_mean": float((dis_o_arr - dis_c_arr).mean()),
        "paired_diff_ci95": paired_bootstrap_ci(dis_o_arr - dis_c_arr, BOOTSTRAP_N, BOOTSTRAP_SEED),
    }
    out["distinctiveness_overlap_flag"] = {
        "pair_level_n": len(pair_flags_c),
        "kappa": float(cohen_kappa_score(pair_flags_c, pair_flags_o)) if pair_flags_c else None,
        "raw_agreement": float(np.mean(np.array(pair_flags_c) == np.array(pair_flags_o))) if pair_flags_c else None,
    }

    # ---------- topic_conditioned_quality (theme resolution + conditioned metrics) ----------
    common_t = sorted(set(closed_tcq) & set(open_tcq))
    out["tcq_paired_partitions"] = len(common_t)
    theme_match = []
    cond = {m: ([], []) for m in ("cond_ari", "cond_nmi", "cond_vi")}
    for key in common_t:
        c, o = closed_tcq[key], open_tcq[key]
        theme_match.append(c["details"]["resolved_theme_id"] == o["details"]["resolved_theme_id"])
        for m, (lc, lo) in cond.items():
            lc.append(c["metrics"][m])
            lo.append(o["metrics"][m])
    if common_t:
        out["theme_resolution"] = {"same_resolved_theme_rate": float(np.mean(theme_match))}
        for m, (lc, lo) in cond.items():
            a_c, a_o = np.array(lc), np.array(lo)
            out["theme_resolution"][m] = {
                "closed_mean": float(a_c.mean()), "open_mean": float(a_o.mean()),
                "pearson_r": float(pearsonr(a_c, a_o).statistic),
                "paired_diff_mean": float((a_o - a_c).mean()),
                "paired_diff_ci95": paired_bootstrap_ci(a_o - a_c, BOOTSTRAP_N, BOOTSTRAP_SEED),
            }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwritten: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
