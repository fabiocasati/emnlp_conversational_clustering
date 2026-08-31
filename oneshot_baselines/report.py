"""Render markdown result tables (R1, R2, orthogonality) from results."""

from __future__ import annotations

import numpy as np


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def table_r1(summary_rows: list[dict]) -> str:
    """Experiment A: one-shot baselines vs all reference partitions."""
    if not summary_rows:
        raise ValueError("No summary rows for Table R1")
    lines = [
        "| Pipeline | ARI mean ± sd | ARI max | NMI mean ± sd | NMI max | Favored POV (ARI) |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['pipeline']} "
            f"| {_fmt(r['ari_mean'])} ± {_fmt(r['ari_sd'])} "
            f"| {_fmt(r['ari_max'])} "
            f"| {_fmt(r['nmi_mean'])} ± {_fmt(r['nmi_sd'])} "
            f"| {_fmt(r['nmi_max'])} "
            f"| {r['favored_pov']} ({_fmt(r['favored_pov_ari'])}) |"
        )
    return "\n".join(lines)


def table_r2(
    summary: dict,
    perm: dict,
    sweep_rows: list[dict],
    sweep_thresholds_to_show: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
) -> str:
    """Experiment B: same/diff POV distributions, permutation test, γ/δ sweep."""
    s, d = summary["same_pov"], summary["diff_pov"]
    head = [
        "**Same-POV vs. different-POV final-partition ARI**",
        "",
        "| Group | n pairs | mean | sd | median |",
        "|---|---|---|---|---|",
        f"| Same POV | {s['n_pairs']} | {_fmt(s['mean'])} | {_fmt(s['sd'])} | {_fmt(s['median'])} |",
        f"| Different POV | {d['n_pairs']} | {_fmt(d['mean'])} | {_fmt(d['sd'])} | {_fmt(d['median'])} |",
        "",
        f"Permutation test (POV labels shuffled across runs, "
        f"{perm['n_permutations']:,} permutations): observed mean difference "
        f"{_fmt(perm['observed_mean_diff'])}, two-sided p = {_fmt(perm['p_value'], 4)}.",
        "",
        "**Invariance / Responsiveness Success across the γ/δ sweep (App. B.1.4)**",
        "",
        "| γ = δ | Invariance Success | Responsiveness Success |",
        "|---|---|---|",
    ]
    shown = 0
    for row in sweep_rows:
        if any(abs(row["threshold"] - t) < 1e-9 for t in sweep_thresholds_to_show):
            head.append(
                f"| {row['threshold']:.2f} | {_fmt(row['invariance_success'])} "
                f"| {_fmt(row['responsiveness_success'])} |"
            )
            shown += 1
    if shown == 0:
        raise ValueError("No sweep rows matched the requested thresholds")
    return "\n".join(head)


def orthogonality_report(names: list[str], mat: np.ndarray, stats: dict) -> str:
    lines = [
        f"**Pairwise ARI over the {len(names)} reference partitions** — "
        f"off-diagonal mean {_fmt(stats['mean'])} ± {_fmt(stats['sd'])}, "
        f"min {_fmt(stats['min'])}, max {_fmt(stats['max'])}.",
        "",
        "Most-correlated pairs (|ARI| descending):",
    ]
    n = len(names)
    pairs = [
        (abs(mat[i, j]), names[i], names[j], mat[i, j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    for _, a, b, v in sorted(pairs, reverse=True)[:8]:
        lines.append(f"- {a} × {b}: {_fmt(v)}")
    return "\n".join(lines)
