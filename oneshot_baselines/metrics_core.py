"""Format-independent metric helpers for the analysis pipeline.

All functions operate on plain label sequences (one label per item, items
aligned by position). Loading/adaptation from the repo's native artifact
formats lives in loaders.py; this module is pure computation so it can be
tested on fixtures without any repo data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _check_labels(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Label arrays differ in shape: {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"Expected 1-D label arrays, got ndim={a.ndim}")
    if a.size == 0:
        raise ValueError("Empty label arrays")


def ari(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    _check_labels(a, b)
    return float(adjusted_rand_score(a, b))


def nmi(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    _check_labels(a, b)
    return float(normalized_mutual_info_score(a, b))


@dataclass(frozen=True)
class ReferenceScores:
    """Scores of one candidate partition against every reference partition."""

    pipeline: str
    ari_by_pov: dict[str, float]
    nmi_by_pov: dict[str, float]

    @property
    def ari_values(self) -> np.ndarray:
        return np.array(list(self.ari_by_pov.values()))

    @property
    def nmi_values(self) -> np.ndarray:
        return np.array(list(self.nmi_by_pov.values()))

    def summary_row(self) -> dict:
        ari_v, nmi_v = self.ari_values, self.nmi_values
        best_pov = max(self.ari_by_pov, key=self.ari_by_pov.get)
        return {
            "pipeline": self.pipeline,
            "ari_mean": ari_v.mean(),
            "ari_sd": ari_v.std(ddof=1),
            "ari_max": ari_v.max(),
            "nmi_mean": nmi_v.mean(),
            "nmi_sd": nmi_v.std(ddof=1),
            "nmi_max": nmi_v.max(),
            "favored_pov": best_pov,
            "favored_pov_ari": self.ari_by_pov[best_pov],
        }


def score_against_references(
    pipeline_name: str,
    candidate_labels,
    references: dict[str, np.ndarray],
) -> ReferenceScores:
    """Score one candidate partition against all reference partitions."""
    if not references:
        raise ValueError("No reference partitions supplied")
    candidate = np.asarray(candidate_labels)
    return ReferenceScores(
        pipeline=pipeline_name,
        ari_by_pov={pov: ari(candidate, ref) for pov, ref in references.items()},
        nmi_by_pov={pov: nmi(candidate, ref) for pov, ref in references.items()},
    )


def pairwise_ari_matrix(references: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Symmetric pairwise-ARI matrix over named partitions (diagonal = 1)."""
    names = list(references)
    if len(names) < 2:
        raise ValueError("Need at least two partitions for a pairwise matrix")
    n = len(names)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = mat[j, i] = ari(references[names[i]], references[names[j]])
    return names, mat


def offdiag_stats(mat: np.ndarray) -> dict:
    off = mat[~np.eye(mat.shape[0], dtype=bool)]
    return {
        "mean": float(off.mean()),
        "sd": float(off.std(ddof=1)),
        "min": float(off.min()),
        "max": float(off.max()),
    }


# ---------------------------------------------------------------------------
# Experiment B: same-POV vs different-POV pairwise similarities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseGroups:
    """Pairwise similarity values split by whether the two runs share a POV."""

    same_pov: np.ndarray
    diff_pov: np.ndarray

    def __post_init__(self):
        if self.same_pov.size == 0 or self.diff_pov.size == 0:
            raise ValueError(
                f"Both groups must be non-empty (same={self.same_pov.size}, "
                f"diff={self.diff_pov.size})"
            )


def pairwise_groups(
    run_labels: dict[str, np.ndarray],
    run_pov: dict[str, str],
    metric=ari,
) -> PairwiseGroups:
    """All unordered run pairs, split into same-POV / different-POV groups.

    run_labels: run_id -> label array (all aligned to the same item order).
    run_pov:    run_id -> resolved POV name for that run.
    """
    run_ids = sorted(run_labels)
    missing = [r for r in run_ids if r not in run_pov]
    if missing:
        raise ValueError(f"Runs without a POV assignment: {missing[:5]}")
    same, diff = [], []
    for i, ra in enumerate(run_ids):
        for rb in run_ids[i + 1 :]:
            value = metric(run_labels[ra], run_labels[rb])
            (same if run_pov[ra] == run_pov[rb] else diff).append(value)
    return PairwiseGroups(same_pov=np.array(same), diff_pov=np.array(diff))


def pairwise_groups_dicts(
    run_assignments: dict[str, dict],
    run_pov: dict[str, str],
    min_overlap: int = 20,
    metric=ari,
) -> tuple[PairwiseGroups, list[tuple[str, str, int]]]:
    """Like pairwise_groups, but for partitions over possibly different item sets.

    run_assignments: run_id -> {item_id: cluster_label}. Each pair is scored on
    the intersection of item ids; pairs with fewer than min_overlap common
    items are skipped and returned separately so they can be reported.
    """
    run_ids = sorted(run_assignments)
    missing = [r for r in run_ids if r not in run_pov]
    if missing:
        raise ValueError(f"Runs without a POV assignment: {missing[:5]}")
    same, diff, skipped = [], [], []
    for i, ra in enumerate(run_ids):
        for rb in run_ids[i + 1 :]:
            common = sorted(run_assignments[ra].keys() & run_assignments[rb].keys())
            if len(common) < min_overlap:
                skipped.append((ra, rb, len(common)))
                continue
            a = np.array([run_assignments[ra][x] for x in common])
            b = np.array([run_assignments[rb][x] for x in common])
            value = metric(a, b)
            (same if run_pov[ra] == run_pov[rb] else diff).append(value)
    if skipped:
        print(f"WARNING: skipped {len(skipped)} run pairs with < {min_overlap} common items")
    return PairwiseGroups(same_pov=np.array(same), diff_pov=np.array(diff)), skipped


def permutation_test_mean_diff_dicts(
    run_assignments: dict[str, dict],
    run_pov: dict[str, str],
    n_permutations: int = 10_000,
    seed: int = 42,
    min_overlap: int = 20,
    metric=ari,
) -> dict:
    """Permutation test over intersection-scored pairs (see pairwise_groups_dicts).

    Pairs below min_overlap contribute NaN and are excluded from both the
    observed statistic and every permuted statistic, keeping the pair set fixed
    across permutations.
    """
    run_ids = sorted(run_assignments)
    povs = [run_pov[r] for r in run_ids]
    n = len(run_ids)
    sim = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            common = sorted(
                run_assignments[run_ids[i]].keys() & run_assignments[run_ids[j]].keys()
            )
            if len(common) < min_overlap:
                continue
            a = np.array([run_assignments[run_ids[i]][x] for x in common])
            b = np.array([run_assignments[run_ids[j]][x] for x in common])
            sim[i, j] = sim[j, i] = metric(a, b)

    iu = np.triu_indices(n, k=1)
    vals = sim[iu]
    valid = ~np.isnan(vals)
    if not valid.any():
        raise ValueError("No run pair reaches the minimum overlap")

    def mean_diff(pov_seq: list[str]) -> float:
        pov_arr = np.asarray(pov_seq)
        same_mask = (pov_arr[iu[0]] == pov_arr[iu[1]]) & valid
        diff_mask = (pov_arr[iu[0]] != pov_arr[iu[1]]) & valid
        if not same_mask.any() or not diff_mask.any():
            raise ValueError("Permutation produced an empty group")
        return float(vals[same_mask].mean() - vals[diff_mask].mean())

    observed = mean_diff(povs)
    rng = np.random.default_rng(seed)
    count_extreme = 0
    for _ in range(n_permutations):
        perm = list(povs)
        rng.shuffle(perm)
        if abs(mean_diff(perm)) >= abs(observed):
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_permutations + 1)
    pov_arr = np.asarray(povs)
    return {
        "observed_mean_diff": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "n_runs": n,
        "n_same_pairs": int(((pov_arr[iu[0]] == pov_arr[iu[1]]) & valid).sum()),
        "n_diff_pairs": int(((pov_arr[iu[0]] != pov_arr[iu[1]]) & valid).sum()),
        "n_skipped_pairs": int((~valid).sum()),
    }


def permutation_test_from_pair_values(
    pair_values: dict[tuple[str, str], float],
    run_pov: dict[str, str],
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Run-level permutation test given precomputed pair similarities.

    pair_values: {(run_a, run_b): similarity}; missing pairs are excluded from
    both the observed statistic and every permuted statistic.
    """
    run_ids = sorted(run_pov)
    index = {r: i for i, r in enumerate(run_ids)}
    n = len(run_ids)
    sim = np.full((n, n), np.nan)
    for (ra, rb), v in pair_values.items():
        if ra not in index or rb not in index:
            raise ValueError(f"Pair references unknown run: {(ra, rb)}")
        sim[index[ra], index[rb]] = sim[index[rb], index[ra]] = v

    iu = np.triu_indices(n, k=1)
    vals = sim[iu]
    valid = ~np.isnan(vals)
    if not valid.any():
        raise ValueError("No valid pairs")
    povs = [run_pov[r] for r in run_ids]

    def mean_diff(pov_seq: list[str]) -> float:
        pov_arr = np.asarray(pov_seq)
        same_mask = (pov_arr[iu[0]] == pov_arr[iu[1]]) & valid
        diff_mask = (pov_arr[iu[0]] != pov_arr[iu[1]]) & valid
        if not same_mask.any() or not diff_mask.any():
            raise ValueError("Permutation produced an empty group")
        return float(vals[same_mask].mean() - vals[diff_mask].mean())

    observed = mean_diff(povs)
    rng = np.random.default_rng(seed)
    count_extreme = 0
    for _ in range(n_permutations):
        perm = list(povs)
        rng.shuffle(perm)
        if abs(mean_diff(perm)) >= abs(observed):
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_permutations + 1)
    pov_arr = np.asarray(povs)
    return {
        "observed_mean_diff": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "n_runs": n,
        "n_same_pairs": int(((pov_arr[iu[0]] == pov_arr[iu[1]]) & valid).sum()),
        "n_diff_pairs": int(((pov_arr[iu[0]] != pov_arr[iu[1]]) & valid).sum()),
        "n_missing_pairs": int((~valid).sum()),
    }


def permutation_test_mean_diff(
    run_labels: dict[str, np.ndarray],
    run_pov: dict[str, str],
    n_permutations: int = 10_000,
    seed: int = 42,
    metric=ari,
) -> dict:
    """Two-sided permutation test on mean(same-POV ARI) - mean(diff-POV ARI).

    Permutes POV assignments across *runs* (not across pairs): pair values are
    not independent because each run participates in many pairs, so shuffling
    run-level POV labels is the exchangeable unit.
    """
    run_ids = sorted(run_labels)
    povs = [run_pov[r] for r in run_ids]

    # Precompute the full pairwise similarity matrix once.
    n = len(run_ids)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim[i, j] = sim[j, i] = metric(run_labels[run_ids[i]], run_labels[run_ids[j]])

    iu = np.triu_indices(n, k=1)

    def mean_diff(pov_seq: list[str]) -> float:
        pov_arr = np.asarray(pov_seq)
        same_mask = (pov_arr[iu[0]] == pov_arr[iu[1]])
        vals = sim[iu]
        if not same_mask.any() or same_mask.all():
            raise ValueError("Permutation produced an empty group")
        return float(vals[same_mask].mean() - vals[~same_mask].mean())

    observed = mean_diff(povs)
    rng = np.random.default_rng(seed)
    count_extreme = 0
    for _ in range(n_permutations):
        perm = list(povs)
        rng.shuffle(perm)
        if abs(mean_diff(perm)) >= abs(observed):
            count_extreme += 1
    # add-one smoothing: permutation p-values cannot be exactly 0
    p_value = (count_extreme + 1) / (n_permutations + 1)
    return {
        "observed_mean_diff": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "n_runs": n,
        "n_same_pairs": int((np.asarray(povs)[iu[0]] == np.asarray(povs)[iu[1]]).sum()),
        "n_diff_pairs": int((np.asarray(povs)[iu[0]] != np.asarray(povs)[iu[1]]).sum()),
    }


def invariance_responsiveness_sweep(
    groups: PairwiseGroups,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """Invariance / Responsiveness Success rates across a γ/δ threshold sweep.

    Paper App. B.1.4 / Table 5:
      Invariance Success     = 1[Sim(Pa, Pb) >= γ] for same-intent pairs
      Responsiveness Success = 1[Sim(Pa, Pb) <= δ] for different-intent pairs
    Reported per threshold value, with γ = δ = t on a shared grid so the two
    curves can be read against each other.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for t in thresholds:
        rows.append(
            {
                "threshold": float(t),
                "invariance_success": float((groups.same_pov >= t).mean()),
                "responsiveness_success": float((groups.diff_pov <= t).mean()),
            }
        )
    return rows


def group_summary(groups: PairwiseGroups) -> dict:
    def stats(x: np.ndarray) -> dict:
        return {
            "n_pairs": int(x.size),
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
            "median": float(np.median(x)),
        }

    return {"same_pov": stats(groups.same_pov), "diff_pov": stats(groups.diff_pov)}
