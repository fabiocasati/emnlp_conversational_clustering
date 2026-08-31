"""One-shot clustering baselines for the one-shot baseline analysis (paper App. C).

Representations: TF-IDF(+SVD) built from ticket texts, and optionally
precomputed embeddings (base or POV-prefixed) loaded from .npy.
Algorithms: K-means over a K grid, Ward agglomerative, HDBSCAN.

Pure computation: takes texts / vectors, returns {pipeline_name: labels}.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

KMEANS_K_GRID = (8, 16, 24, 32, 48)
AGGLO_K_GRID = (8, 16, 24, 32, 48)
HDBSCAN_MIN_CLUSTER_SIZES = (5, 10, 20)
SVD_COMPONENTS = 100
SEED = 42


def tfidf_svd_matrix(texts: list[str], n_components: int = SVD_COMPONENTS) -> np.ndarray:
    if not texts:
        raise ValueError("No texts supplied")
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        bad = sum(1 for t in texts if not isinstance(t, str) or not t.strip())
        raise ValueError(f"{bad} empty/non-string texts in input")
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    x = tfidf.fit_transform(texts)
    n_components = min(n_components, x.shape[1] - 1, len(texts) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    reduced = svd.fit_transform(x)
    return normalize(reduced)


def load_embedding_matrix(npy_path: str, expected_rows: int) -> np.ndarray:
    emb = np.load(npy_path)
    if emb.ndim != 2:
        raise ValueError(f"{npy_path}: expected 2-D array, got shape {emb.shape}")
    if emb.shape[0] != expected_rows:
        raise ValueError(
            f"{npy_path}: {emb.shape[0]} rows but {expected_rows} items expected"
        )
    return normalize(emb.astype(np.float32))


def run_baselines(matrix: np.ndarray, representation: str) -> dict[str, np.ndarray]:
    """Run every baseline on a (n_items, dim) matrix.

    Returns {pipeline_name: labels}. HDBSCAN noise points (-1) are kept as a
    singleton-ish 'noise' cluster label, which is the conservative choice for
    ARI (treating noise as its own cluster).
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {matrix.shape}")
    out: dict[str, np.ndarray] = {}
    for k in KMEANS_K_GRID:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        out[f"kmeans_k{k}_{representation}"] = km.fit_predict(matrix)
    for k in AGGLO_K_GRID:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        out[f"ward_k{k}_{representation}"] = agg.fit_predict(matrix)
    for mcs in HDBSCAN_MIN_CLUSTER_SIZES:
        hdb = HDBSCAN(min_cluster_size=mcs)
        labels = hdb.fit_predict(matrix)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            print(
                f"WARNING: hdbscan_mcs{mcs}_{representation} found {n_clusters} "
                "clusters; keeping anyway (will score near 0)"
            )
        out[f"hdbscan_mcs{mcs}_{representation}"] = labels
    return out
