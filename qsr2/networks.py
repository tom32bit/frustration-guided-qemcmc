"""Connectome loading, weighted community detection and ensemble statistics (fixed versions of QSR v1 Phase 1).

Fixes relative to QSR v1 (audit P1-1, P1-1b, P1-3): real Leiden (leidenalg) WITH edge weights, explicit seeds,
vectorised co-assignment / entropy.
"""
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "external", "enigma_hcp")

FILES = {
    "FC100": ("funcMatrix_ctx_schaefer_100.csv", "funcLabels_ctx_schaefer_100.csv"),
    "FC200": ("funcMatrix_ctx_schaefer_200.csv", "funcLabels_ctx_schaefer_200.csv"),
    "FC400": ("funcMatrix_ctx_schaefer_400.csv", "funcLabels_ctx_schaefer_400.csv"),
    "SC100": ("strucMatrix_ctx_schaefer_100.csv", "funcLabels_ctx_schaefer_100.csv"),
    "SC200": ("strucMatrix_ctx_schaefer_200.csv", "funcLabels_ctx_schaefer_200.csv"),
    "SC400": ("strucMatrix_ctx_schaefer_400.csv", "funcLabels_ctx_schaefer_400.csv"),   # added 21 Nov 2026 (E3)
}


def load_connectome(name: str):
    """HCP group-average connectome (ENIGMA toolbox), symmetrised, negatives clipped, zero diagonal.
    Returns (W, labels or None)."""
    mf, lf = FILES[name]
    W = np.loadtxt(os.path.join(DATA_DIR, mf), delimiter=",")
    W = np.clip((W + W.T) / 2, 0, None)
    np.fill_diagonal(W, 0)
    labels = None
    if lf is not None:
        labels = open(os.path.join(DATA_DIR, lf)).read().strip().split(",")
        assert len(labels) == len(W)
    return W, labels


def modularity_matrix(W: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """B^gamma_ij = W_ij - gamma k_i k_j / 2m (diagonal kept; callers zero it for Ising couplings)."""
    k = W.sum(1)
    return W - gamma * np.outer(k, k) / W.sum()


def modularity(W: np.ndarray, labels: np.ndarray, gamma: float = 1.0) -> float:
    """Q = (1/2m) sum_ij B_ij delta(c_i, c_j). VERIFY: equals networkx modularity on the weighted graph."""
    B = modularity_matrix(W, gamma)
    same = labels[:, None] == labels[None, :]
    return float((B * same).sum() / W.sum())


def leiden(W: np.ndarray, gamma: float = 1.0, seed: int = 0) -> np.ndarray:
    """Weighted Leiden (RB configuration model) - the algorithm QSR v1 *claimed* to use."""
    import igraph as ig
    import leidenalg as la
    n = len(W)
    iu = np.triu_indices(n, 1)
    w = W[iu]
    keep = w > 0
    g = ig.Graph(n=n, edges=list(zip(iu[0][keep].tolist(), iu[1][keep].tolist())), directed=False)
    g.es["weight"] = w[keep].tolist()
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                             resolution_parameter=float(gamma), seed=int(seed))
    return np.asarray(part.membership, dtype=int)


def best_partition(W: np.ndarray, gamma: float = 1.0, runs: int = 50) -> np.ndarray:
    parts = [leiden(W, gamma, s) for s in range(runs)]
    return max(parts, key=lambda p: modularity(W, p, gamma))


def coassignment(partitions) -> np.ndarray:
    """p_ij = fraction of partitions with i, j together (QSR v1 Eq. 5, vectorised)."""
    E = np.asarray(partitions)
    P = np.zeros((E.shape[1], E.shape[1]))
    for part in E:
        P += part[:, None] == part[None, :]
    return P / len(E)


def pair_entropy(P: np.ndarray) -> np.ndarray:
    """H_ij = h2(p_ij) in bits (QSR v1 Eq. 6); 0 where p in {0, 1}."""
    H = np.zeros_like(P)
    m = (P > 0) & (P < 1)
    p = P[m]
    H[m] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return H


def network_scale(W: np.ndarray, gamma: float = 1.0) -> float:
    """Global energy scale sigma = std of off-diagonal B entries. Physical temperature: beta = beta_tilde / sigma,
    fixed per NETWORK (not per patch) - this is the 'physically fixed beta' of the G1 gate."""
    B = modularity_matrix(W, gamma)
    return float(np.std(B[np.triu_indices(len(W), 1)]))
