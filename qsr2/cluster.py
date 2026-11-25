"""Exact transition matrices of cluster Monte Carlo for field-free Ising models E(z) = -sum_{i<j} J_ij z_i z_j
(mixed-sign J allowed), plus the Houdayer cluster (used to show it is trivial on dense couplings).

Bonds: given state z, each SATISFIED coupling (J_ij z_i z_j > 0) is opened with p_ij = 1 - exp(-2 beta |J_ij|);
unsatisfied couplings are never opened. This satisfies detailed balance for any sign pattern of J.

Exact matrices via subset dynamic programming (vectorised over all 2^n states s):
  w_s(X)        = sum_{i<j in X} log(1 - p_ij(s))          (log-probability that no bond is open inside X)
  cut_s(T, U)   = w_s(T u U) - w_s(T) - w_s(U)            (log-prob. that no bond joins disjoint T and U)
  g_s(T)        = P(T is connected using bonds inside T)
                = 1 - sum_{U strict subset of T, min(T) in U} g_s(U) exp(cut_s(U, T \\ U))
  h_s(S)        = E[2^{-C(S)}]  (C = number of clusters of the bond graph restricted to S), h_s(empty) = 1
                = sum_{T subset of S, min(S) in T} (1/2) g_s(T) exp(cut_s(T, S \\ T)) h_s(S \\ T)
Swendsen-Wang (flip each cluster with prob 1/2):  P(s -> s xor A) = exp(cut_s(A, A^c)) h_s(A) h_s(A^c)
Wolff (uniform seed, flip its cluster):            P(s -> s xor A) = |A|/n g_s(A) exp(cut_s(A, A^c)),  A != 0
Cost O(n^2 2^n + 3^n 2^n): fine for n <= 10.  VERIFY: tests compare with brute-force bond enumeration.
"""
import numpy as np

from .ising import spins


def _tables(J, beta):
    """Tables in (mask x state) layout so that every DP step is a contiguous vector operation over states."""
    n = len(J)
    N = 2 ** n
    Z = spins(n).astype(float)                                   # (N, n), row s <-> state s
    sat = (J[None, :, :] * Z[:, :, None] * Z[:, None, :]) > 0
    l = np.where(sat, -2.0 * beta * np.abs(J)[None], 0.0)        # log(1 - p) on satisfied couplings
    M = ((np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)  # mask X <-> member indicator
    w = np.zeros((N, N))
    for X in range(1, N):
        k = X.bit_length() - 1
        Xp = X ^ (1 << k)
        w[X] = w[Xp] + l[:, k, :] @ M[Xp]
    g = np.zeros((N, N))
    for T in range(1, N):
        low = T & -T
        rest = T ^ low
        if rest == 0:
            g[T] = 1.0
            continue
        acc = np.zeros(N)
        wT = w[T]
        sub = (rest - 1) & rest                                  # all submasks of rest except rest itself
        while True:
            U = low | sub
            acc += g[U] * np.exp(wT - w[U] - w[T ^ U])
            if sub == 0:
                break
            sub = (sub - 1) & rest
        g[T] = np.clip(1.0 - acc, 0.0, 1.0)
    return n, N, w, g


def _h_table(N, w, g):
    h = np.zeros((N, N))
    h[0] = 1.0
    for S in range(1, N):
        low = S & -S
        rest = S ^ low
        acc = np.zeros(N)
        wS = w[S]
        sub = rest
        while True:
            U = low | sub
            acc += g[U] * np.exp(wS - w[U] - w[S ^ U]) * h[S ^ U]
            if sub == 0:
                break
            sub = (sub - 1) & rest
        h[S] = 0.5 * acc
    return h


def _tables_fast(J, beta):
    try:
        from .cluster_numba import h_table, tables
    except ImportError:
        n, N, w, g = _tables(J, beta)
        return n, N, w, g, _h_table(N, w, g)
    n = len(J)
    N = 2 ** n
    Z = spins(n).astype(float)
    sat = (J[None, :, :] * Z[:, :, None] * Z[:, None, :]) > 0
    l = np.ascontiguousarray(np.where(sat, -2.0 * beta * np.abs(J)[None], 0.0))
    w, g = tables(l, n)
    return n, N, w, g, h_table(w, g)


def sw_wolff_transitions(J, beta, use_numba=True):
    """(Swendsen-Wang, Wolff) exact transition matrices from one shared set of tables."""
    if use_numba:
        n, N, w, g, h = _tables_fast(J, beta)
    else:
        n, N, w, g = _tables(J, beta)
        h = _h_table(N, w, g)
    full = N - 1
    idx = np.arange(N)
    SW = np.zeros((N, N))
    WO = np.zeros((N, N))
    for A in range(N):
        cut = np.exp(w[full] - w[A] - w[full ^ A])
        SW[idx, idx ^ A] = cut * h[A] * h[full ^ A]
        if A:
            WO[idx, idx ^ A] = bin(A).count("1") / n * g[A] * cut
    return SW, WO


def sw_transition(J, beta):
    """Exact Swendsen-Wang transition matrix (rows sum to 1, reversible w.r.t. the Gibbs distribution)."""
    return sw_wolff_transitions(J, beta)[0]


def wolff_transition(J, beta):
    """Exact Wolff single-cluster transition matrix."""
    return sw_wolff_transitions(J, beta)[1]


def houdayer_cluster(J, z1, z2, seed):
    """Houdayer cluster: connected component of `seed` among sites with z1_i != z2_i, connectivity J_ij != 0.
    On couplings with every off-diagonal entry non-zero this is the whole set {i : z1_i != z2_i}, so flipping it
    in both replicas maps (z1, z2) -> (z2, z1): a same-temperature replica exchange (no effect on estimators)."""
    diff = set(np.where(np.asarray(z1) != np.asarray(z2))[0].tolist())
    if seed not in diff:
        return set()
    seen, stack = {seed}, [seed]
    while stack:
        i = stack.pop()
        for j in list(diff - seen):
            if J[i, j] != 0:
                seen.add(j)
                stack.append(j)
    return seen


def offdiag_density(J):
    iu = np.triu_indices(len(J), 1)
    return float(np.mean(J[iu] != 0))
