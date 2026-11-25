"""Partition-space Markov-chain runs for E4-B (docs/E4_whole_cortex_protocol_2026-11-22.md).

run_chains(kind, cdf, Er, beta, n, s0, burn, steps, seeds) runs one replicate per seed (parallel over replicates).
Each replicate seeds Numba's generator at its start and draws only inside its own iteration, so the output does not
depend on thread scheduling (tests/test_e4.py checks 1 thread == several threads). Every replicate starts at s0,
discards `burn` steps, then records its occupation histogram over `steps` steps; the estimate of a partition
observable f is hist @ f / steps. States are partition representatives s < M = 2^(n-1) (top bit 0).

  kind 0  MH with a dense symmetric lumped proposal: c ~ cdf[s], accept with min(1, exp(-beta (Er[c] - Er[s])))
  kind 1  dense transition kernel (lumped Swendsen-Wang / Wolff): s -> c ~ cdf[s]
  kind 2  local MH: flip one uniformly chosen node (mapped back to the representative), accept as above
  kind 3  label-swap MH: with probability 1/2 stay (flip-all is the identity on partitions), else a local move
  kind 4  uniform MH: c uniform on the M partitions, accept as above
  kind 5  i.i.d. Gibbs draws (heat-bath): s ~ cdf[0] each step (cdf[0] = CDF of pi); no acceptance
cdf rows must be non-decreasing with last entry exactly 1.0 (use row_cdf). Kinds 2-4 ignore cdf.
"""
import numpy as np
from numba import njit, prange

KINDS = {"quantum": 0, "qaoa_a0": 0, "sw": 1, "wolff": 1, "local": 2, "label_swap": 3, "uniform": 4, "heat_bath": 5}


def row_cdf(P):
    """Row-wise CDF of a stochastic matrix (or of a single distribution), normalised so the last entry is 1.0."""
    C = np.cumsum(np.atleast_2d(np.asarray(P, float)), axis=1)
    C /= C[:, -1:]
    C[:, -1] = 1.0
    return np.ascontiguousarray(C)


@njit(cache=True)
def _draw(row, u):
    lo, hi = 0, row.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if row[mid] > u:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def _accept(Er, beta, s, c):
    if c == s:
        return s
    dE = Er[c] - Er[s]
    if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
        return c
    return s


@njit(cache=True)
def _one(kind, cdf, Er, beta, n, s0, burn, steps, seed, hist):
    np.random.seed(seed)
    M = Er.shape[0]
    full = 2 * M - 1
    s = s0
    for t in range(burn + steps):
        if kind == 0:
            s = _accept(Er, beta, s, _draw(cdf[s], np.random.random()))
        elif kind == 1:
            s = _draw(cdf[s], np.random.random())
        elif kind == 2 or kind == 3:
            if kind == 3 and np.random.random() < 0.5:
                pass
            else:
                c = s ^ (1 << np.random.randint(0, n))
                if c >= M:
                    c = c ^ full
                s = _accept(Er, beta, s, c)
        elif kind == 4:
            s = _accept(Er, beta, s, np.random.randint(0, M))
        else:
            s = _draw(cdf[0], np.random.random())
        if t >= burn:
            hist[s] += 1


@njit(parallel=True, cache=True)
def run_chains(kind, cdf, Er, beta, n, s0, burn, steps, seeds):
    R = seeds.shape[0]
    H = np.zeros((R, Er.shape[0]), np.int64)
    for r in prange(R):
        _one(kind, cdf, Er, beta, n, s0, burn, steps, seeds[r], H[r])
    return H
