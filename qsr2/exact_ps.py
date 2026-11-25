"""Exact partition-space quantities for E3c (field-free patches, n <= 13), computed directly on the 2^(n-1)
bipartitions so that no 2^n x 2^n matrix is formed (n = 13: 4096 x 4096 instead of 8192 x 8192).

Identity used throughout: for field-free E, the lumped Metropolis kernel equals Metropolis applied to the lumped
proposal Qbar[s, c] = Q[s, c] + Q[s, NOT c] on representatives s, c < 2^(n-1) (E(NOT c) = E(c)).
Lumped proposals: local = (sum_k X_k on partition space)/n; label-swap = local/2 + identity/2 (the flip-all move is
the identity on partitions); uniform = 1/2^(n-1); Layden quench = proposals.quench_mean_sectors(full=False);
QAOA-MC at one angle = lumped |U|^2 from the 2^(n-1) representative columns. Swendsen-Wang and Wolff come from the
parallel lumped kernel cluster_numba.sw_wolff_lumped.
Output keys match exact.compute_exact (b<bt>_<chain>_lumped, _s2mean, _s2max, b<bt>_varmean), so the E3 analysis
definitions apply unchanged. VERIFY (tests/test_exact_ps.py): equal to exact.compute_exact at n = 5-8.
"""
import time

import numpy as np

from . import ising, mcmc, proposals
from .ising import spins

CHAINS = ("local", "label_swap", "uniform", "quantum", "qaoa_a0", "sw", "wolff")


def lumped_local(n):
    return proposals._sector_mixers(n)[0] / n


def qaoa_lumped(E, alpha, b1, b2, g1, block=1024):
    """Partition-space QAOA-MC proposal Qbar[s, c] (rows = representative start states), in column blocks."""
    N = len(E)
    M = N // 2
    ph = np.exp(-1j * g1 * alpha * E)[:, None]
    Q = np.empty((M, M))
    r = np.arange(M)
    for c0 in range(0, M, block):
        cols = np.arange(c0, min(M, c0 + block))
        A = np.zeros((N, len(cols)), dtype=complex)
        A[cols, np.arange(len(cols))] = 1.0
        A = proposals._apply_rx_layer(A, b1)
        A = proposals._apply_rx_layer(ph * A, b2)
        A = proposals._apply_rx_layer(ph * A, b1)
        P = np.abs(A) ** 2                                       # P[x', j] = prob(start cols[j] -> x')
        Q[cols] = (P[r] + P[r ^ (N - 1)]).T
        del A, P
    return Q


def cluster_lumped(J, beta):
    from .cluster_numba import sw_wolff_lumped
    n = len(J)
    M = 2 ** (n - 1)
    Z = spins(n)[:M].astype(float)
    sat = (J[None, :, :] * Z[:, :, None] * Z[:, None, :]) > 0
    l = np.ascontiguousarray(np.where(sat, -2.0 * beta * np.abs(J)[None], 0.0))
    pc = np.array([bin(A).count("1") for A in range(2 * M)], dtype=np.float64)
    return sw_wolff_lumped(l, n, pc)


def _record(row, key, Pbar, lp, Fobs, asym):
    ev = mcmc._sym_eigs(Pbar, lp)
    row[f"{key}_lumped"] = float(max(1.0 - max(abs(ev[-2]), abs(ev[0])), 0.0))
    if asym:
        s2, var = mcmc.asymptotic_variances(Pbar, lp, Fobs)
        row[f"{key}_s2mean"] = float(np.mean(s2))
        row[f"{key}_s2max"] = float(np.max(s2))
        return float(np.mean(var))
    return None


def compute_ps(J, sigma, betas_tilde=(1.0,), chains=CHAINS, asym=True, qaoa_angle=0):
    t0 = time.time()
    n = len(J)
    M = 2 ** (n - 1)
    E = ising.energies(J)
    Er = E[:M]
    Fobs = mcmc.coassignment_observables(n) if asym else None
    row = {"n": n, "timing": {}}
    betas = {bt: bt / sigma for bt in betas_tilde}
    props = {}
    ts = time.time()
    if "local" in chains or "label_swap" in chains:
        loc = lumped_local(n)
        if "local" in chains:
            props["local"] = loc
        if "label_swap" in chains:
            props["label_swap"] = 0.5 * loc + 0.5 * np.eye(M)
    if "uniform" in chains:
        props["uniform"] = np.full((M, M), 1.0 / M)
    row["timing"]["classical_props"] = time.time() - ts
    if "quantum" in chains:
        ts = time.time()
        props["quantum"] = proposals.quench_mean_sectors(E, J, full=False)[0]
        row["timing"]["quantum_prop"] = time.time() - ts
    if "qaoa_a0" in chains:
        ts = time.time()
        a = proposals.layden_alpha(J, np.zeros(n))
        props["qaoa_a0"] = qaoa_lumped(E, a, *proposals.QAOA_GRID[qaoa_angle])
        row["timing"]["qaoa_prop"] = time.time() - ts
    for bt, beta in betas.items():
        lp = -beta * (Er - E.min())
        ts = time.time()
        for name, Q in props.items():
            var = _record(row, f"b{bt}_{name}", mcmc.mh_matrix(Q, Er, beta), lp, Fobs, asym)
            if var is not None:
                row[f"b{bt}_varmean"] = var
        row["timing"][f"mh_b{bt}"] = time.time() - ts
        if "sw" in chains or "wolff" in chains:
            ts = time.time()
            SW, WO = cluster_lumped(J, beta)
            row["timing"][f"cluster_tables_b{bt}"] = time.time() - ts
            ts = time.time()
            for name, P in (("sw", SW), ("wolff", WO)):
                if name in chains:
                    _record(row, f"b{bt}_{name}", P, lp, Fobs, asym)
            del SW, WO
            row["timing"][f"cluster_eigs_b{bt}"] = time.time() - ts
    row["timing"]["total"] = time.time() - t0
    return row
