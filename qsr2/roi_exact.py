"""Exact per-ROI quantities for E4 (docs/E4_whole_cortex_protocol_2026-11-22.md), on partition space (field-free ROI
models, n <= 13; 2^(n-1) bipartitions).

For each inverse temperature beta = beta_tilde / sigma_ROI:
  - Gibbs co-assignment probabilities p_ij = (1 + E[z_i z_j]) / 2 for every pair (np.triu_indices order),
    their Gibbs variances var_ij = Var(z_i z_j), and the node uncertainty u_i = mean_{j != i} h2(p_ij) in bits;
  - for every chain: the absolute spectral gap and the exact asymptotic variance s2_ij of EVERY co-assignment
    estimator (z units; the estimator of p_ij has asymptotic variance s2_ij / 4).

Chains (all reuse the validated partition-space kernels of exact_ps / mcmc):
  local, label_swap, uniform            Metropolis with the lumped classical proposals
  quantum                               Metropolis with the untuned Layden quench (G1 grid average)
  qaoa_a0                               Metropolis with QAOA-MC at angle 0 (the hardware circuit)
  quantum_noisy_<eps>                   Metropolis with the mixture proposal (1 - eps) Q_quench + eps Q_uniform, a
                                        depolarising-noise model. For symmetric proposals the MH kernel is linear in
                                        the proposal, so this kernel is exactly (1 - eps) P_quench + eps P_uniform
  sw, wolff                             lumped Swendsen-Wang / Wolff kernels (cluster_numba.sw_wolff_lumped)
  heat_bath                             i.i.d. Gibbs draws, each counted as 2^(n-1) proposals: s2 = var * 2^(n-1)
VERIFY (tests/test_e4.py): means / maxima of s2 equal exact_ps.compute_ps; p_ij equals brute-force enumeration over
all 2^n labelled states; the mixture identity; heat-bath accounting.
"""
import time

import numpy as np

from . import ising, mcmc, proposals
from .exact_ps import cluster_lumped, lumped_local, qaoa_lumped

NOISE_EPS = (0.2, 0.5)
CLASSICAL = ("local", "label_swap", "uniform", "sw", "wolff", "heat_bath")


def gibbs(Er, beta):
    lp = -beta * (Er - Er.min())
    pi = np.exp(lp - lp.max())
    return lp, pi / pi.sum()


def coassignment(pi, Fobs):
    """(p_ij, Var_pi(z_i z_j)) for every pair."""
    m = pi @ Fobs
    return (1.0 + m) / 2.0, pi @ (Fobs - m) ** 2


def node_uncertainty(p, n):
    """u_i = mean over j != i of the binary entropy h2(p_ij), in bits (0 = certain, 1 = maximally uncertain)."""
    iu = np.triu_indices(n, 1)
    P = np.full((n, n), 0.5)
    P[iu] = p
    P[(iu[1], iu[0])] = p
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(P * np.log2(P) + (1 - P) * np.log2(1 - P))
    H = np.nan_to_num(H, nan=0.0)
    np.fill_diagonal(H, 0.0)
    return H.sum(1) / (n - 1)


def lumped_proposals(J, chains=("local", "label_swap", "uniform", "quantum", "qaoa_a0")):
    """Partition-space proposal matrices (M x M, symmetric, rows sum to 1) for the proposal-based chains."""
    n = len(J)
    M = 2 ** (n - 1)
    E = ising.energies(J)
    Q = {}
    if "local" in chains or "label_swap" in chains:
        loc = lumped_local(n)
        if "local" in chains:
            Q["local"] = loc
        if "label_swap" in chains:
            Q["label_swap"] = 0.5 * loc + 0.5 * np.eye(M)
    if "uniform" in chains:
        Q["uniform"] = np.full((M, M), 1.0 / M)
    if "quantum" in chains:
        Q["quantum"] = proposals.quench_mean_sectors(E, J, full=False)[0]
    if "qaoa_a0" in chains:
        Q["qaoa_a0"] = qaoa_lumped(E, proposals.layden_alpha(J, np.zeros(n)), *proposals.QAOA_GRID[0])
    return Q


def _gap(K, lp):
    ev = mcmc._sym_eigs(K, lp)
    return float(max(1.0 - max(abs(ev[-2]), abs(ev[0])), 0.0))


def roi_exact(J, sigma, betas_tilde=(1.0, 2.0), noise_eps=NOISE_EPS):
    t0 = time.time()
    n = len(J)
    M = 2 ** (n - 1)
    Er = ising.energies(J)[:M]
    Fobs = mcmc.coassignment_observables(n)
    row = {"n": n, "timing": {}}
    ts = time.time()
    Q = lumped_proposals(J)
    row["timing"]["proposals"] = time.time() - ts
    for bt in betas_tilde:
        ts = time.time()
        beta = bt / sigma
        lp, pi = gibbs(Er, beta)
        p, var = coassignment(pi, Fobs)
        out = {"beta": beta, "p": p.tolist(), "var": var.tolist(), "u": node_uncertainty(p, n).tolist(), "chains": {}}

        def rec(name, K):
            s2, _ = mcmc.asymptotic_variances(K, lp, Fobs)
            out["chains"][name] = {"gap": _gap(K, lp), "s2": s2.tolist()}

        Pq = mcmc.mh_matrix(Q["quantum"], Er, beta)
        Pu = mcmc.mh_matrix(Q["uniform"], Er, beta)
        rec("quantum", Pq)
        rec("uniform", Pu)
        for eps in noise_eps:
            rec(f"quantum_noisy_{eps}", (1.0 - eps) * Pq + eps * Pu)
        del Pq, Pu
        for name in ("local", "label_swap", "qaoa_a0"):
            rec(name, mcmc.mh_matrix(Q[name], Er, beta))
        SW, WO = cluster_lumped(J, beta)
        rec("sw", SW)
        rec("wolff", WO)
        del SW, WO
        out["chains"]["heat_bath"] = {"gap": None, "s2": (var * M).tolist()}
        row[f"b{bt}"] = out
        row["timing"][f"b{bt}"] = time.time() - ts
    row["timing"]["total"] = time.time() - t0
    return row
