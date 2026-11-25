"""Exact per-instance quantities for G1c and E3 Tier A (field-free Ising patches, n <= 12).

For every chain and temperature beta = beta_tilde / sigma:
  <chain>_lumped     partition-space absolute spectral gap (PRIMARY from G1c / E3 amendment 7)
  <chain>_full       labelled-state gap (continuity with G1/G1b; optional, costs a 2^n eigensolve)
  <chain>_s2mean / _s2max   exact asymptotic variance of the co-assignment estimators z_i z_j (mean / max over
                     pairs): steps needed for RMSE eps ~ s2 / eps^2  (secondary)
Chains: local, label_swap, uniform, quantum (Layden quench, grid average, untuned), qaoa (palindromic QAOA-MC grid
average), sw, wolff; PT-2 / PT-3 partition-space gaps per ladder; QAOA-MC per grid angle (partition-space gap).
Rows are plain dicts of floats (None where an eigensolver failed, with the error recorded).
"""
import time

import numpy as np

from . import cluster, ising, mcmc, proposals

R2 = [0.5, 1 / 3, 0.2, 0.1, 1 / 30, 0.0]
R3 = [0.3, 0.1, 0.0]


def _single(row, name, P, E, beta, bt, n, Fobs, full, asym):
    """Partition-space gap (even block A + B), labelled-state gap (spectrum of even U odd block, A - B), asymptotic
    variances on the even block; two 2^(n-1) eigenproblems instead of one 2^n (mcmc.sector_gaps, tested)."""
    N, M = 2 ** n, 2 ** (n - 1)
    r = np.arange(M)
    A, B = P[np.ix_(r, r)], P[np.ix_(r, r ^ (N - 1))]
    lp = -beta * (E[:M] - E.min())
    L = A + B
    ev = mcmc._sym_eigs(L, lp)
    row[f"b{bt}_{name}_lumped"] = float(max(1.0 - max(abs(ev[-2]), abs(ev[0])), 0.0))
    if full:
        od = mcmc._sym_eigs(A - B, lp)
        row[f"b{bt}_{name}_full"] = float(max(1.0 - max(abs(ev[-2]), abs(ev[0]), np.max(np.abs(od))), 0.0))
    if asym:
        s2, var = mcmc.asymptotic_variances(L, lp, Fobs)
        row[f"b{bt}_{name}_s2mean"] = float(np.mean(s2))
        row[f"b{bt}_{name}_s2max"] = float(np.max(s2))
        row[f"b{bt}_varmean"] = float(np.mean(var))


def compute_exact(J, sigma, betas_tilde=(0.5, 1.0, 2.0), full=True, asym=True, pt=True, qaoa_angles=True,
                  chains=("local", "label_swap", "uniform", "quantum", "qaoa", "sw", "wolff")):
    t0 = time.time()
    n = len(J)
    E = ising.energies(J)
    Fobs = mcmc.coassignment_observables(n) if asym else None
    row = {"n": n, "timing": {}}
    betas = {bt: bt / sigma for bt in betas_tilde}
    a = proposals.layden_alpha(J, np.zeros(n))

    def mh_chain(name, Q):
        for bt, beta in betas.items():
            _single(row, name, mcmc.mh_matrix(Q, E, beta), E, beta, bt, n, Fobs, full, asym)

    for name in ("local", "label_swap", "uniform"):
        if name in chains:
            ts = time.time()
            mh_chain(name, getattr(proposals, name)(n))
            row["timing"][name] = time.time() - ts
    if "quantum" in chains:
        ts = time.time()
        mh_chain("quantum", proposals.quench_mean_sectors(E, J)[1])
        row["timing"]["quantum"] = time.time() - ts
    if "qaoa" in chains or qaoa_angles:
        ts = time.time()
        Qsum = np.zeros((2 ** n, 2 ** n))
        b1 = betas.get(1.0)
        for i, (x1, x2, g1) in enumerate(proposals.QAOA_GRID):
            Qi = proposals.qaoa_mc_proposal(E, a, x1, x2, g1)          # = |U|^2 (symmetric), half the work
            Qsum += Qi
            if qaoa_angles and b1 is not None:
                M = 2 ** (n - 1)
                rr = np.arange(M)
                Qb = Qi[np.ix_(rr, rr)] + Qi[np.ix_(rr, rr ^ (2 ** n - 1))]       # partition-space proposal
                row[f"b1.0_qaoa_a{i}_lumped"] = mcmc.spectral_gap(mcmc.mh_matrix(Qb, E[:M], b1),
                                                                  logpi=-b1 * (E[:M] - E.min()))
            del Qi
        if "qaoa" in chains:
            mh_chain("qaoa", Qsum / len(proposals.QAOA_GRID))
        del Qsum
        row["timing"]["qaoa"] = time.time() - ts
    if "sw" in chains or "wolff" in chains:
        ts = time.time()
        for bt, beta in betas.items():
            SW, WO = cluster.sw_wolff_transitions(J, beta)
            if "sw" in chains:
                _single(row, "sw", SW, E, beta, bt, n, Fobs, full, asym)
            if "wolff" in chains:
                _single(row, "wolff", WO, E, beta, bt, n, Fobs, full, asym)
            del SW, WO
        row["timing"]["cluster"] = time.time() - ts
    if pt:
        ts = time.time()
        for bt, beta in betas.items():
            if n <= 8 and bt in (1.0, 2.0):
                for r in R2:
                    _pt(row, f"b{bt}_pt2_r{r:.4f}_lumped", E, n, [beta, r * beta])
            if n <= 6 and bt == 1.0:
                for r in R3:
                    _pt(row, f"b{bt}_pt3_r{r:.4f}_lumped", E, n, [beta, (1 + r) / 2 * beta, r * beta])
        row["timing"]["pt"] = time.time() - ts
    row["timing"]["total"] = time.time() - t0
    return row


def _pt(row, key, E, n, betas):
    try:
        row[key] = mcmc.pt_gap(E, n, betas, lumped=True)
    except Exception as e:  # ARPACK non-convergence
        row[key] = None
        row.setdefault("errors", []).append(f"{key}: {str(e)[:60]}")
