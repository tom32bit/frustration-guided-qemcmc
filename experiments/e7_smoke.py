"""E7 feasibility smoke test, run BEFORE the E7 protocol was fixed, on instances that are NOT in the E7 study set:
the first 13 parcels of the Schaefer-400 LH_Cont ROI (FC and SC; the full ROI has 22 parcels and is beyond exact
computation) and one random Gaussian model at n = 13. Purpose: run times, the annealing step size dt, and the QAOA
optimiser settings. It prints timings and numerical diagnostics only (Trotter differences in p_ij between step sizes,
spread of optimiser restarts), no bias floors, so that no study-like outcome informs the protocol.
Writes results/e7_smoke.log (stdout)."""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import direct, ising, mcmc, networks, patches, proposals, roi_exact  # noqa: E402
from qsr2.exact_ps import qaoa_lumped  # noqa: E402
from instances import sigma_of  # noqa: E402


def instances():
    out = []
    for ds in ("FC400", "SC400"):
        W, labels = networks.load_connectome(ds)
        ix = patches.canonical_rois(labels, 2, 60)["LH_Cont"][:13]
        J, _ = patches.standalone_patch(W, ix)
        out.append((f"{ds}_LH_Cont_first13", J))
    rng = np.random.default_rng(99)
    J = np.triu(rng.standard_normal((13, 13)), 1)
    out.append(("random_n13", J + J.T))
    return out


def main():
    for name, J in instances():
        n = len(J)
        M = 2 ** (n - 1)
        E = ising.energies(J)
        Er = E[:M]
        a = proposals.layden_alpha(J, np.zeros(n))
        aE = a * E
        F = mcmc.coassignment_observables(n)
        sig = sigma_of(J)
        print(f"== {name}: n = {n}, sigma = {sig:.4g}", flush=True)
        for p in (1, 2, 3, 4):
            th = np.full(2 * p, 0.3)
            t = time.time()
            for _ in range(20):
                direct.qaoa_probs(aE, th[:p], th[p:])
            print(f"  QAOA p={p}: {1e3 * (time.time() - t) / 20:.2f} ms per evaluation", flush=True)
        for T in (1.0, 3.0, 10.0, 30.0, 100.0):
            t = time.time()
            res = {dt: direct.coassign(direct.anneal_probs(aE, T, dt), F) for dt in (0.02, 0.01, 0.005)}
            print(f"  anneal T={T:5.0f}: max|p(0.02)-p(0.01)| = {np.max(np.abs(res[0.02] - res[0.01])):.2e}, "
                  f"max|p(0.01)-p(0.005)| = {np.max(np.abs(res[0.01] - res[0.005])):.2e} ({time.time() - t:.1f} s for 3 dt)",
                  flush=True)
        t = time.time()
        direct.greedy_probs(J)
        print(f"  greedy: {time.time() - t:.2f} s", flush=True)
        logpi, pi = direct.gibbs(Er, 1.0 / sig)
        p_t = direct.coassign(pi, F)
        for kind in ("energy", "kl", "oracle_smooth"):
            f = direct.make_objective(kind, aE, n, logpi=logpi, p_target=p_t, Fobs=F)
            pol = direct.make_objective("oracle_max", aE, n, p_target=p_t, Fobs=F) if kind == "oracle_smooth" else None
            t = time.time()
            res = direct.optimise_qaoa(f, 4, np.random.default_rng(1), np.random.default_rng(2), polish=pol)
            diag = "; ".join(f"p={q}: second-set gain {res[q]['smooth_first'] - res[q]['smooth_best']:.1e}, "
                             f"{res[q]['n_within_1e-6']}/{res[q]['n_starts']} starts at best, nfev {res[q]['nfev']}"
                             for q in res)
            print(f"  optimise {kind:13s}: {time.time() - t:6.1f} s | {diag}", flush=True)
        t = time.time()
        Q = qaoa_lumped(E, a, *proposals.QAOA_GRID[0])
        P = mcmc.mh_matrix(Q, Er, 1.0 / sig)
        st = float(np.max(np.abs(pi @ P - pi)))
        del Q, P
        print(f"  QeMCMC a0 kernel build + stationarity check: {time.time() - t:.1f} s (max|pi P - pi| = {st:.1e})",
              flush=True)


if __name__ == "__main__":
    main()
