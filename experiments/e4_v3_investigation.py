"""POST HOC investigation of the E4-B V3 failure (docs/E4_whole_cortex_protocol_2026-11-22.md, section 7: "a failed
check is investigated before anything is reported"). Not pre-registered.

Question: is the V3 failure (binding-pair kappa > 1.3) a code error, or does the asymptotic RMSE prediction
sqrt(s2 / (4N)) simply not describe a run of N recorded steps when N is small?

Method: for every E4-B run on a network with M = 2^(n-1) <= 256 partitions, compute the EXACT finite-run law of the
estimator under the sampler's own semantics (qsr2/sampler_numba._one): start at s0 = argmin(Er); apply `burn` steps;
record the next N states X_1..X_N; p_hat = (1 + mean_t f(X_t)) / 2. With g = f - E_pi f:
  E[p_hat] - p   = (1 / 2N) sum_t mu_t g,          mu_t = delta_{s0} P^(burn + t)
  E[(p_hat-p)^2] = (1 / 4N^2) sum_t mu_t (g^2 + 2 g F_t),   F_N = 0,  F_t = P (g + F_(t+1))
P is the chain's exact partition-space kernel (the matrix whose rows the sampler draws from, or the MH matrix built
from the same proposal). Compared with the realised replicate values stored in results/e4_run_rows.jsonl:
  kappa_exact = exact finite-run RMSE / predicted RMSE  (what kappa should be, up to replicate noise)
  bias z      = (realised mean - exact mean) / (exact SD / sqrt(R))
  SD ratio    = realised SD / exact SD
If the sampler is correct, the realised values match the exact finite-run law within replicate noise, whatever the
asymptotic prediction says. Writes results/e4_v3_investigation.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import exact_ps, ising, mcmc, roi_exact  # noqa: E402
from instances import RES, key  # noqa: E402
from e4_instances import PRIMARY, tiling  # noqa: E402

M_MAX = 256


def kernel(J, chain, Er, beta, n):
    if chain in ("sw", "wolff"):
        SW, WO = exact_ps.cluster_lumped(J, beta)
        return SW if chain == "sw" else WO
    if chain == "heat_bath":
        _, pi = roi_exact.gibbs(Er, beta)
        return np.tile(pi, (len(Er), 1))
    Q = roi_exact.lumped_proposals(J, (chain,))[chain]
    return mcmc.mh_matrix(Q, Er, beta)


def finite_law(P, s0, burn, N, f, m):
    """Exact E[p_hat - p] and E[(p_hat - p)^2] for every pair (columns of f)."""
    M = P.shape[0]
    g = f - m                                    # M x pairs
    mu = np.zeros(M); mu[s0] = 1.0
    for _ in range(burn + 1):                    # distribution of the first recorded state
        mu = mu @ P
    mus = np.empty((N, M))
    for t in range(N):
        mus[t] = mu
        mu = mu @ P
    bias = (mus @ g).sum(0) / (2 * N)
    Fn = np.zeros_like(g)
    second = np.zeros(g.shape[1])
    for t in range(N - 1, -1, -1):               # F_t for recorded index t (0-based), F_(N-1) = 0
        second += mus[t] @ (g * g + 2 * g * Fn)
        Fn = P @ (g + Fn)
    return bias, second / (4 * N ** 2)


def main():
    rows = [json.loads(l) for l in open(os.path.join(RES, "e4_run_rows.jsonl"))]
    inst = {}
    for ds in PRIMARY:
        for d in tiling(ds):
            inst[tuple(json.loads(json.dumps(list(key(d)))))] = d
    out = []
    for r in rows:
        n = r["n"]; M = 2 ** (n - 1)
        if M > M_MAX:
            continue
        d = inst[tuple(r["key"])]
        ex = [json.loads(l) for l in open(os.path.join(RES, "e4_exact_rows.jsonl")) if json.loads(l)["key"] == r["key"]][0]
        beta = ex["b1.0"]["beta"]
        Er = ising.energies(d["J"])[:M]
        P = kernel(d["J"], r["chain"], Er, beta, n)
        Fobs = mcmc.coassignment_observables(n)
        p = np.array(r["p"]); m = 2 * p - 1
        steps = r["steps"] if r["chain"] != "heat_bath" else r["steps"]
        burn = r["burn"]
        bias, mse = finite_law(P, int(np.argmin(Er)), burn, steps, Fobs, m)
        pred = np.array(r["pred"]); rmse = np.array(r["rmse"]); R = r["R"]
        sd_exact = np.sqrt(np.maximum(mse - bias ** 2, 0))
        k_exact = np.sqrt(mse) / pred
        k_real = rmse / pred
        b = int(np.argmax(pred))                 # binding pair: largest predicted variance
        with np.errstate(divide="ignore", invalid="ignore"):
            zb = (np.array(r["mean"]) - (p + bias)) / (sd_exact / np.sqrt(R))
            sdr = np.array(r["sd"]) / sd_exact
        ok = sd_exact > 1e-9
        out.append({"ds": r["ds"], "roi": r["roi"], "n": n, "chain": r["chain"], "burn": burn, "steps": steps, "R": R,
                    "binding_pair": b, "binding_kappa_realised": float(k_real[b]), "binding_kappa_exact_finite": float(k_exact[b]),
                    "binding_min_p": float(min(p[b], 1 - p[b])),
                    "max_abs_bias_z": float(np.nanmax(np.abs(zb[ok]))) if ok.any() else None,
                    "sd_ratio_range": [float(np.nanmin(sdr[ok])), float(np.nanmax(sdr[ok]))] if ok.any() else None,
                    "realised_over_exact_rmse_range": [float(np.min(rmse[ok] / np.sqrt(mse[ok]))), float(np.max(rmse[ok] / np.sqrt(mse[ok])))] if ok.any() else None,
                    "pairs_with_nonzero_spread": int(ok.sum()), "pairs": int(len(p))})
    S = {"note": __doc__, "runs_checked": len(out), "runs": out}
    json.dump(S, open(os.path.join(RES, "e4_v3_investigation.json"), "w"), indent=1)
    for o in out:
        print(f"{o['ds']} {o['roi']:14s} n={o['n']:2d} {o['chain']:9s} burn={o['burn']:5d} N={o['steps']:6d} | binding kappa "
              f"realised {o['binding_kappa_realised']:.3f} exact-finite {o['binding_kappa_exact_finite']:.3f} | max|bias z| "
              f"{o['max_abs_bias_z'] if o['max_abs_bias_z'] is None else round(o['max_abs_bias_z'], 2)} | realised/exact RMSE "
              f"{o['realised_over_exact_rmse_range'] and [round(x, 2) for x in o['realised_over_exact_rmse_range']]}")


if __name__ == "__main__":
    main()
