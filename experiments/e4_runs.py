"""E4-B end-to-end runs (docs/E4_whole_cortex_protocol_2026-11-22.md, section 6) on the primary tilings.

For every ROI, each arm runs its own chain with its own exact asymptotic budget for RMSE eps = 0.02 on every
co-assignment of the ROI:
  arm H (QSR 2.0): the untuned quench on routed ROIs (sealed routing), the classical oracle C* elsewhere;
  arm C: C* on every ROI.
Where both arms use the same chain (non-routed ROIs) one run serves both. Each of R replicates starts at the exact
ground-state partition, discards burn = ceil(10 / gap) steps (counted as proposals), then runs
N = ceil(max_ij s2_ij / (4 eps^2)) steps. Heat-bath: K = ceil(max_ij var_ij / (4 eps^2)) i.i.d. draws, counted as
K * 2^(n-1) proposals, no burn-in. Seeds: SeedSequence([20261127, tiling, ROI, chain]).
Compute cap: if sum over runs of (burn + N) * R exceeds 3e10 sampler steps, R is lowered to
max(50, floor(R * 3e10 / total)); if R = 50 still exceeds the cap, eps is raised to the first of 0.03, 0.04, 0.05
that fits. The values used are recorded in every row.
Writes results/e4_run_rows.jsonl (resumable by (key, chain)). Usage: e4_runs.py
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import exact_ps, ising, mcmc, roi_exact, sampler_numba as smp  # noqa: E402
from instances import RES, key  # noqa: E402
from e4_instances import PRIMARY, tiling  # noqa: E402
from e4_analyze import oracle  # noqa: E402

EPS0, R0, R_MIN, CAP, SEED = 0.02, 200, 50, 3e10, 20261127
EPS_STEPS = (0.03, 0.04, 0.05)


def budget(b, chain, n, eps):
    if chain == "heat_bath":
        K = math.ceil(max(b["var"]) / (4 * eps ** 2))
        return 0, K, K * 2 ** (n - 1)
    N = math.ceil(max(b["chains"][chain]["s2"]) / (4 * eps ** 2))
    burn = math.ceil(10 / b["chains"][chain]["gap"])
    return burn, N, burn + N


def plan(rows, R1, eps):
    tasks = []
    for r in rows:
        k = tuple(r["key"]); b = r["b1.0"]; cs = oracle(b)
        chains = [cs] + (["quantum"] if R1[k] else [])
        for c in chains:
            burn, steps, cst = budget(b, c, r["n"], eps)
            arms = (["C"] + ([] if R1[k] else ["H"])) if c == cs else ["H"]
            tasks.append({"row": r, "chain": c, "arms": arms, "burn": burn, "steps": steps, "cost": cst})
    return tasks


def kernel_cdf(J, chain, beta, pi):
    if chain in ("quantum", "qaoa_a0"):
        return smp.row_cdf(roi_exact.lumped_proposals(J, (chain,))[chain])
    if chain in ("sw", "wolff"):
        SW, WO = exact_ps.cluster_lumped(J, beta)
        return smp.row_cdf(SW if chain == "sw" else WO)
    if chain == "heat_bath":
        return smp.row_cdf(pi)
    return np.ones((1, 1))


def main():
    rows = [json.loads(l) for l in open(os.path.join(RES, "e4_exact_rows.jsonl"))]
    rows = [r for r in rows if r["ds"] in PRIMARY]
    P = json.load(open(os.path.join(RES, "e4_routing_predictions.json")))
    R1 = {tuple(x["key"]): x["routed_b1.0"] for x in P["instances"]}
    eps, R = EPS0, R0
    total = sum((t["burn"] + t["steps"]) for t in plan(rows, R1, eps)) * R
    if total > CAP:
        R = max(R_MIN, int(R * CAP / total))
        for e in EPS_STEPS:
            if sum((t["burn"] + t["steps"]) for t in plan(rows, R1, eps)) * R <= CAP:
                break
            eps = e
    tasks = plan(rows, R1, eps)
    total = sum((t["burn"] + t["steps"]) for t in tasks) * R
    print(f"E4-B: {len(tasks)} runs, R = {R}, eps = {eps}, {total:.3g} sampler steps", flush=True)
    inst, order = {}, {}
    for ds in PRIMARY:
        for j, d in enumerate(tiling(ds)):
            kk = tuple(json.loads(json.dumps(list(key(d)))))
            inst[kk], order[kk] = d, j
    path = os.path.join(RES, "e4_run_rows.jsonl")
    done = {(tuple(x["key"]), x["chain"]) for x in map(json.loads, open(path))} if os.path.exists(path) else set()
    ds_ix = {ds: i for i, ds in enumerate(PRIMARY)}
    t0 = time.time()
    with open(path, "a") as f:
        for i, t in enumerate(tasks):
            r = t["row"]; k = tuple(r["key"]); c = t["chain"]
            if (k, c) in done:
                continue
            ts = time.time()
            d = inst[k]; n = r["n"]; b = r["b1.0"]; beta = b["beta"]
            Er = ising.energies(d["J"])[: 2 ** (n - 1)]
            lp, pi = roi_exact.gibbs(Er, beta)
            cdf = kernel_cdf(d["J"], c, beta, pi)
            ss = np.random.SeedSequence([SEED, ds_ix[r["ds"]], order[k], list(smp.KINDS).index(c)])
            seeds = ss.generate_state(R, dtype=np.uint32).astype(np.int64)
            H = smp.run_chains(smp.KINDS[c], cdf, Er, beta, n, int(np.argmin(Er)), t["burn"], t["steps"], seeds)
            Fobs = mcmc.coassignment_observables(n)
            ph = (1 + H @ Fobs / t["steps"]) / 2
            p = np.array(b["p"])
            s2 = np.array(b["var"]) if c == "heat_bath" else np.array(b["chains"][c]["s2"])
            pred = np.sqrt(s2 / (4 * t["steps"]))
            rmse = np.sqrt(np.mean((ph - p) ** 2, 0))
            U = np.array([roi_exact.node_uncertainty(x, n) for x in ph])
            out = {"key": list(k), "ds": r["ds"], "roi": r["roi"], "n": n, "chain": c, "arms": t["arms"], "R": R, "eps": eps,
                   "burn": t["burn"], "steps": t["steps"], "cost": t["cost"], "p": p.tolist(),
                   "mean": ph.mean(0).tolist(), "sd": ph.std(0, ddof=1).tolist(), "rmse": rmse.tolist(), "pred": pred.tolist(),
                   "z": (np.sqrt(R) * (ph.mean(0) - p) / pred).tolist(), "kappa": (rmse / pred).tolist(),
                   "u_hat_mean": U.mean(0).tolist(), "u_hat_sd": U.std(0, ddof=1).tolist(), "seconds": time.time() - ts}
            f.write(json.dumps(out) + "\n"); f.flush()
            print(f"{i + 1}/{len(tasks)} {r['ds']} {r['roi']} n={n} {c}: {(t['burn'] + t['steps']) * R:.3g} steps, "
                  f"{out['seconds']:.0f}s, median kappa {np.median(out['kappa']):.2f} (elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
