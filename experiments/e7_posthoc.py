"""POST HOC analyses of E7 (not pre-specified; chosen after the E7 results were seen). Reads results/e7_rows.jsonl.

1. Why direct sampling works where it does: for every primary network at beta_tilde = 1, the exact Gibbs probability
   of the most probable bipartition (pi_max), the effective number of bipartitions exp(H(pi)), and the mean node
   uncertainty, against whether the best deployable floor is within EPS.
2. Where the uncertainty is: the share of the summed parcel uncertainty (sum of u_i over the 98 parcels) that lies in
   networks where no deployable direct sampler is within EPS.
3. Noise tolerance: the number of primary networks on which the best deployable sampler stays within EPS after
   depolarising noise P -> (1 - eta) P + eta uniform, for eta from 0 to 0.2 (p_ij -> (1 - eta) p_ij + eta / 2).
4. Calls where direct sampling works: the direct sampler that reaches EPS with the fewest shots (amendment 1:
   at least one shot) against QeMCMC calls (QAOA-MC angle 0 and quench).
Writes results/e7_posthoc.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import direct, ising  # noqa: E402
from instances import RES  # noqa: E402
from e7_instances import primary  # noqa: E402
from e7_run import DEPLOYABLE, skey  # noqa: E402

EPS = 0.02
ETAS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]


def main():
    rows = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(os.path.join(RES, "e7_rows.jsonl"))}
    nets = []
    for d in primary():
        r = rows[skey(d)]
        b = r["b1.0"]
        n = len(d["J"])
        Er = ising.energies(d["J"])[: 2 ** (n - 1)]
        logpi, pi = direct.gibbs(Er, b["beta"])
        H = -float(np.sum(pi[pi > 0] * logpi[pi > 0]))
        fl = {nm: b["samplers"][nm]["floor"] for nm in DEPLOYABLE}
        best = min(fl, key=fl.get)
        p = np.array(b["p"])
        noise = {}
        for eta in ETAS:
            noise[str(eta)] = min(float(np.max(np.abs((1 - eta) * np.array(b["samplers"][nm]["pD"]) + eta / 2 - p)))
                                  for nm in DEPLOYABLE)
        calls = {nm: max(1.0, b["samplers"][nm]["calls_to_eps"]) for nm in DEPLOYABLE}
        cbest = min(calls, key=calls.get)
        nets.append({"network": f"{r['ds']} {r['roi']}", "ds": r["ds"], "n": n, "F": r["F"],
                     "pi_max": float(pi.max()), "effective_partitions": float(np.exp(H)),
                     "mean_u": float(np.mean(b["u"])), "sum_u": float(np.sum(b["u"])),
                     "max_pair_uncertainty": float(np.max(np.minimum(p, 1 - p))),
                     "best_deployable": best, "best_floor": fl[best], "direct_works": fl[best] <= EPS,
                     "fixed_samplers_within_eps": [nm for nm in DEPLOYABLE if fl[nm] <= EPS],
                     "noise_best_floor": noise, "direct_min_calls": calls[cbest], "direct_min_calls_sampler": cbest,
                     "qemcmc_a0_calls": b["qemcmc"]["qaoa_a0"]["calls"], "qemcmc_quench_calls": b["qemcmc"]["quantum"]["calls"]})
    works = [x for x in nets if x["direct_works"]]
    fails = [x for x in nets if not x["direct_works"]]
    out = {"note": __doc__, "networks": nets}
    for ds in ("FC100", "SC100"):
        N = [x for x in nets if x["ds"] == ds]
        tot = sum(x["sum_u"] for x in N)
        out[f"uncertainty_share_in_failing_networks_{ds}"] = sum(x["sum_u"] for x in N if not x["direct_works"]) / tot
        out[f"parcels_in_failing_networks_{ds}"] = sum(x["n"] for x in N if not x["direct_works"])
    out["noise_networks_within_eps"] = {str(eta): sum(x["noise_best_floor"][str(eta)] <= EPS for x in nets) for eta in ETAS}
    out["separation"] = {
        "works": {"n": len(works), "pi_max_range": [min(x["pi_max"] for x in works), max(x["pi_max"] for x in works)],
                  "mean_u_range": [min(x["mean_u"] for x in works), max(x["mean_u"] for x in works)],
                  "max_pair_uncertainty_range": [min(x["max_pair_uncertainty"] for x in works), max(x["max_pair_uncertainty"] for x in works)]},
        "fails": {"n": len(fails), "pi_max_range": [min(x["pi_max"] for x in fails), max(x["pi_max"] for x in fails)],
                  "mean_u_range": [min(x["mean_u"] for x in fails), max(x["mean_u"] for x in fails)],
                  "max_pair_uncertainty_range": [min(x["max_pair_uncertainty"] for x in fails), max(x["max_pair_uncertainty"] for x in fails)],
                  "floor_range": [min(x["best_floor"] for x in fails), max(x["best_floor"] for x in fails)]}}
    out["fixed_sampler_networks_within_eps"] = {nm: sum(nm in x["fixed_samplers_within_eps"] for x in nets) for nm in DEPLOYABLE}
    json.dump(out, open(os.path.join(RES, "e7_posthoc.json"), "w"), indent=1)
    print("network            n   pi_max  eff.parts  mean_u  max min(p,1-p)  best floor [sampler]          direct calls [sampler]   QeMCMC a0 / quench")
    for x in sorted(nets, key=lambda x: x["mean_u"]):
        print(f"{x['network']:20s} {x['n']:2d}  {x['pi_max']:.3f}  {x['effective_partitions']:8.1f}  {x['mean_u']:.3f}   {x['max_pair_uncertainty']:.4f}"
              f"       {x['best_floor']:.4f} [{x['best_deployable']:14s}] {'W' if x['direct_works'] else 'F'}  "
              f"{x['direct_min_calls']:9.3g} [{x['direct_min_calls_sampler']:14s}]  {x['qemcmc_a0_calls']:8.3g} / {x['qemcmc_quench_calls']:8.3g}")
    for k in ("uncertainty_share_in_failing_networks_FC100", "uncertainty_share_in_failing_networks_SC100",
              "parcels_in_failing_networks_FC100", "parcels_in_failing_networks_SC100"):
        print(k, round(out[k], 3) if isinstance(out[k], float) else out[k])
    print("noise: networks within eps by eta:", out["noise_networks_within_eps"])
    print("separation:", json.dumps(out["separation"], indent=1))
    print("fixed samplers, networks within eps:", out["fixed_sampler_networks_within_eps"])


if __name__ == "__main__":
    main()
