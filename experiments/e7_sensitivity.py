"""E7 pre-specified sensitivity re-search (docs/E7_direct_sampling_protocol_2026-11-23.md, section 7).

For every PRIMARY network whose best deployable floor at beta_tilde = 1 lies in (EPS, 1.5 EPS] (just above the
target, where a better angle search could change the E7-A count), the energy and KL QAOA searches are repeated at
every depth with 64 fresh random starts (seed streams 3, distinct from the main run) plus the main run's optima as
starts, and the floors are recomputed. The new best deployable floor is the minimum of the main-run value and the
re-searched QAOA floors. Writes results/e7_sensitivity.json (an empty list if no network qualifies). E7-A is
re-evaluated with these floors as a sensitivity analysis; the primary endpoint uses the main run.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import direct, ising, mcmc, proposals  # noqa: E402
from instances import RES  # noqa: E402
from e7_instances import all_networks  # noqa: E402
from e7_run import DEPLOYABLE, P_MAX, rng, skey  # noqa: E402

EPS = 0.02
BAND = (EPS, 1.5 * EPS)
N_EXTRA = 64


def main():
    rows = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(os.path.join(RES, "e7_rows.jsonl"))}
    nets = {skey(d): d for d in all_networks()}
    out = []
    for k, r in sorted(rows.items(), key=lambda kv: kv[1]["index"]):
        if r["set"] != "primary":
            continue
        b = r["b1.0"]
        fl = {nm: b["samplers"][nm]["floor"] for nm in DEPLOYABLE}
        best_nm = min(fl, key=fl.get)
        if not (BAND[0] < fl[best_nm] <= BAND[1]):
            continue
        d = nets[k]
        J = d["J"]
        n = len(J)
        E = ising.energies(J)
        Er = E[: 2 ** (n - 1)]
        aE = np.ascontiguousarray(proposals.layden_alpha(J, np.zeros(n)) * E)
        F = mcmc.coassignment_observables(n)
        logpi, pi = direct.gibbs(Er, b["beta"])
        p_t = direct.coassign(pi, F)
        assert np.max(np.abs(p_t - np.array(b["p"]))) < 1e-12
        idx = r["index"]
        new = {}
        for kind, obj in (("energy", direct.make_objective("energy", aE, n)),
                          ("kl", direct.make_objective("kl", aE, n, logpi=logpi))):
            main_opt = r["opt_energy"] if kind == "energy" else b["opt_kl"]
            extra = {p: [main_opt[str(p)]["theta"]] for p in range(1, P_MAX + 1)}
            res = direct.optimise_qaoa(obj, P_MAX, rng(idx, kind, 1.0, 3), rng(idx, kind, 1.0, 4), n_random=N_EXTRA,
                                       n_second=0, extra=extra)
            for p in range(1, P_MAX + 1):
                th = np.array(res[p]["theta"])
                P = direct.qaoa_probs(aE, th[:p], th[p:])
                new[f"qaoa_{kind}_p{p}"] = {"floor": float(np.max(np.abs(direct.coassign(P, F) - p_t))),
                                            "objective": res[p]["value"], "main_objective": main_opt[str(p)]["value"],
                                            "main_floor": fl[f"qaoa_{kind}_p{p}"]}
        new_best = min([fl[best_nm]] + [v["floor"] for v in new.values()])
        out.append({"key": list(k), "network": f"{r['ds']} {r['roi']}", "n": n, "main_best_floor": fl[best_nm],
                    "main_best": best_nm, "researched": new, "new_best_floor": new_best})
        print(f"{r['ds']} {r['roi']} n={n}: main best {fl[best_nm]:.4f} [{best_nm}] -> after re-search {new_best:.4f}",
              flush=True)
    json.dump({"band": BAND, "n_extra_random_starts": N_EXTRA, "networks": out},
              open(os.path.join(RES, "e7_sensitivity.json"), "w"), indent=1)
    print(f"sensitivity re-search: {len(out)} primary networks in the band {BAND}", flush=True)


if __name__ == "__main__":
    main()
