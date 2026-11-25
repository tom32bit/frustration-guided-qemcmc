"""E5 pilot metrics (docs/E5_hardware_pilot_protocol_2026-11-21.md) for results/e5_pilot_<MOD>_<dry|hw>.json.
Usage: e5_pilot_analyze.py SC|FC dry|hw   -> prints metrics and writes results/e5_pilot_<MOD>_<tag>_metrics.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from qsr2 import cluster, ising, mcmc, proposals  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")


def probs(counts, n):
    p = np.zeros(2 ** n)
    for b, c in counts.items():
        p[int(b, 2)] += c
    return p / p.sum()


def gap_general(L):
    """1 - (second largest |eigenvalue|): equals the absolute gap for reversible chains, defined for any chain."""
    lam = np.sort(np.abs(np.linalg.eigvals(L)))
    return float(1 - lam[-2])


def stationary(P):
    w, V = np.linalg.eig(P.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1))])
    return v / v.sum()


def main():
    mod, tag = sys.argv[1], sys.argv[2]
    d = json.load(open(os.path.join(RES, f"e5_pilot_{mod}_{tag}.json")))
    from qsr2 import networks, patches
    W, labels = networks.load_connectome(f"{mod}100")
    J, _ = patches.standalone_patch(W, np.array(d["nodes"]))
    n, N = len(J), 2 ** len(J)
    E = ising.energies(J)
    beta = d["beta_tilde"] / d["sigma"]
    b1, b2, g1 = d["angles"]
    Qid = (np.abs(proposals.qaoa_mc_unitary(E, d["alpha"], b1, b2, g1)) ** 2).T      # Q[x, x'] = P(x -> x')
    Qhw = np.array([probs(c, n) for c in d["counts"]])
    rng = np.random.default_rng(3)
    tv = lambda p, q: 0.5 * np.abs(p - q).sum(-1)
    floor = np.mean([tv(np.array([rng.multinomial(d["shots"], q) / d["shots"] for q in Qid]), Qid).mean()
                     for _ in range(20)])
    F = np.arange(N) ^ (N - 1)
    Qflip = Qhw[np.ix_(F, F)]
    resamp = np.array([rng.multinomial(d["shots"], q) / d["shots"] for q in Qid])
    m = {"modality": mod, "tag": tag, "backend": d["transpile"]["backend"], "transpile": d["transpile"],
         "angle_index": d["angle_index"], "usage_quantum_seconds": d.get("usage_quantum_seconds"),
         "row_TV_mean": float(tv(Qhw, Qid).mean()), "row_TV_finite_shot_floor": float(floor),
         "flip_asym_TV_mean": float(tv(Qhw, Qflip).mean()),
         "flip_asym_floor": float(tv(resamp, resamp[np.ix_(F, F)]).mean()),
         "transpose_asym_rel": float(np.linalg.norm(Qhw - Qhw.T) / np.linalg.norm(Qhw)),
         "transpose_asym_floor": float(np.linalg.norm(resamp - resamp.T) / np.linalg.norm(resamp))}
    Qf = (Qhw + Qflip) / 2                                                         # flip average (= flip twirl)
    Php = mcmc.mh_matrix(Qf, E, beta)
    pi_hw, pi = stationary(Php), ising.gibbs(E, beta)
    Z = ising.spins(n).astype(float)
    iu = np.triu_indices(n, 1)
    co = lambda p: ((1 + (Z.T * p) @ Z) / 2)[iu]
    m["stationary_TV_to_gibbs"] = float(tv(pi_hw, pi))
    m["coassignment_max_abs_error"] = float(np.max(np.abs(co(pi_hw) - co(pi))))
    m["coassignment_mean_abs_error"] = float(np.mean(np.abs(co(pi_hw) - co(pi))))
    lp = -beta * (E - E.min())
    gaps = {"hardware_qaoa": gap_general(mcmc.lump(Php, n)),
            "ideal_qaoa": gap_general(mcmc.lump(mcmc.mh_matrix(Qid, E, beta), n)),
            "ideal_quench": gap_general(mcmc.lump(mcmc.mh_matrix(proposals.quench_mean(E, J), E, beta), n))}
    SW, WO = cluster.sw_wolff_transitions(J, beta)
    cl = {"local": mcmc.mh_matrix(proposals.local(n), E, beta), "label_swap": mcmc.mh_matrix(proposals.label_swap(n), E, beta),
          "uniform": mcmc.mh_matrix(proposals.uniform(n), E, beta), "sw": SW, "wolff": WO}
    for k, P in cl.items():
        gaps[k] = gap_general(mcmc.lump(P, n))
    gaps["heat_bath_per_cost"] = 2.0 ** -(n - 1)
    gaps["classical_oracle"] = max(gaps[k] for k in list(cl) + ["heat_bath_per_cost"])
    m["partition_space_gaps"] = {k: round(v, 5) for k, v in gaps.items()}
    m["hardware_over_ideal_qaoa"] = gaps["hardware_qaoa"] / gaps["ideal_qaoa"]
    m["hardware_over_classical_oracle"] = gaps["hardware_qaoa"] / gaps["classical_oracle"]
    json.dump(m, open(os.path.join(RES, f"e5_pilot_{mod}_{tag}_metrics.json"), "w"), indent=1, default=str)
    print(json.dumps(m, indent=1, default=str))


if __name__ == "__main__":
    main()
