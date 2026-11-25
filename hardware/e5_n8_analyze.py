"""E5 proper, n = 8: pre-registered metrics (docs/E5_hardware_n8_protocol_2026-11-22.md) for
results/e5_n8_<RANK>_<dry|hw>.json.   Usage: e5_n8_analyze.py dry|hw   -> results/e5_n8_<tag>_metrics.json

Partition space (representatives s < 2^(n-1)); beta = 1/sigma_ROI. Per instance:
  measured proposal  Qhat[s, c] = share of shots from representative s whose outcome lies in class c = {x, NOT x}
  hardware chain     Metropolis with Qhat treated as symmetric (how the device would be used); exact analysis:
                     gap = 1 - |lambda_2| (defined for non-reversible chains), stationary law, co-assignment bias
  primary criteria   H1 retention  gap_hw / gap_ideal >= 0.5
                     H2 advantage  gap_hw / gap_oracle > 1   (oracle: exact best classical chain, e5_selection.json)
                     H3 bias       max_ij |p_hw_ij - p_ij| <= 0.05
                     overall: H1-H3 all hold on at least 2 of the 3 instances
  descriptive        row TV and transpose asymmetry vs finite-shot floors; Hastings-corrected chain (exactly
                     Gibbs-reversible for Qhat); finite-shot null (ideal rows resampled at the same shots); parametric
                     bootstrap of the hardware rows (95 % intervals)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from qsr2 import ising, mcmc, networks, patches, proposals  # noqa: E402

RES = os.path.join(ROOT, "results")
K_NULL, K_BOOT = 100, 200


def lumped_counts(counts, n):
    N, M = 2 ** n, 2 ** (n - 1)
    p = np.zeros(M)
    for b, c in counts.items():
        x = int(b.replace(" ", ""), 2)
        p[x if x < M else x ^ (N - 1)] += c
    return p


def gap_general(P):
    lam = np.sort(np.abs(np.linalg.eigvals(P)))
    return float(1 - lam[-2])


def stationary(P):
    w, V = np.linalg.eig(P.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1))])
    return v / v.sum()


def hastings(Q, Er, beta):
    """MH with the Hastings ratio of Q itself: reversible w.r.t. Gibbs for any Q (moves with Q[c, s] = 0 rejected)."""
    lp = -beta * (Er - Er.min())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Q > 0, np.exp(lp[None, :] - lp[:, None]) * Q.T / np.where(Q > 0, Q, 1.0), 0.0)
    P = Q * np.minimum(1.0, ratio)
    np.fill_diagonal(P, 0.0)
    P[np.diag_indices_from(P)] = 1.0 - P.sum(1)
    return P


def chain_metrics(Q, Er, beta, Fobs, pi, co_exact):
    P = mcmc.mh_matrix(Q, Er, beta)
    pis = stationary(P)
    co = (1 + pis @ Fobs) / 2
    return {"gap": gap_general(P), "stationary_TV": float(0.5 * np.abs(pis - pi).sum()),
            "coassign_max_err": float(np.max(np.abs(co - co_exact))), "coassign_mean_err": float(np.mean(np.abs(co - co_exact)))}


def analyse(d, cand, rng):
    n = d["n"]
    N, M = 2 ** n, 2 ** (n - 1)
    W, _ = networks.load_connectome(d["ds"])
    J, _ = patches.standalone_patch(W, np.array(d["nodes"]))
    E = ising.energies(J)
    Er = E[:M]
    beta = d["beta_tilde"] / d["sigma"]
    b1, b2, g1 = d["angles"]
    Qf = proposals.qaoa_mc_proposal(E, d["alpha"], b1, b2, g1)
    r = np.arange(M)
    Qid = Qf[np.ix_(r, r)] + Qf[np.ix_(r, r ^ (N - 1))]
    shots = d["shots"]
    C = np.array([lumped_counts(c, n) for c in d["counts"]])
    Qhw = C / C.sum(1, keepdims=True)
    lp = -beta * (Er - Er.min())
    pi = np.exp(lp); pi /= pi.sum()
    Fobs = mcmc.coassignment_observables(n)
    co_exact = (1 + pi @ Fobs) / 2
    tv = lambda p, q: 0.5 * np.abs(p - q).sum(-1)
    m = {"rank": d["rank"], "key": d["key"], "backend": d["transpile"]["backend"], "transpile": d["transpile"],
         "usage_quantum_seconds": d.get("usage_quantum_seconds"), "job_id": d.get("job_id"), "shots_per_row": shots}
    ideal = chain_metrics(Qid, Er, beta, Fobs, pi, co_exact)
    hw = chain_metrics(Qhw, Er, beta, Fobs, pi, co_exact)
    Ph = hastings(Qhw, Er, beta)
    gap_h = gap_general(Ph)
    orc = cand["gap_oracle"]
    m.update({"gap_ideal": ideal["gap"], "gap_ideal_selection_file": cand["gap_qaoa_a0"],
              "ideal_stationary_TV_check": ideal["stationary_TV"], "gap_oracle": orc, "oracle_chain": cand["oracle_chain"],
              "gap_quench_ideal": cand["gap_quench"], "gaps_classical": cand["gaps_classical"],
              "gap_hw": hw["gap"], "retention": hw["gap"] / ideal["gap"], "advantage": hw["gap"] / orc,
              "advantage_ideal": ideal["gap"] / orc,
              "stationary_TV_to_gibbs": hw["stationary_TV"], "coassign_max_err": hw["coassign_max_err"],
              "coassign_mean_err": hw["coassign_mean_err"],
              "hastings_gap": gap_h, "hastings_retention": gap_h / ideal["gap"], "hastings_advantage": gap_h / orc,
              "hastings_stationary_TV_check": float(tv(stationary(Ph), pi)),
              "row_TV_mean": float(tv(Qhw, Qid).mean()),
              "transpose_asym_rel": float(np.linalg.norm(Qhw - Qhw.T) / np.linalg.norm(Qhw)),
              "gibbs": {"p_max": float(pi.max()), "n_eff": float(1 / np.sum(pi ** 2))}})
    # finite-shot null: ideal rows resampled at the same number of shots
    null = []
    for _ in range(K_NULL):
        Qs = np.array([rng.multinomial(shots, q) for q in Qid]) / shots
        cm = chain_metrics(Qs, Er, beta, Fobs, pi, co_exact)
        null.append([cm["gap"] / ideal["gap"], cm["coassign_max_err"], float(tv(Qs, Qid).mean()),
                     float(np.linalg.norm(Qs - Qs.T) / np.linalg.norm(Qs))])
    null = np.array(null)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.median(a)), float(np.percentile(a, 97.5))]
    m["finite_shot_null"] = {"retention": q(null[:, 0]), "coassign_max_err": q(null[:, 1]), "row_TV": q(null[:, 2]),
                             "transpose_asym": q(null[:, 3])}
    # parametric bootstrap of the hardware rows
    boot = []
    for _ in range(K_BOOT):
        Qb = np.array([rng.multinomial(shots, q_) for q_ in Qhw]) / shots
        cm = chain_metrics(Qb, Er, beta, Fobs, pi, co_exact)
        boot.append([cm["gap"] / ideal["gap"], cm["gap"] / orc, cm["coassign_max_err"]])
    boot = np.array(boot)
    m["bootstrap_95"] = {"retention": q(boot[:, 0]), "advantage": q(boot[:, 1]), "coassign_max_err": q(boot[:, 2])}
    m["H1_retention"] = bool(m["retention"] >= 0.5)
    m["H2_advantage"] = bool(m["advantage"] > 1.0)
    m["H3_bias"] = bool(m["coassign_max_err"] <= 0.05)
    m["all_criteria"] = bool(m["H1_retention"] and m["H2_advantage"] and m["H3_bias"])
    return m


def main():
    tag = sys.argv[1]
    S = json.load(open(os.path.join(RES, "e5_selection.json")))
    cands = {r["selected_rank"]: r for r in S["candidates"] if r.get("selected_rank")}
    rng = np.random.default_rng(31)
    out = {"tag": tag, "instances": []}
    for rank in (1, 2, 3):
        p = os.path.join(RES, f"e5_n8_{rank}_{tag}.json")
        if not os.path.exists(p):
            print("missing", p)
            continue
        out["instances"].append(analyse(json.load(open(p)), cands[rank], rng))
    ok = sum(m["all_criteria"] for m in out["instances"])
    out["instances_meeting_all_criteria"] = ok
    out["E5_success"] = bool(len(out["instances"]) == 3 and ok >= 2)
    json.dump(out, open(os.path.join(RES, f"e5_n8_{tag}_metrics.json"), "w"), indent=1, default=str)
    for m in out["instances"]:
        print(f"rank {m['rank']}: gap hw {m['gap_hw']:.4f} ideal {m['gap_ideal']:.4f} oracle {m['gap_oracle']:.4f} | "
              f"retention {m['retention']:.2f} {m['bootstrap_95']['retention']} | advantage {m['advantage']:.2f} "
              f"(ideal {m['advantage_ideal']:.2f}) | bias max {m['coassign_max_err']:.3f} mean {m['coassign_mean_err']:.3f} "
              f"| rowTV {m['row_TV_mean']:.3f} (null {m['finite_shot_null']['row_TV'][1]:.3f}) | "
              f"Hastings ret {m['hastings_retention']:.2f} | H1-3 {m['H1_retention']} {m['H2_advantage']} {m['H3_bias']}")
    print("instances meeting all criteria:", ok, "| E5 success:", out["E5_success"])


if __name__ == "__main__":
    main()
