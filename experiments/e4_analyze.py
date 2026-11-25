"""E4 analysis (docs/E4_whole_cortex_protocol_2026-11-22.md), written before any E4 exact row existed.

Usage: e4_analyze.py exact|full    -> results/e4_summary.json, e4_roi_table.csv (+ e4_parcel_map.csv for full)

DEFINITIONS (beta_tilde = 1 unless stated).
  Cost of chain X on ROI R: c_X(R) = max over the ROI's pairs of the exact asymptotic variance s2_ij of the
    co-assignment estimator (z units). The proposals X needs for RMSE eps on EVERY p_ij of R are c_X(R) / (4 eps^2),
    so cost ratios do not depend on eps. Heat-bath counts 2^(n-1) proposals per i.i.d. draw (c = max var * 2^(n-1)).
  Classical oracle C*(R): the classical chain (local, label-swap, uniform, Swendsen-Wang, Wolff, heat-bath) with the
    smallest c(R). PT is excluded, as in E3 endpoint A.
  QSR 2.0 hybrid H: the untuned quench on ROIs routed by the SEALED two-factor rule
    (results/e4_routing_predictions.json), C* on every other ROI.
PRIMARY E4-A, per primary tiling (Schaefer-100 FC, SC): rho = sum_R c_H(R) / sum_R c_C*(R).
  E4-A passes for a tiling iff rho < 1; "material saving" iff rho <= 0.5.
E4-B VERIFICATION (stage full; results/e4_run_rows.jsonl from experiments/e4_runs.py), per tiling and arm (H, C*):
  pairs entering V1/V2: min(p_ij, 1 - p_ij) >= 0.01.
  V1 exactness: max |z| < z_thr, z = sqrt(R) (mean_r p_hat - p) / RMSE_pred, z_thr = Bonferroni two-sided at
     family-wise 0.001 over the entering pairs.
  V2 calibration: median over entering pairs of kappa = RMSE_realised / RMSE_pred lies in [0.85, 1.15].
  V3 target reached: on every ROI the binding pair (largest predicted s2) has kappa <= 1.3.
  rho_B = total proposals including burn-in, H / C*.
  End-to-end pass for a tiling iff E4-A passes and V1-V3 hold for both arms. The G2 end-to-end criterion is met
  iff the end-to-end pass holds on at least one primary tiling.
SECONDARY / DESCRIPTIVE (no gate): beta_tilde = 2 (with its own sealed routing); QAOA-MC angle 0 on routed ROIs;
  depolarising-noise quench with eps_noise = 0.2 and 0.5; oracle routing (per ROI the cheaper of quantum and C*);
  quantum on every ROI; mean-over-pairs aggregator; deployable single classical chain X on every ROI; leave-one-ROI-
  out range; ROIs never inspected before E4; the Schaefer-200 partial tilings; per-ROI routing accuracy.
"""
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from instances import RES  # noqa: E402

CLASSICAL = ("local", "label_swap", "uniform", "sw", "wolff", "heat_bath")
PRIMARY = ("FC100", "SC100")
SECONDARY = ("FC200", "SC200")
P_MIN = 0.01
FWER = 0.001
KAPPA_MEDIAN = (0.85, 1.15)
KAPPA_BINDING = 1.3


def cost(b, chain, agg=max):
    return float(agg(b["chains"][chain]["s2"]))


def oracle(b, agg=max):
    return min(CLASSICAL, key=lambda c: cost(b, c, agg))


def previously_inspected(ds, n):
    """ROIs whose per-ROI exact results were computed AND inspected before E4 was written: canonical ROIs of 4-10
    parcels in FC100/FC200/SC100/SC200 (G1, G1b, G1c) and of 7-12 parcels in FC200/SC200 (E3 Tier A). The 13-parcel
    ROIs were computed in the E3c canonical stage but not inspected before E4 was frozen."""
    return (ds in ("FC100", "FC200", "SC100", "SC200") and 4 <= n <= 10) or (ds in ("FC200", "SC200") and 7 <= n <= 12)


def rho(rows, routed, bt="b1.0", q="quantum", agg=max, mode="rule", classical=None):
    num = den = 0.0
    for r in rows:
        b = r[bt]
        cs = cost(b, classical, agg) if classical else cost(b, oracle(b, agg), agg)
        cq = cost(b, q, agg)
        den += cs
        num += {"rule": cq if routed[tuple(r["key"])] else cs, "oracle": min(cq, cs), "all": cq}[mode]
    return num / den


def tiling_block(rows, R1, R2):
    n_par = sum(r["n"] for r in rows)
    out = {"rois": len(rows), "parcels": n_par, "routed": [r["roi"] for r in rows if R1[tuple(r["key"])]]}
    out["rho"] = rho(rows, R1)
    out["E4A_pass"] = bool(out["rho"] < 1)
    out["material_saving"] = bool(out["rho"] <= 0.5)
    sec = {"beta2": rho(rows, R2, bt="b2.0"), "qaoa_a0": rho(rows, R1, q="qaoa_a0"),
           "noisy_0.2": rho(rows, R1, q="quantum_noisy_0.2"), "noisy_0.5": rho(rows, R1, q="quantum_noisy_0.5"),
           "oracle_routing": rho(rows, R1, mode="oracle"), "quantum_everywhere": rho(rows, R1, mode="all"),
           "mean_aggregator": rho(rows, R1, agg=np.mean)}
    sec["routing_efficiency"] = ((1 - out["rho"]) / (1 - sec["oracle_routing"])) if sec["oracle_routing"] < 1 else None
    tot = {c: sum(cost(r["b1.0"], c) for r in rows) for c in CLASSICAL}
    best_single = min(tot, key=tot.get)
    sec["deployable_single_chain"] = {c: rho(rows, R1, classical=c) for c in CLASSICAL}
    sec["best_single_classical_chain"] = best_single
    sec["oracle_vs_best_single_chain_cost"] = sum(cost(r["b1.0"], oracle(r["b1.0"])) for r in rows) / tot[best_single]
    loo = [rho([x for x in rows if x is not r], R1) for r in rows] if len(rows) > 1 else []
    sec["leave_one_roi_out_range"] = [min(loo), max(loo)] if loo else None
    fresh = [r for r in rows if not previously_inspected(r["ds"], r["n"])]
    sec["not_previously_inspected"] = {"rois": [r["roi"] for r in fresh], "rho": rho(fresh, R1) if fresh else None}
    good = [(R1[tuple(r["key"])], cost(r["b1.0"], "quantum") < cost(r["b1.0"], oracle(r["b1.0"]))) for r in rows]
    sec["routing_accuracy"] = {"routed_and_quantum_cheaper": sum(a and b for a, b in good),
                               "routed": sum(a for a, _ in good),
                               "not_routed_and_classical_cheaper": sum((not a) and (not b) for a, b in good),
                               "not_routed": sum(not a for a, _ in good)}
    out["secondary"] = sec
    table = []
    tot_c = sum(cost(r["b1.0"], oracle(r["b1.0"])) for r in rows)
    for r in rows:
        b = r["b1.0"]
        cs, cq = cost(b, oracle(b)), cost(b, "quantum")
        table.append({"ds": r["ds"], "roi": r["roi"], "n": r["n"], "F": r["F"], "routed": R1[tuple(r["key"])],
                      "routed_beta2": R2[tuple(r["key"])], "classical_oracle": oracle(b), "c_classical": cs,
                      "c_quantum": cq, "c_qaoa_a0": cost(b, "qaoa_a0"), "log2_c_classical_over_c_quantum": math.log2(cs / cq),
                      "share_of_classical_cost": cs / tot_c, "gap_quantum": b["chains"]["quantum"]["gap"],
                      "gap_classical_oracle": b["chains"][oracle(b)]["gap"], "previously_inspected": previously_inspected(r["ds"], r["n"])})
    return out, table


def verification(rows, runs, R1):
    """E4-B checks for one tiling. runs: {(roi_key, chain): run row}."""
    res = {}
    for arm in ("H", "C"):
        z, kap, binding, cost_tot, n_excl = [], [], [], 0.0, 0
        for r in rows:
            k = tuple(r["key"])
            ch = "quantum" if (arm == "H" and R1[k]) else oracle(r["b1.0"])
            x = runs[(k, ch)]
            p = np.array(x["p"]); keep = np.minimum(p, 1 - p) >= P_MIN
            n_excl += int((~keep).sum())
            z += list(np.array(x["z"])[keep]); kap += list(np.array(x["kappa"])[keep])
            ib = int(np.argmax(r["b1.0"]["chains"][ch]["s2"] if ch != "heat_bath" else r["b1.0"]["var"]))
            binding.append({"roi": r["roi"], "chain": ch, "kappa": float(x["kappa"][ib])})
            cost_tot += x["cost"]
        m = len(z)
        zthr = float(norm.isf(FWER / (2 * m))) if m else None
        res[arm] = {"pairs_entering": m, "pairs_excluded_near_certain": n_excl, "max_abs_z": float(np.max(np.abs(z))) if m else None,
                    "z_threshold": zthr, "V1": bool(m and np.max(np.abs(z)) < zthr),
                    "median_kappa": float(np.median(kap)) if m else None,
                    "V2": bool(m and KAPPA_MEDIAN[0] <= np.median(kap) <= KAPPA_MEDIAN[1]),
                    "binding": binding, "max_binding_kappa": max(b["kappa"] for b in binding),
                    "V3": bool(max(b["kappa"] for b in binding) <= KAPPA_BINDING), "total_proposals": cost_tot}
    res["rho_B"] = res["H"]["total_proposals"] / res["C"]["total_proposals"]
    res["checks_pass"] = all(res[a][v] for a in ("H", "C") for v in ("V1", "V2", "V3"))
    return res


def main():
    stage = sys.argv[1]
    rows = [json.loads(l) for l in open(os.path.join(RES, "e4_exact_rows.jsonl"))]
    P = json.load(open(os.path.join(RES, "e4_routing_predictions.json")))
    R1 = {tuple(x["key"]): x["routed_b1.0"] for x in P["instances"]}
    R2 = {tuple(x["key"]): x["routed_b2.0"] for x in P["instances"]}
    S = {"stage": stage, "routing_written_at": P["written_at"], "rows_when_routing_written": P["e4_exact_rows_when_written"],
         "tilings": {}}
    table = []
    for ds in PRIMARY + SECONDARY:
        g = [r for r in rows if r["ds"] == ds]
        if not g:
            continue
        S["tilings"][ds], t = tiling_block(g, R1, R2)
        S["tilings"][ds]["primary"] = ds in PRIMARY
        table += t
    runs = {}
    rp = os.path.join(RES, "e4_run_rows.jsonl")
    if stage == "full" and os.path.exists(rp):
        for l in open(rp):
            x = json.loads(l)
            runs[(tuple(x["key"]), x["chain"])] = x
        parcel = []
        for ds in PRIMARY:
            g = [r for r in rows if r["ds"] == ds]
            V = verification(g, runs, R1)
            S["tilings"][ds]["E4B"] = V
            S["tilings"][ds]["end_to_end_pass"] = bool(S["tilings"][ds]["E4A_pass"] and V["checks_pass"])
            for r in g:
                k = tuple(r["key"]); ch = "quantum" if R1[k] else oracle(r["b1.0"]); x = runs[(k, ch)]
                for i, (node, lab) in enumerate(zip(r["nodes"], r["labels"])):
                    parcel.append({"ds": ds, "parcel": node, "label": lab, "roi": r["roi"], "n_roi": r["n"], "routed": R1[k],
                                   "chain_H": ch, "u_exact": r["b1.0"]["u"][i], "u_hat_mean": x["u_hat_mean"][i],
                                   "u_hat_sd": x["u_hat_sd"][i]})
        with open(os.path.join(RES, "e4_parcel_map.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(parcel[0])); w.writeheader(); w.writerows(parcel)
        S["G2_end_to_end_met"] = any(S["tilings"][ds].get("end_to_end_pass", False) for ds in PRIMARY)
    S["E4A_pass_any_primary"] = any(S["tilings"][ds]["E4A_pass"] for ds in PRIMARY if ds in S["tilings"])
    json.dump(S, open(os.path.join(RES, "e4_summary.json"), "w"), indent=1)
    with open(os.path.join(RES, "e4_roi_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    for ds, v in S["tilings"].items():
        print(f"{ds}{' (primary)' if v['primary'] else ''}: rho {v['rho']:.3f}  E4-A pass {v['E4A_pass']}  routed {v['routed']}"
              + (f"  | E4-B rho_B {v['E4B']['rho_B']:.3f} checks {v['E4B']['checks_pass']}" if "E4B" in v else ""))


if __name__ == "__main__":
    main()
