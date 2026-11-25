"""E7 analysis (docs/E7_direct_sampling_protocol_2026-11-23.md), fixed before any E7 row existed and run unchanged.
Reads results/e7_rows.jsonl; writes results/e7_summary.json and results/e7_parcel_map.csv; prints the tables.

Floor of a sampler = worst-pair infinite-shot co-assignment error max_ij |p_ij(sampler) - p_ij(Gibbs)|.
E7-K (gate)  every known-answer check passes on every network (thresholds below).
E7-A         primary set (26 networks), beta_tilde = 1: the best DEPLOYABLE floor (minimum over the 15 configurations,
             chosen per network with hindsight) exceeds EPS on at least 14 of the 26 networks.
E7-B         same, for the oracle-tuned QAOA floor (minimum over depths 1-4).
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from instances import RES  # noqa: E402
from e7_run import DEPLOYABLE, ETAS, ORACLE, P_MAX, T_GRID  # noqa: E402

EPS = 0.02
N_PRIMARY, MAJORITY_PRIMARY = 26, 14
N_SECONDARY, MAJORITY_SECONDARY = 18, 10
TOL_EXACT, TOL_TROTTER, TOL_REL = 1e-10, 1e-3, 1e-8
HEADLINE = ["optimiser", "greedy", "anneal_T100", "qaoa_energy_p4", "qaoa_kl_p4", "qaoa_oracle_p4"]
BTS = ("1.0", "2.0")


def load():
    rows = [json.loads(l) for l in open(os.path.join(RES, "e7_rows.jsonl"))]
    assert len(rows) == N_PRIMARY + N_SECONDARY and len({tuple(r["key"]) for r in rows}) == len(rows)
    return sorted(rows, key=lambda r: r["index"])


def gate(rows):
    fails = []
    for r in rows:
        tag = f"{r['ds']} {r['roi']}"
        for name, c in r["checks"].items():
            if not (c["norm"] <= TOL_EXACT and c["flip"] <= TOL_EXACT and c.get("trotter", 0.0) <= TOL_TROTTER):
                fails.append([tag, name, c])
        for bt in BTS:
            b = r[f"b{bt}"]
            for name, c in b["checks"].items():
                if name == "p_vs_stored":
                    if c > TOL_EXACT:
                        fails.append([tag, bt, name, c])
                elif not (c["norm"] <= TOL_EXACT and c["flip"] <= TOL_EXACT):
                    fails.append([tag, bt, name, c])
            qe = b["qemcmc"]
            if not (qe["a0_stationarity"] <= TOL_EXACT and qe["a0_row_sum"] <= TOL_EXACT):
                fails.append([tag, bt, "a0 stationarity", qe["a0_stationarity"], qe["a0_row_sum"]])
            for c in ("quantum", "qaoa_a0"):
                if not qe[c]["gap"] > 0:
                    fails.append([tag, bt, c, "gap", qe[c]["gap"]])
            if "a0_recomputed" in qe:
                for k in ("gap", "s2max"):
                    if abs(qe["a0_recomputed"][k] - qe["qaoa_a0"][k]) > TOL_REL * abs(qe["qaoa_a0"][k]):
                        fails.append([tag, bt, "a0 recomputed", k, qe["a0_recomputed"][k], qe["qaoa_a0"][k]])
    n_checks = sum(len(r["checks"]) + sum(len(r[f"b{bt}"]["checks"]) + 3 for bt in BTS) for r in rows)
    return fails, n_checks


def best(r, bt, names):
    S = r[f"b{bt}"]["samplers"]
    k = min(names, key=lambda nm: S[nm]["floor"])
    return S[k]["floor"], k


def count_endpoint(rows, bt, names, majority):
    fl = [best(r, bt, names)[0] for r in rows]
    k = int(sum(f > EPS for f in fl))
    return {"networks": len(rows), "above_eps": k, "majority_needed": majority, "passes": k >= majority,
            "binomial_p_one_sided": float(stats.binomtest(k, len(rows), 0.5, alternative="greater").pvalue),
            "parcels_in_networks_above_eps": int(sum(r["n"] for r, f in zip(rows, fl) if f > EPS)),
            "parcels": int(sum(r["n"] for r in rows)), "median_floor": float(np.median(fl)),
            "floors": {f"{r['ds']} {r['roi']}": float(f) for r, f in zip(rows, fl)}}


def q(x, *ps):
    x = np.asarray(x, float)
    return [float(np.percentile(x, p)) for p in ps]


def sampler_table(rows, bt):
    out = {}
    for nm in DEPLOYABLE + ORACLE:
        S = [r[f"b{bt}"]["samplers"][nm] for r in rows]
        fl = [s["floor"] for s in S]
        out[nm] = {"median_floor": float(np.median(fl)), "iqr_floor": q(fl, 25, 75), "min_floor": float(np.min(fl)),
                   "max_floor": float(np.max(fl)), "share_above_eps": float(np.mean(np.array(fl) > EPS)),
                   "median_rms": float(np.median([s["rms"] for s in S])), "median_tv": float(np.median([s["tv"] for s in S])),
                   "median_kl_P_pi_bits": float(np.median([s["kl_P_pi_bits"] for s in S])),
                   "median_log10_suppression": float(np.median(np.log10([s["suppression_ratio"] for s in S]))),
                   "median_p_ground": float(np.median([s["p_ground"] for s in S])),
                   "mean_du": float(np.mean([s["du_mean"] for s in S])),
                   "share_du_negative": float(np.mean([s["du_mean"] < 0 for s in S])),
                   "median_precise_but_wrong": float(np.median([s["precise_but_wrong"] for s in S])),
                   "share_precise_but_wrong_gt3": float(np.mean([s["precise_but_wrong"] > 3 for s in S])),
                   "finite_calls": int(sum(np.isfinite(s["calls_to_eps"]) for s in S))}
    return out


def _calls(x):
    """E7 amendment 1: the run stored 0 calls where a deterministic sampler met the target on every pair; the
    protocol's 'fewest shots' is 1 there (qsr2/direct.calls_to_eps now returns >= 1)."""
    return max(1.0, x)


def calls_table(rows, bt):
    out = []
    for r in rows:
        b = r[f"b{bt}"]
        dc = {nm: _calls(b["samplers"][nm]["calls_to_eps"]) for nm in DEPLOYABLE}
        k = min(dc, key=dc.get)
        out.append({"network": f"{r['ds']} {r['roi']}", "n": r["n"], "direct_best_calls": dc[k], "direct_best": k,
                    "qemcmc_a0_calls": b["qemcmc"]["qaoa_a0"]["calls"], "qemcmc_quench_calls": b["qemcmc"]["quantum"]["calls"]})
    fin = [o for o in out if np.isfinite(o["direct_best_calls"])]
    return {"rows": out, "networks_direct_finite": len(fin),
            "median_ratio_a0_over_direct_where_finite": float(np.median([o["qemcmc_a0_calls"] / o["direct_best_calls"] for o in fin])) if fin else None,
            "median_qemcmc_a0_calls": float(np.median([o["qemcmc_a0_calls"] for o in out])),
            "median_qemcmc_quench_calls": float(np.median([o["qemcmc_quench_calls"] for o in out]))}


def parcel_map(rows, bt="1.0"):
    recs = []
    for r in rows:
        b = r[f"b{bt}"]
        _, kbest = best(r, bt, DEPLOYABLE)
        for j, node in enumerate(r["nodes"]):
            rec = {"ds": r["ds"], "roi": r["roi"], "n": r["n"], "parcel": node,
                   "label": r["labels"][j] if r.get("labels") else "", "u_true": b["u"][j]}
            for nm in HEADLINE:
                rec[f"u_{nm}"] = b["samplers"][nm]["u"][j]
            rec["u_best_deployable"] = b["samplers"][kbest]["u"][j]
            rec["best_deployable"] = kbest
            recs.append(rec)
    summ = {}
    for ds in sorted({x["ds"] for x in recs}):
        R = [x for x in recs if x["ds"] == ds]
        ut = np.array([x["u_true"] for x in R])
        s = {"parcels": len(R), "true": {"mean_u": float(ut.mean()), "share_u_gt_0.1": float(np.mean(ut > 0.1))}}
        for nm in HEADLINE + ["best_deployable"]:
            ud = np.array([x[f"u_{nm}"] for x in R])
            rho = stats.spearmanr(ud, ut).statistic if np.std(ud) > 0 else float("nan")
            s[nm] = {"mean_u": float(ud.mean()), "share_u_gt_0.1": float(np.mean(ud > 0.1)), "spearman_vs_true": float(rho),
                     "mean_abs_du": float(np.mean(np.abs(ud - ut))), "mean_du": float(np.mean(ud - ut))}
        summ[ds] = s
    return recs, summ


def diagnostics(rows):
    out = {"first_vs_final_floor": {}, "oracle_second_set_gain_gt_1e-6": 0, "single_start_optima": {}}
    for kind in ("energy", "kl"):
        dl = []
        for r in rows:
            for bt in BTS:
                b = r[f"b{bt}"]
                for p in range(1, P_MAX + 1):
                    nm = f"qaoa_{kind}_p{p}"
                    dl.append(abs(b["first_set_floor"][nm] - b["samplers"][nm]["floor"]))
        out["first_vs_final_floor"][kind] = {"max": float(np.max(dl)), "count_gt_0.005": int(np.sum(np.array(dl) > 0.005)),
                                             "cases": len(dl)}
    for r in rows:
        for bt in BTS:
            for p, rec in r[f"b{bt}"]["opt_oracle"].items():
                out["oracle_second_set_gain_gt_1e-6"] += int(rec["smooth_first"] - rec["smooth_best"] > 1e-6)
    for kind in ("energy", "kl", "oracle"):
        c = t = 0
        for r in rows:
            recs = [r["opt_energy"]] if kind == "energy" else [r[f"b{bt}"][f"opt_{kind}"] for bt in BTS]
            for rr in recs:
                for rec in rr.values():
                    c += int(rec["n_within_1e-6"] == 1)
                    t += 1
        out["single_start_optima"][kind] = [c, t]
    return out


def main():
    rows = load()
    prim = [r for r in rows if r["set"] == "primary"]
    sec = [r for r in rows if r["set"] == "secondary"]
    assert len(prim) == N_PRIMARY and len(sec) == N_SECONDARY
    fails, n_checks = gate(rows)
    A = count_endpoint(prim, "1.0", DEPLOYABLE, MAJORITY_PRIMARY)
    B = count_endpoint(prim, "1.0", ORACLE, MAJORITY_PRIMARY)
    recs, mapsum = parcel_map(prim)
    sens_path = os.path.join(RES, "e7_sensitivity.json")
    if os.path.exists(sens_path):
        sens = json.load(open(sens_path))
        newf = {x["network"]: x["new_best_floor"] for x in sens["networks"]}
        fl = {k: newf.get(k, v) for k, v in A["floors"].items()}
        kS = int(sum(v > EPS for v in fl.values()))
        sensitivity = {"networks_researched": len(sens["networks"]), "above_eps": kS, "passes": kS >= MAJORITY_PRIMARY,
                       "changed": {k: [A["floors"][k], v] for k, v in newf.items()}}
    else:
        sensitivity = None
    a_pass = A["passes"] and (sensitivity is None or sensitivity["passes"])   # the re-search can only lower floors
    if fails:
        outcome = "E7-K failed: investigate before reporting (protocol section 8)"
    elif a_pass and B["passes"]:
        outcome = "row 1: no direct sampler meets the target on most networks, even with oracle-tuned angles"
    elif a_pass:
        outcome = "row 2: deployable direct samplers miss the target on most networks; oracle-tuned QAOA can meet it"
    else:
        outcome = "row 3: some deployable direct sampler meets the target on most networks"
    S = {
        "eps": EPS, "gate": {"passes": not fails, "failures": fails, "checks_evaluated": n_checks},
        "E7_A": A, "E7_B": B, "outcome": outcome, "E7_A_sensitivity": sensitivity,
        "secondary": {
            "replication_secondary_set": {"A": count_endpoint(sec, "1.0", DEPLOYABLE, MAJORITY_SECONDARY),
                                          "B": count_endpoint(sec, "1.0", ORACLE, MAJORITY_SECONDARY)},
            "beta2": {"primary_A": count_endpoint(prim, "2.0", DEPLOYABLE, MAJORITY_PRIMARY),
                      "primary_B": count_endpoint(prim, "2.0", ORACLE, MAJORITY_PRIMARY),
                      "secondary_A": count_endpoint(sec, "2.0", DEPLOYABLE, MAJORITY_SECONDARY),
                      "secondary_B": count_endpoint(sec, "2.0", ORACLE, MAJORITY_SECONDARY)},
            "best_deployable_sampler_counts": {bt: {k: sum(best(r, bt, DEPLOYABLE)[1] == k for r in rows)
                                                    for k in DEPLOYABLE if any(best(r, bt, DEPLOYABLE)[1] == k for r in rows)}
                                               for bt in BTS},
            "sampler_table_primary_b1": sampler_table(prim, "1.0"),
            "sampler_table_primary_b2": sampler_table(prim, "2.0"),
            "sampler_table_secondary_b1": sampler_table(sec, "1.0"),
            "calls_primary_b1": calls_table(prim, "1.0"),
            "calls_secondary_b1": calls_table(sec, "1.0"),
            "noise_primary_b1": {str(eta): int(sum(min(r["b1.0"]["noisy_floor"][f"{nm}|{eta}"] for nm in DEPLOYABLE) > EPS
                                                   for r in prim)) for eta in ETAS},
            "size_trend_all_b1": {"spearman_n_vs_best_deployable_floor": float(stats.spearmanr(
                [r["n"] for r in rows], [best(r, "1.0", DEPLOYABLE)[0] for r in rows]).statistic),
                "by_network": [[r["ds"], r["roi"], r["n"], best(r, "1.0", DEPLOYABLE)[0], best(r, "1.0", DEPLOYABLE)[1],
                                best(r, "1.0", ORACLE)[0]] for r in rows]},
            "depth_trend_primary_b1": {kind: [float(np.median([r["b1.0"]["samplers"][f"qaoa_{kind}_p{p}"]["floor"] for r in prim]))
                                              for p in range(1, P_MAX + 1)] for kind in ("energy", "kl", "oracle")},
            "anneal_trend_primary_b1": {f"T{T:g}": float(np.median([r["b1.0"]["samplers"][f"anneal_T{T:g}"]["floor"] for r in prim]))
                                        for T in T_GRID},
            "uncertainty_map_primary_b1": mapsum,
            "optimiser_diagnostics": diagnostics(rows),
        },
    }
    json.dump(S, open(os.path.join(RES, "e7_summary.json"), "w"), indent=1)
    with open(os.path.join(RES, "e7_parcel_map.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"E7-K gate: {'PASS' if not fails else 'FAIL'} ({n_checks} checks, {len(fails)} failures)")
    for f_ in fails[:20]:
        print("   ", f_)
    for nm, E in (("E7-A (best deployable)", A), ("E7-B (oracle QAOA)", B)):
        print(f"{nm}: floor > {EPS} on {E['above_eps']}/{E['networks']} networks (need {E['majority_needed']}), "
              f"{E['parcels_in_networks_above_eps']}/{E['parcels']} parcels, median floor {E['median_floor']:.3f}, "
              f"binomial p = {E['binomial_p_one_sided']:.2g} -> {'PASS' if E['passes'] else 'FAIL'}")
    print("Outcome:", outcome)
    print("E7-A sensitivity (re-searched borderline networks):", sensitivity)
    print("\nPer network (beta_tilde = 1): best deployable floor [sampler] | oracle floor")
    for ds, roi, n, fd, kd, fo in S["secondary"]["size_trend_all_b1"]["by_network"]:
        print(f"  {ds:6s} {roi:15s} n={n:2d}  {fd:.4f} [{kd}] | {fo:.4f}")
    print("\nSampler table (primary, beta_tilde = 1): median floor [IQR], share > eps, mean du, median TV, median log10 SR")
    for nm, t in S["secondary"]["sampler_table_primary_b1"].items():
        print(f"  {nm:16s} {t['median_floor']:.3f} [{t['iqr_floor'][0]:.3f}, {t['iqr_floor'][1]:.3f}]  {t['share_above_eps']:.2f}  "
              f"{t['mean_du']:+.3f}  {t['median_tv']:.3f}  {t['median_log10_suppression']:.2f}")
    rep = S["secondary"]["replication_secondary_set"]
    print(f"\nReplication (secondary set): A {rep['A']['above_eps']}/{rep['A']['networks']}, B {rep['B']['above_eps']}/{rep['B']['networks']}")
    b2 = S["secondary"]["beta2"]
    print(f"beta_tilde = 2: primary A {b2['primary_A']['above_eps']}/26, B {b2['primary_B']['above_eps']}/26; "
          f"secondary A {b2['secondary_A']['above_eps']}/18, B {b2['secondary_B']['above_eps']}/18")
    print("Depth trend (median floor, p = 1..4):", S["secondary"]["depth_trend_primary_b1"])
    print("Anneal trend (median floor):", S["secondary"]["anneal_trend_primary_b1"])
    print("Noise (networks with best deployable floor > eps):", S["secondary"]["noise_primary_b1"])
    c = S["secondary"]["calls_primary_b1"]
    print(f"Calls: direct finite on {c['networks_direct_finite']}/26; median QeMCMC calls a0 {c['median_qemcmc_a0_calls']:.3g}, "
          f"quench {c['median_qemcmc_quench_calls']:.3g}; median a0/direct ratio where finite {c['median_ratio_a0_over_direct_where_finite']}")
    print("Uncertainty map:", json.dumps(mapsum, indent=1))
    print("Optimiser diagnostics:", S["secondary"]["optimiser_diagnostics"])


if __name__ == "__main__":
    main()
