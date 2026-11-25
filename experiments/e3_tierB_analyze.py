"""E3 Tier B endpoints (docs/E3_scaling_protocol_2026-11-21.md, amendment 10), written before any Tier B data.

Per instance: rho(T) = 2 log2(RMSE_oracle(T) / RMSE_quantum(T)), RMSE over pairs of the pooled co-assignment estimate,
oracle = min RMSE over {local, label_swap, uniform, sw, wolff, heat_bath (i.i.d., cost 2^(n-1) per sample)} at the
same number of steps T (per-step accounting).
Endpoint B per modality (FC = FC200+FC400, SC = SC200+SC400): PASS iff
  (i)  slope of mean rho over n in {12, 14, 16} > 0, 95% bootstrap CI (1000 resamples of instances within n) > 0
  (ii) mean rho at n = 14 > 0, 95% bootstrap CI > 0.
Descriptive: mean rho at n = 16; rho at n*T for the classical chains (sweep-cost sensitivity); at n = 12 the Spearman
correlation and mean difference between simulated rho and the exact log2 asymptotic-variance ratio (Tier A rows).
Amendment 13 (22 Nov 2026, added before any Tier B data at n = 14-16): prospective two-factor routing test against the
sealed predictions in results/e3_tierB_routing_predictions.json (share rho > 0, predicted wins vs rest, one-sided Fisher).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
CL = ["local", "label_swap", "uniform", "sw", "wolff", "heat_bath"]
rng = np.random.default_rng(23)


def rho(r, t_cl, t_q):
    q = r["rmse"]["quantum"][str(t_q)]
    cl = [r["rmse"][c].get(str(t_cl)) for c in CL]
    cl = [v for v in cl if v is not None]
    return 2 * np.log2(min(cl) / q)


def endpoint(pairs, ns=(12, 14, 16), at=14):
    by = defaultdict(list)
    for n, v in pairs:
        by[n].append(v)
    ns = [n for n in ns if len(by[n]) >= 2]
    if len(ns) < 2:
        return None

    def stat(b):
        m = [np.mean(b[n]) for n in ns]
        return np.polyfit(ns, m, 1)[0], np.mean(b[at]) if at in b and len(b[at]) else np.nan

    s0, e0 = stat(by)
    boots = np.array([stat({n: [v[i] for i in rng.integers(0, len(v), len(v))] for n, v in by.items()}) for _ in range(1000)])
    ci = lambda x: [round(float(v), 3) for v in np.nanpercentile(x, [2.5, 97.5])]
    return {"slope": round(float(s0), 3), "slope_ci95": ci(boots[:, 0]), f"mean_rho_at_n{at}": round(float(e0), 3),
            f"mean_rho_at_n{at}_ci95": ci(boots[:, 1]),
            "mean_rho_by_n": {n: round(float(np.mean(by[n])), 3) for n in sorted(by) if by[n]},
            "instances_by_n": {n: len(by[n]) for n in sorted(by) if by[n]},
            "passes": bool(np.nanpercentile(boots[:, 0], 2.5) > 0 and np.nanpercentile(boots[:, 1], 2.5) > 0)}


def main():
    rows = [json.loads(l) for l in open(os.path.join(RES, "e3_tierB_rows.jsonl"))]
    mod = lambda r: r["family"].split("_")[-1][:2]
    S = {"protocol": "docs/E3_scaling_protocol_2026-11-21.md (amendment 10)", "rule": __doc__, "instances": len(rows),
         "modalities": {}}
    for m in ("FC", "SC"):
        rr = [r for r in rows if mod(r) == m]
        prim = [(r["n"], rho(r, r["T"], r["T"])) for r in rr]
        sens = [(r["n"], rho(r, r["T"] * r["n"], r["T"])) for r in rr]
        S["modalities"][m] = {"instances": len(rr), "PRIMARY_endpoint_B": endpoint(prim),
                              "sensitivity_classical_nT_steps": endpoint(sens),
                              "mean_rho_n16": round(float(np.mean([v for n, v in prim if n == 16])), 3) if any(n == 16 for n, _ in prim) else None,
                              "best_classical_counts": dict(sorted({c: sum(1 for r in rr if min(
                                  (r["rmse"][x].get(str(r["T"])) or np.inf, x) for x in CL)[1] == c) for c in CL}.items()))}
    # validation at n = 12 against the exact asymptotic-variance ratio of the same instances (Tier A)
    ta = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(os.path.join(RES, "e3_tierA_rows.jsonl"))}
    pairs = []
    for r in rows:
        a = ta.get(tuple(r["key"]))
        if r["n"] == 12 and a is not None:
            s2c = [a[f"b1.0_{c}_s2mean"] for c in CL[:-1]] + [a["b1.0_varmean"] * 2.0 ** 11]
            pairs.append((rho(r, r["T"], r["T"]), float(np.log2(min(s2c) / a["b1.0_quantum_s2mean"]))))
    if len(pairs) >= 3:
        from scipy.stats import spearmanr
        x, y = np.array(pairs).T
        S["validation_n12_simulated_vs_exact"] = {"pairs": len(pairs), "spearman": round(float(spearmanr(x, y)[0]), 3),
                                                  "mean_simulated_rho": round(float(x.mean()), 3),
                                                  "mean_exact_log2_s2_ratio": round(float(y.mean()), 3)}
    # amendment 13 (22 Nov 2026, before any Tier B data at n = 14-16): sealed two-factor routing predictions
    pp = os.path.join(RES, "e3_tierB_routing_predictions.json")
    if os.path.exists(pp):
        P = json.load(open(pp))
        pred = {tuple(x["key"]): x for x in P["instances"]}
        got = [(pred[tuple(r["key"])]["predicted_quantum_win"], rho(r, r["T"], r["T"])) for r in rows
               if r["n"] in (14, 16) and tuple(r["key"]) in pred]
        if got:
            from scipy.stats import fisher_exact
            a = sum(1 for p_, v in got if p_ and v > 0); b = sum(1 for p_, v in got if p_ and v <= 0)
            c = sum(1 for p_, v in got if not p_ and v > 0); d = sum(1 for p_, v in got if not p_ and v <= 0)
            S["amendment13_routing_test"] = {
                "predictions_written_at": P["written_at"], "rows_existing_when_written": P["tierB_rows_at_n14_16_when_written"],
                "instances": len(got), "predicted_win_class": {"wins": a, "losses": b,
                                                               "mean_rho": round(float(np.mean([v for p_, v in got if p_])), 3) if a + b else None},
                "rest": {"wins": c, "losses": d, "mean_rho": round(float(np.mean([v for p_, v in got if not p_])), 3) if c + d else None},
                "fisher_one_sided_p": float(fisher_exact([[a, b], [c, d]], alternative="greater")[1]),
                "accuracy": round((a + d) / len(got), 3)}
    S["E3_endpoint_B"] = {m: (S["modalities"][m]["PRIMARY_endpoint_B"] or {}).get("passes") for m in ("FC", "SC")}
    json.dump(S, open(os.path.join(RES, "e3_tierB_summary.json"), "w"), indent=1)
    print(json.dumps(S, indent=1)[:6000])


if __name__ == "__main__":
    main()
