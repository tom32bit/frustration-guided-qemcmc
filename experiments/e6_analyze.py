"""E6 analysis (docs/E6_generality_protocol_2026-11-22.md), written before any E6 row existed.

Usage: e6_analyze.py    -> results/e6_summary.json

r = log2(gap_quantum / gap_oracle) at beta_tilde = 1, partition space; oracle = best of local, label-swap, uniform,
Swendsen-Wang, Wolff and heat-bath at per-cost gap 2^-(n-1) (E3 endpoint-A definition; experiments/e3c_analyze.r_of).
PRIMARY
  E6-A  routing rule transfers: share of r > 0 among sealed predicted wins vs the rest, one-sided Fisher exact test
        over all E6 instances (sizes 7-12); passes iff p < 0.05.
  E6-B  advantage transfers: mean r over the predicted-win class > 0, 95 % CI excluding 0 (stratified bootstrap within
        (network, size) cells, 10,000 resamples, seed 53).
  E6 passes iff E6-A and E6-B pass.
  E6-C  prospective scaling projection (class all, metric r): the observed mean r at size 11 and at size 12 both lie
        inside the sealed 95 % prediction intervals of results/e6_projection.json. Governs how projections of the brain
        results may be presented (protocol section 7).
SECONDARY / DESCRIPTIVE: E6-C for the predicted-win class and for the steps-to-accuracy metric; projection errors;
  per network (instances, predicted wins, wins, mean r by class, Spearman rho(F, r)); pooled Spearman rho(F, r) (the
  plan's original E6 criterion, rho > 0.3); QAOA-MC angle 0 and steps-to-accuracy in the predicted-win class;
  beta_tilde = 2; means by size.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import fisher_exact, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from instances import RES  # noqa: E402
from e3c_analyze import r_of, s2_ratio  # noqa: E402

NB = 10000


def strat_mean_ci(vals, cells, rng):
    vals = np.asarray(vals, float)
    groups = {}
    for i, c in enumerate(cells):
        groups.setdefault(c, []).append(i)
    groups = [np.array(v) for _, v in sorted(groups.items())]
    boot = np.empty(NB)
    for b in range(NB):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        boot[b] = vals[idx].mean()
    return {"mean": float(vals.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n": int(len(vals))}


def main():
    rows = [json.loads(l) for l in open(os.path.join(RES, "e6_rows.jsonl"))]
    P = json.load(open(os.path.join(RES, "e6_routing_predictions.json")))
    pred = {tuple(x["key"]): x["predicted_quantum_win"] for x in P["instances"]}
    J = json.load(open(os.path.join(RES, "e6_projection.json")))
    rng = np.random.default_rng(53)
    for r in rows:
        r["_r"], r["_s2"], r["_r2"], r["_a0"] = r_of(r), s2_ratio(r), r_of(r, 2.0), r_of(r, 1.0, "qaoa_a0")
        r["_pw"] = pred[tuple(r["key"])]
    w = [r for r in rows if r["_pw"]]; o = [r for r in rows if not r["_pw"]]
    a, b = sum(r["_r"] > 0 for r in w), sum(r["_r"] > 0 for r in o)
    p = float(fisher_exact([[a, len(w) - a], [b, len(o) - b]], alternative="greater")[1]) if w and o else None
    S = {"rows": len(rows), "instances_sealed": len(P["instances"]), "routing_written_at": P["written_at"],
         "rows_when_routing_written": P["e6_rows_when_written"], "projection_written_at": J["written_at"],
         "projection_rows_by_size_when_written": J["rows_by_size_when_written"]}
    S["E6A"] = {"predicted_win": {"instances": len(w), "wins": a}, "rest": {"instances": len(o), "wins": b},
                "fisher_one_sided_p": p, "pass": bool(p is not None and p < 0.05)}
    eb = strat_mean_ci([r["_r"] for r in w], [(r["ds"], r["n"]) for r in w], rng) if w else None
    S["E6B"] = {"mean_r_predicted_win": eb, "pass": bool(eb is not None and eb["ci95"][0] > 0)}
    S["E6_pass"] = bool(S["E6A"]["pass"] and S["E6B"]["pass"])
    S["E6C"] = {}
    for name, proj in J["projections"].items():
        if proj is None:
            S["E6C"][name] = None
            continue
        cls, m = name.split("_", 1)[0] if not name.startswith("predicted") else "predicted_win", name.rsplit("_", 1)[1]
        res = {}
        for t, v in proj["targets"].items():
            sel = [r for r in rows if r["n"] == int(t) and (cls == "all" or r["_pw"])]
            obs = float(np.mean([r["_" + m] for r in sel])) if sel else None
            res[t] = {"observed_mean": obs, "instances": len(sel), "expected_instances": v["m"], "point": v["point"],
                      "pi95": v["pi95"], "inside": bool(obs is not None and v["pi95"][0] <= obs <= v["pi95"][1]),
                      "error": (obs - v["point"]) if obs is not None else None}
        S["E6C"][name] = {"slope": proj["slope"], "slope_ci95": proj["slope_ci95"], "targets": res,
                          "all_inside": all(x["inside"] for x in res.values())}
    S["E6C_pass"] = bool(S["E6C"].get("all_r") and S["E6C"]["all_r"]["all_inside"])
    nets = sorted({r["ds"] for r in rows})
    S["by_network"] = {}
    for ds in nets:
        g = [r for r in rows if r["ds"] == ds]
        gw = [r for r in g if r["_pw"]]
        S["by_network"][ds] = {"instances": len(g), "predicted_wins": len(gw), "wins_in_predicted": sum(r["_r"] > 0 for r in gw),
                               "wins_in_rest": sum(r["_r"] > 0 for r in g if not r["_pw"]),
                               "mean_r_predicted": float(np.mean([r["_r"] for r in gw])) if gw else None,
                               "mean_r_rest": float(np.mean([r["_r"] for r in g if not r["_pw"]])) if len(gw) < len(g) else None,
                               "spearman_F_r": float(spearmanr([r["F"] for r in g], [r["_r"] for r in g])[0]) if len(g) > 2 else None}
    S["spearman_F_r_pooled"] = float(spearmanr([r["F"] for r in rows], [r["_r"] for r in rows])[0])
    for k, lab in (("_a0", "qaoa_a0_r_predicted_win"), ("_s2", "s2_ratio_predicted_win"), ("_r2", "beta2_r_predicted_win")):
        S[lab] = strat_mean_ci([r[k] for r in w], [(r["ds"], r["n"]) for r in w], rng) if w else None
    S["by_size"] = {str(n): {"instances": sum(r["n"] == n for r in rows), "mean_r": float(np.mean([r["_r"] for r in rows if r["n"] == n])),
                             "mean_r_predicted_win": (float(np.mean([r["_r"] for r in w if r["n"] == n])) if any(r["n"] == n for r in w) else None),
                             "predicted_wins": sum(r["n"] == n for r in w)}
                    for n in sorted({r["n"] for r in rows})}
    json.dump(S, open(os.path.join(RES, "e6_summary.json"), "w"), indent=1)
    print("E6-A", S["E6A"], "\nE6-B", S["E6B"], "\nE6 pass", S["E6_pass"], "| E6-C pass", S["E6C_pass"])


if __name__ == "__main__":
    main()
