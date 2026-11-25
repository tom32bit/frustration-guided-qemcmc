"""Scaling projection of the brain results beyond the exact range (docs/E6_generality_protocol_2026-11-22.md,
section 7). DESCRIPTIVE: the brain data were seen before this procedure was written. The procedure itself
(qsr2.projection.project) was frozen with the E6 hashes, before it was applied here; its prospective test is E6-C.

Data: E3 Tier A random ROI node sets (exact, n = 7-12; FC = FC-200 + FC-400, SC = SC-200 + SC-400; 24 per modality and
size) and the E3c nested-growth sets (exact, n = 13; 24 per modality). Groups = datasets.
Metrics: r = log2(gap_quantum / gap_oracle) (E3 endpoint A) and s2 = log2(s2_best classical / s2_quantum) (steps to
accuracy; the quantity the simulated Tier B metric rho estimates).
Analyses per modality and metric (B = 10,000, seed 61):
  backtest_short  fit 7-10  -> predict the mean at 11 and 12 (observed: Tier A)
  backtest_13     fit 7-12  -> predict the mean at 13 (observed: E3c)
  projection      fit 7-13  -> project 14, 15, 16, and 20 (illustrative only)
  tierB           the s2 projections at 14 and 16 against the simulated Tier B mean rho (n = 14: mean and 95 % CI)
Writes results/scaling_projection_brain.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import projection  # noqa: E402
from instances import RES  # noqa: E402
from e3c_analyze import r_of, s2_ratio  # noqa: E402

B, SEED = 10000, 61
METRICS = {"r": r_of, "s2": s2_ratio}


def main():
    tierA = [json.loads(l) for l in open(os.path.join(RES, "e3_tierA_rows.jsonl"))]
    tierA = [r for r in tierA if r["family"].startswith("P3_roi_")]
    e3c = [json.loads(l) for l in open(os.path.join(RES, "e3c_rows.jsonl"))]
    TB = json.load(open(os.path.join(RES, "e3_tierB_summary.json")))["modalities"]
    out = {"note": __doc__, "B": B, "seed": SEED, "modalities": {}}
    for mod in ("FC", "SC"):
        A = [r for r in tierA if r["ds"].startswith(mod)]
        C = [r for r in e3c if r["ds"].startswith(mod)]
        res = {}
        for m, f in METRICS.items():
            def fit(rows, targets):
                return projection.project([r["n"] for r in rows], [f(r) for r in rows], [r["ds"] for r in rows],
                                          targets, B=B, seed=SEED)
            obs = {n: float(np.mean([f(r) for r in A if r["n"] == n])) for n in range(7, 13)}
            obs[13] = float(np.mean([f(r) for r in C]))
            bs = fit([r for r in A if r["n"] <= 10], {11: 24, 12: 24})
            b13 = fit(A, {13: len(C)})
            pr = fit(A + C, {14: 24, 15: 24, 16: 24, 20: 24})
            for blk, ns in ((bs, (11, 12)), (b13, (13,))):
                for n in ns:
                    t = blk["targets"][str(n)]
                    t["observed_mean"] = obs[n]
                    t["inside"] = bool(t["pi95"][0] <= obs[n] <= t["pi95"][1])
            res[m] = {"observed_means": {str(k): v for k, v in obs.items()}, "backtest_short": bs, "backtest_13": b13,
                      "projection": pr}
        tb = TB[mod]["PRIMARY_endpoint_B"]
        res["tierB"] = {"simulated_mean_rho_by_n": tb["mean_rho_by_n"], "n14_ci95": tb["mean_rho_at_n14_ci95"],
                        "instances_by_n": tb["instances_by_n"],
                        "s2_projection_n14": res["s2"]["projection"]["targets"]["14"],
                        "s2_projection_n16": res["s2"]["projection"]["targets"]["16"]}
        out["modalities"][mod] = res
    json.dump(out, open(os.path.join(RES, "scaling_projection_brain.json"), "w"), indent=1)
    for mod, res in out["modalities"].items():
        for m in METRICS:
            p = res[m]["projection"]
            print(f"{mod} {m}: slope {p['slope']:+.2f}/node {p['slope_ci95']}; backtest 11,12 inside "
                  f"{[res[m]['backtest_short']['targets'][k]['inside'] for k in ('11', '12')]}, 13 inside "
                  f"{res[m]['backtest_13']['targets']['13']['inside']}; projection 14-16 "
                  f"{[round(p['targets'][k]['point'], 2) for k in ('14', '15', '16')]}")


if __name__ == "__main__":
    main()
