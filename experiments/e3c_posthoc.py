"""E3c POST HOC robustness checks. NOT pre-registered: written on 22 Nov 2026 after e3c_analyze.py study had run.
Nothing here enters endpoint C; the report labels every number from this script as post hoc.

Usage: e3c_posthoc.py    -> results/e3c_posthoc.json

  - per dataset (FC-200, FC-400, SC-200, SC-400): mean dr and r(13), bootstrap CIs
  - per modality: exact one-sided sign test and one-sided Wilcoxon signed-rank test on dr
  - per modality: leave-one-pair-out (jackknife) range of mean dr and of its bootstrap CI lower bound
  - per modality: paired change of log2 gap of the best classical sampler (the oracle) and of the quantum proposal,
    and the share of pairs whose oracle chain differs between n = 12 and n = 13
  - per modality: paired change of the local chain's log2 gap, split by which chain is the oracle at n = 12
Bootstrap: 10,000 resamples, seed 43, percentile 95 % intervals (same scheme as e3c_analyze.py, different seed).
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from instances import RES  # noqa: E402
from e3c_analyze import CL, mean_ci, r_of  # noqa: E402


def oracle(row):
    n = row["n"]
    gaps = {c: row[f"b1.0_{c}_lumped"] for c in CL}
    gaps["heat_bath"] = 2.0 ** -(n - 1)
    best = max(gaps, key=gaps.get)
    return best, gaps[best]


def main():
    rows = [json.loads(l) for l in open(os.path.join(RES, "e3c_rows.jsonl"))]
    par = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(os.path.join(RES, "e3_tierA_rows.jsonl"))}
    rng = np.random.default_rng(43)
    out = {"note": "post hoc, not pre-registered; does not enter endpoint C", "by_dataset": {}, "by_modality": {}}
    for ds in ("FC200", "FC400", "SC200", "SC400"):
        g = [r for r in rows if r["ds"] == ds]
        dr = [r_of(r) - r_of(par[tuple(r["parent_key"])]) for r in g]
        out["by_dataset"][ds] = {"dr": mean_ci(dr, rng), "r13": mean_ci([r_of(r) for r in g], rng),
                                 "dr_positive": int(sum(x > 0 for x in dr)), "pairs": len(g)}
    for mod in ("FC", "SC"):
        g = [r for r in rows if r["ds"].startswith(mod)]
        ps = [par[tuple(r["parent_key"])] for r in g]
        dr = np.array([r_of(r) - r_of(p) for r, p in zip(g, ps)])
        k, m = int((dr > 0).sum()), int((dr != 0).sum())
        jack = []
        for i in range(len(dr)):
            x = np.delete(dr, i)
            b = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
            jack.append((float(x.mean()), float(np.percentile(b, 2.5))))
        orc = [(oracle(r), oracle(p)) for r, p in zip(g, ps)]
        d_orc = [float(np.log2(a[1] / b[1])) for a, b in orc]
        d_q = [float(np.log2(r["b1.0_quantum_lumped"] / p["b1.0_quantum_lumped"])) for r, p in zip(g, ps)]
        out["by_modality"][mod] = {
            "sign_test": {"positive": k, "nonzero": m, "p_one_sided": float(binomtest(k, m, 0.5, alternative="greater").pvalue)},
            "wilcoxon_p_one_sided": float(wilcoxon(dr, alternative="greater").pvalue),
            "jackknife_mean_dr_range": [min(j[0] for j in jack), max(j[0] for j in jack)],
            "jackknife_min_ci_low": min(j[1] for j in jack),
            "dlog2_gap_oracle": mean_ci(d_orc, rng), "dlog2_gap_quantum": mean_ci(d_q, rng),
            "oracle_chain_changed_share": float(np.mean([a[0] != b[0] for a, b in orc])),
            "oracle_chain_n12": {c: sum(b[0] == c for _, b in orc) for c in list(CL) + ["heat_bath"]},
            "oracle_chain_n13": {c: sum(a[0] == c for a, _ in orc) for c in list(CL) + ["heat_bath"]}}
        d_loc = [float(np.log2(r["b1.0_local_lumped"] / p["b1.0_local_lumped"])) for r, p in zip(g, ps)]
        out["by_modality"][mod]["dlog2_gap_local_by_oracle_n12"] = {
            c: {"pairs": sum(b[0] == c for _, b in orc), "mean": float(np.mean([x for x, (_, b) in zip(d_loc, orc) if b[0] == c]))}
            for c in sorted({b[0] for _, b in orc})}
    json.dump(out, open(os.path.join(RES, "e3c_posthoc.json"), "w"), indent=1)

    def f(x):
        return f"{x['mean']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]"
    for ds, v in out["by_dataset"].items():
        print(f"{ds}: dr {f(v['dr'])}  r13 {f(v['r13'])}  dr>0 {v['dr_positive']}/{v['pairs']}")
    for mod, v in out["by_modality"].items():
        print(f"{mod}: sign {v['sign_test']}  wilcoxon p {v['wilcoxon_p_one_sided']:.4f}  jackknife mean "
              f"[{v['jackknife_mean_dr_range'][0]:+.2f}, {v['jackknife_mean_dr_range'][1]:+.2f}]  min CI low "
              f"{v['jackknife_min_ci_low']:+.2f}")
        print(f"    dlog2 oracle {f(v['dlog2_gap_oracle'])}  quantum {f(v['dlog2_gap_quantum'])}  oracle changed "
              f"{v['oracle_chain_changed_share']:.0%}  n12 {v['oracle_chain_n12']}  n13 {v['oracle_chain_n13']}")


if __name__ == "__main__":
    main()
