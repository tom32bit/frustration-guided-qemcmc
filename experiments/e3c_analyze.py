"""E3c analysis (docs/E3c_exact_n13_protocol_2026-11-22.md), written before any n = 13 study row existed.

Usage: e3c_analyze.py pilot|study    -> results/e3c_<stage>_summary.json

Pairs: each nested instance (size n0 + 1) with its Tier A parent (size n0). r = log2(gap_quantum / gap_oracle) at
beta_tilde = 1, partition space; oracle = best of local, label-swap, uniform, Swendsen-Wang, Wolff and heat-bath at
per-cost gap 2^-(n-1) (the E3 endpoint-A definition, PT excluded). Modalities: FC = FC-200 + FC-400, SC = SC-200 +
SC-400 (24 pairs each). Bootstrap: 10,000 resamples of pairs within modality, seed 41, percentile 95 % CIs.

PRIMARY (study, per modality): (i) mean dr = r(13) - r(12) > 0, CI excluding 0; (ii) mean r(13) > 0, CI excluding 0.
Endpoint C passes for a modality iff both hold.
SECONDARY / DESCRIPTIVE: exact asymptotic-variance ratio at n = 13 and its paired change; beta_tilde = 2 (r and dr);
QAOA-MC angle 0 (r and dr); sealed two-factor routing predictions (one-sided Fisher on r(13) > 0, predicted win vs
rest) and dr within the predicted-win class; paired change of log2 gap per chain.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from instances import RES  # noqa: E402

CL = ("local", "label_swap", "uniform", "sw", "wolff")
NB = 10000


def r_of(row, bt=1.0, q="quantum"):
    n = row["n"]
    orc = max([row[f"b{bt}_{c}_lumped"] for c in CL] + [2.0 ** -(n - 1)])
    return float(np.log2(row[f"b{bt}_{q}_lumped"] / orc))


def s2_ratio(row, bt=1.0):
    n = row["n"]
    best = min([row[f"b{bt}_{c}_s2mean"] for c in CL] + [row[f"b{bt}_varmean"] * 2 ** (n - 1)])
    return float(np.log2(best / row[f"b{bt}_quantum_s2mean"]))


def mean_ci(x, rng):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    b = x[rng.integers(0, len(x), (NB, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "n": int(len(x)), "sd": float(x.std(ddof=1)) if len(x) > 1 else None}


def main():
    stage = sys.argv[1]
    fname = {"pilot": "e3c_pilot_rows.jsonl", "study": "e3c_rows.jsonl"}[stage]
    rows = [json.loads(l) for l in open(os.path.join(RES, fname))]
    parents = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(os.path.join(RES, "e3_tierA_rows.jsonl"))}
    rng = np.random.default_rng(41)
    pred = None
    pp = os.path.join(RES, "e3c_routing_predictions.json")
    if stage == "study" and os.path.exists(pp):
        P = json.load(open(pp))
        pred = {tuple(x["key"]): x["predicted_quantum_win"] for x in P["instances"]}
    pairs = []
    for r in rows:
        p = parents[tuple(r["parent_key"])]
        e = {"key": r["key"], "ds": r["ds"], "kind": r["kind"], "mod": r["ds"][:2], "n": r["n"], "F": r["F"],
             "r": r_of(r), "r_parent": r_of(p), "r_b2": r_of(r, 2.0), "r_b2_parent": r_of(p, 2.0),
             "r_a0": r_of(r, 1.0, "qaoa_a0"), "r_a0_parent": r_of(p, 1.0, "qaoa_a0"),
             "s2": s2_ratio(r), "s2_parent": s2_ratio(p)}
        for c in CL + ("quantum",):
            e[f"dlog2_{c}"] = float(np.log2(r[f"b1.0_{c}_lumped"] / p[f"b1.0_{c}_lumped"]))
        e["oracle_chain"] = max(list(CL) + ["heat_bath"], key=lambda c: r[f"b1.0_{c}_lumped"] if c != "heat_bath" else 2.0 ** -(r["n"] - 1))
        if pred is not None:
            e["predicted_win"] = pred[tuple(r["key"])]
        pairs.append(e)
    S = {"stage": stage, "pairs": len(pairs), "modalities": {}}
    for mod in ("FC", "SC"):
        g = [e for e in pairs if e["mod"] == mod]
        dr = [e["r"] - e["r_parent"] for e in g]
        m = {"dr": mean_ci(dr, rng), "r": mean_ci([e["r"] for e in g], rng),
             "r_parent": mean_ci([e["r_parent"] for e in g], rng),
             "win_share": float(np.mean([e["r"] > 0 for e in g])),
             "win_share_parent": float(np.mean([e["r_parent"] > 0 for e in g])),
             "dr_positive_share": float(np.mean([x > 0 for x in dr])),
             "s2_ratio": mean_ci([e["s2"] for e in g], rng), "ds2": mean_ci([e["s2"] - e["s2_parent"] for e in g], rng),
             "beta2_r": mean_ci([e["r_b2"] for e in g], rng), "beta2_dr": mean_ci([e["r_b2"] - e["r_b2_parent"] for e in g], rng),
             "qaoa_a0_r": mean_ci([e["r_a0"] for e in g], rng),
             "qaoa_a0_dr": mean_ci([e["r_a0"] - e["r_a0_parent"] for e in g], rng),
             "dlog2_gap_by_chain": {c: float(np.mean([e[f"dlog2_{c}"] for e in g])) for c in CL + ("quantum",)},
             "oracle_chain_counts": {c: sum(e["oracle_chain"] == c for e in g) for c in list(CL) + ["heat_bath"]},
             "by_kind": {k: {"dr": mean_ci([e["r"] - e["r_parent"] for e in g if e["kind"] == k], rng),
                             "r": mean_ci([e["r"] for e in g if e["kind"] == k], rng)} for k in ("border-set", "within-set")}}
        if stage == "study":
            m["endpoint_C_i"] = bool(m["dr"]["ci95"][0] > 0)
            m["endpoint_C_ii"] = bool(m["r"]["ci95"][0] > 0)
            m["endpoint_C_pass"] = bool(m["endpoint_C_i"] and m["endpoint_C_ii"])
        S["modalities"][mod] = m
    S["pooled_FC_SC"] = {"dr": mean_ci([e["r"] - e["r_parent"] for e in pairs], rng),
                         "r": mean_ci([e["r"] for e in pairs], rng)}
    cp = os.path.join(RES, "e3c_canon_rows.jsonl")
    if stage == "study" and os.path.exists(cp):          # optional descriptive extension: canonical 13-parcel ROIs
        S["canonical_13"] = [{"ds": r["ds"], "roi": r["kind"], "F": r["F"], "r": r_of(r), "r_b2": r_of(r, 2.0),
                              "r_a0": r_of(r, 1.0, "qaoa_a0"), "s2_ratio": s2_ratio(r)}
                             for r in (json.loads(l) for l in open(cp))]
    if pred is not None:
        w = [e for e in pairs if e["predicted_win"]]; o = [e for e in pairs if not e["predicted_win"]]
        a, b = sum(e["r"] > 0 for e in w), sum(e["r"] > 0 for e in o)
        S["routing_test"] = {"predictions_written_at": P["written_at"], "rows_existing_when_written": P["study_rows_when_written"],
                             "predicted_win": {"instances": len(w), "wins": a, "mean_r": mean_ci([e["r"] for e in w], rng),
                                               "dr": mean_ci([e["r"] - e["r_parent"] for e in w], rng)},
                             "rest": {"instances": len(o), "wins": b, "mean_r": mean_ci([e["r"] for e in o], rng),
                                      "dr": mean_ci([e["r"] - e["r_parent"] for e in o], rng)},
                             "fisher_one_sided_p": float(fisher_exact([[a, len(w) - a], [b, len(o) - b]], alternative="greater")[1])
                             if w and o else None}
    json.dump(S, open(os.path.join(RES, f"e3c_{stage}_summary.json"), "w"), indent=1)
    with open(os.path.join(RES, f"e3c_{stage}_pairs.jsonl"), "w") as f:
        for e in pairs:
            f.write(json.dumps(e) + "\n")
    for mod, m in S["modalities"].items():
        print(mod, "dr", m["dr"], "| r", m["r"], "| r_parent", m["r_parent"]["mean"],
              "| C:", m.get("endpoint_C_pass"))
    if "routing_test" in S:
        print("routing", {k: v for k, v in S["routing_test"].items() if k in ("fisher_one_sided_p",)},
              S["routing_test"]["predicted_win"]["wins"], "/", S["routing_test"]["predicted_win"]["instances"], "vs",
              S["routing_test"]["rest"]["wins"], "/", S["routing_test"]["rest"]["instances"])


if __name__ == "__main__":
    main()
