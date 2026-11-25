"""G1b analysis: implements docs/G1b_hardening_protocol_2026-11-21.md exactly (frozen before results existed).

Per instance and beta_tilde:
  pt2_best = max_r pt2_r (oracle ladder), pt3_best = max_r pt3_r
  oracle_a = max(local, label_swap, uniform, sw, wolff, 2^-n)       (2^-n = exact heat-bath gap per unit cost)
  oracle_b = max(oracle_a, pt2_best)       oracle_c = max(oracle_b, pt3_best)
Decision at beta_tilde = 1 for P3_roi_FC and P3_roi_SC, Delta_k CI > 0 in BOTH
  (a') n 4..10 vs {local, label_swap, uniform, heat_bath(k=1), sw, wolff, oracle_a}
  (b') n 4..8  vs (a') + {pt2_best, oracle_b}
Exponents and bootstrap exactly as in g1_analyze.py (geometric-mean log2 gap vs n, floor 1e-14, 1000 resamples).
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
FLOOR = 1e-14
B_BOOT = 1000
rng = np.random.default_rng(7)
SET_A = ["local", "label_swap", "uniform", "heat_bath", "sw", "wolff", "oracle_a"]
SET_B = SET_A + ["pt2_best", "oracle_b"]
SET_C = SET_B + ["pt3_best", "oracle_c"]


def pooled(fam):
    if fam == "P3_canonical":
        return "P3_canonical"
    kind, mod = fam.split("_")[0], fam.split("_")[-1][:2]
    return f"{ {'P1': 'P1_border', 'P2': 'P2_split', 'P3': 'P3_roi'}[kind] }_{mod}"


def key(d):
    return (d["family"], d["ds"], d["n"], d["kind"], round(float(d["sigma"]), 10), round(float(d["F"]), 10))


def merge():
    g1 = [json.loads(l) for l in open(os.path.join(RES, "g1_rows.jsonl"))]
    out = []
    for b in map(json.loads, open(os.path.join(RES, "g1b_rows.jsonl"))):
        r = dict(g1[b["g1_line"]]); r.update(b)
        assert key(r) == tuple(b["key"])
        n = r["n"]
        for bt in (0.5, 1.0, 2.0):
            p = f"b{bt}_"
            r[p + "oracle_a"] = max(r[p + "local"], r[p + "label_swap"], r[p + "uniform"], r[p + "sw"], r[p + "wolff"], 2.0 ** -n)
            pt2 = [v for k, v in r.items() if k.startswith(p + "pt2_r") and v is not None]
            pt3 = [v for k, v in r.items() if k.startswith(p + "pt3_r") and v is not None]
            r[p + "pt2_best"] = max(pt2) if pt2 else None
            r[p + "pt3_best"] = max(pt3) if pt3 else None
            r[p + "oracle_b"] = max(r[p + "oracle_a"], r[p + "pt2_best"]) if pt2 else None
            r[p + "oracle_c"] = max(r[p + "oracle_b"], r[p + "pt3_best"]) if pt2 and pt3 else None
        out.append(r)
    return out


def lg(x):
    return np.log2(max(x, FLOOR))


def kfit(by_n, ns):
    return -np.polyfit(ns, [np.mean(by_n[n]) for n in ns], 1)[0]


def analyse(rows, bt, ns, chains, qkey="quantum"):
    by_n = defaultdict(list)
    for r in rows:
        if r["n"] in ns:
            by_n[r["n"]].append(r)
    if len(by_n) < len(ns) or min(len(v) for v in by_n.values()) < 3:
        return None
    for c in chains:
        if c != "heat_bath" and any(r.get(f"b{bt}_{c}") is None for v in by_n.values() for r in v):
            return {"error": f"missing {c}"}

    def exps(s):
        ks = {c: 1.0 if c == "heat_bath" else kfit({n: [lg(r[f"b{bt}_{c}"]) for r in s[n]] for n in ns}, ns) for c in chains}
        ks["quantum"] = kfit({n: [lg(r[f"b{bt}_{qkey}"]) for r in s[n]] for n in ns}, ns)
        return ks

    pt = exps(by_n)
    best = min((v, k) for k, v in pt.items() if k != "quantum")
    boots = []
    for _ in range(B_BOOT):
        s = {n: [v[i] for i in rng.integers(0, len(v), len(v))] for n, v in by_n.items()}
        e = exps(s)
        boots.append(min(v for k, v in e.items() if k != "quantum") - e["quantum"])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"k": {k: round(float(v), 3) for k, v in pt.items()}, "best_classical": best[1],
            "k_best_classical": round(float(best[0]), 3), "delta_k": round(float(best[0] - pt["quantum"]), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)], "instances": int(sum(len(v) for v in by_n.values()))}


def oracle_key(r, bt):
    return f"b{bt}_oracle_b" if r.get(f"b{bt}_oracle_b") is not None else f"b{bt}_oracle_a"


def wins(rows, bt):
    return [(r, r[f"b{bt}_quantum"] > r[oracle_key(r, bt)], r[f"b{bt}_quantum"] / r[oracle_key(r, bt)]) for r in rows]


def main():
    rows = merge()
    fams = defaultdict(list)
    for r in rows:
        fams[pooled(r["family"])].append(r)
    S = {"protocol": "docs/G1b_hardening_protocol_2026-11-21.md", "rule": __doc__,
         "instances": len(rows),
         "determinism": "every instance matched one-to-one to its G1 row by fingerprint + local gap (rel. 1e-9)",
         "min_offdiag_density": min(r["density"] for r in rows),
         "pt_errors": sum(len(r.get("errors", [])) for r in rows),
         "pt2_r_chosen_beta1": {}, "families": {}}
    for r in rows:
        if r["n"] <= 8:
            vals = {k: v for k, v in r.items() if k.startswith("b1.0_pt2_r") and v is not None}
            if vals:
                kbest = max(vals, key=vals.get)
                S["pt2_r_chosen_beta1"][kbest[10:]] = S["pt2_r_chosen_beta1"].get(kbest[10:], 0) + 1
    for fam in sorted(fams):
        rr = fams[fam]
        e = {"instances": len(rr)}
        for bt in (0.5, 1.0, 2.0):
            d = {"A_prime_4to10": analyse(rr, bt, list(range(4, 11)), SET_A)}
            if bt in (1.0, 2.0):
                d["B_prime_4to8"] = analyse(rr, bt, list(range(4, 9)), SET_B)
            if bt == 1.0:
                d["C_prime_4to6"] = analyse(rr, bt, list(range(4, 7)), SET_C)
                d["G1_original_A_check"] = analyse(rr, bt, list(range(4, 11)), ["local", "label_swap", "uniform", "heat_bath"])
            w = wins(rr, bt)
            d["win_share_vs_oracle"] = round(float(np.mean([x[1] for x in w])), 3)
            d["median_ratio_vs_oracle"] = round(float(np.median([x[2] for x in w])), 3)
            e[f"beta_tilde={bt}"] = d
        S["families"][fam] = e
    passed = []
    for fam in ("P3_roi_FC", "P3_roi_SC"):
        d = S["families"][fam]["beta_tilde=1.0"]
        a, b = d["A_prime_4to10"], d["B_prime_4to8"]
        if a and b and "ci95" in a and "ci95" in b and a["ci95"][0] > 0 and b["ci95"][0] > 0:
            passed.append(fam)
    S["G1b_families_passing"] = passed
    S["G1b_outcome"] = {2: "main route confirmed", 1: f"main route restricted to {passed[0] if passed else ''}",
                        0: "fallback characterisation paper"}[len(passed)]
    # prospective check of the post-hoc routing rule (field-free, beta_tilde = 1, vs classical oracle)
    w = wins(rows, 1.0)
    F = np.array([r["F"] for r, _, _ in w]); y = np.array([b for _, b, _ in w])
    S["win_share_by_F_bin_vs_oracle"] = {f"[{lo},{hi})": {"n": int(np.sum((F >= lo) & (F < hi))),
                                                          "win_share": round(float(np.mean(y[(F >= lo) & (F < hi)])), 3)}
                                         for lo, hi in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, 1.01))}
    rule = F >= 0.05
    S["routing_rule_F>=0.05"] = {"routed": int(rule.sum()), "win_share_routed": round(float(y[rule].mean()), 3),
                                 "win_share_not_routed": round(float(y[~rule].mean()), 3)}
    S["canonical_rois_beta1"] = [{"ds": r["ds"], "roi": r["kind"], "n": r["n"], "F": round(r["F"], 3), "quantum_wins": bool(b),
                                  "ratio_vs_oracle": round(float(q), 3)} for r, b, q in wins(fams["P3_canonical"], 1.0)]
    json.dump(S, open(os.path.join(RES, "g1b_summary.json"), "w"), indent=1)

    def fmt(x):
        if not x or "ci95" not in x:
            return str(x)
        return f"dk {x['delta_k']:+.3f} CI {x['ci95']} kq {x['k']['quantum']} best {x['best_classical']} {x['k_best_classical']}"
    for fam, e in S["families"].items():
        print(f"\n== {fam} ({e['instances']})")
        for bt in (0.5, 1.0, 2.0):
            d = e[f"beta_tilde={bt}"]
            for k, v in d.items():
                print(f"  b~{bt} {k}: {fmt(v) if isinstance(v, dict) or v is None else v}")
    for k in ("instances", "determinism", "min_offdiag_density", "pt_errors", "pt2_r_chosen_beta1",
              "G1b_families_passing", "G1b_outcome", "win_share_by_F_bin_vs_oracle", "routing_rule_F>=0.05"):
        print(k, ":", S[k])
    print("canonical wins vs oracle:", sum(x["quantum_wins"] for x in S["canonical_rois_beta1"]), "/", len(S["canonical_rois_beta1"]))


if __name__ == "__main__":
    main()
