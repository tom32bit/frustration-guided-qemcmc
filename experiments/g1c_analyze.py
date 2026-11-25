"""G1c analysis (docs/G1c_partition_space_protocol_2026-11-21.md): the frozen G1b rule on PARTITION-SPACE gaps.

Per instance / beta_tilde, every chain gap is its lumped (partition-space) gap; heat-bath per-cost gap 2^-(n-1);
oracle_a/b/c as in G1b. Decision at beta_tilde = 1 for P3_roi_FC / P3_roi_SC: (a') n 4..10 and (b') n 4..8 (+PT-2)
CIs of Delta_k strictly above 0. Exponent fit, bootstrap and seeds exactly as g1b_analyze.
Descriptive: r = log2(gap_q / gap_oracle) by n; full-vs-lumped ratios per chain; asymptotic-variance ratio
(best classical estimator variance, incl. i.i.d. heat-bath at cost 2^(n-1), over quantum); QAOA-MC angle ranking.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g1b_analyze as gb  # noqa: E402

RES = gb.RES
BTS = (0.5, 1.0, 2.0)
CL = ["local", "label_swap", "uniform", "sw", "wolff"]


def merge():
    g1 = [json.loads(l) for l in open(os.path.join(RES, "g1_rows.jsonl"))]
    g1b = {json.loads(l)["g1_line"]: json.loads(l) for l in open(os.path.join(RES, "g1b_rows.jsonl"))}
    out, det = [], defaultdict(list)
    for c in map(json.loads, open(os.path.join(RES, "g1c_rows.jsonl"))):
        base = g1[c["g1_line"]]
        n = c["n"]
        r = {k: base[k] for k in ("family", "ds", "n", "kind", "sigma", "F")}
        r["g1_line"] = c["g1_line"]
        for bt in BTS:
            p = f"b{bt}_"
            # determinism: full-space gaps recomputed here vs the G1 / G1b values
            for ch, ref in (("quantum", base.get(p + "quantum")), ("local", base.get(p + "local")),
                            ("sw", g1b.get(c["g1_line"], {}).get(p + "sw"))):
                v = c.get(p + ch + "_full")
                if ref is not None and v is not None and ref > 1e-12:
                    det[ch].append(abs(v - ref) / ref)
            for ch in CL + ["quantum", "qaoa"]:
                r[p + ch] = c[p + ch + "_lumped"]
                r[p + ch + "_full"] = c[p + ch + "_full"]
                r[p + ch + "_s2"] = c.get(p + ch + "_s2mean")
            r[p + "varmean"] = c.get(p + "varmean")
            pt2 = [v for k, v in c.items() if k.startswith(p + "pt2_r") and v is not None]
            pt3 = [v for k, v in c.items() if k.startswith(p + "pt3_r") and v is not None]
            r[p + "pt2_best"] = max(pt2) if pt2 else None
            r[p + "pt3_best"] = max(pt3) if pt3 else None
            r[p + "oracle_a"] = max([r[p + ch] for ch in CL] + [2.0 ** -(n - 1)])
            r[p + "oracle_b"] = max(r[p + "oracle_a"], r[p + "pt2_best"]) if pt2 else None
            r[p + "oracle_c"] = max(r[p + "oracle_b"], r[p + "pt3_best"]) if pt2 and pt3 else None
            s2c = [r[p + ch + "_s2"] for ch in CL if r[p + ch + "_s2"] is not None]
            if r[p + "varmean"] is not None:
                s2c.append(r[p + "varmean"] * 2.0 ** (n - 1))          # i.i.d. heat-bath samples at cost 2^(n-1)
            r[p + "s2_best_classical"] = min(s2c) if s2c else None
        r["qaoa_angles_b1"] = [c.get(f"b1.0_qaoa_a{i}_lumped") for i in range(18)]
        r["errors"] = c.get("errors", [])
        out.append(r)
    return out, {k: float(np.max(v)) for k, v in det.items()}


def main():
    rows, det = merge()
    fams = defaultdict(list)
    for r in rows:
        fams[gb.pooled(r["family"])].append(r)
    S = {"protocol": "docs/G1c_partition_space_protocol_2026-11-21.md", "rule": __doc__, "instances": len(rows),
         "determinism_max_rel_err_full_gaps": det, "pt_errors": sum(len(r["errors"]) for r in rows), "families": {}}
    for fam in sorted(fams):
        rr = fams[fam]
        e = {"instances": len(rr)}
        for bt in BTS:
            d = {"A_prime_4to10": gb.analyse(rr, bt, list(range(4, 11)), gb.SET_A)}
            if bt in (1.0, 2.0):
                d["B_prime_4to8"] = gb.analyse(rr, bt, list(range(4, 9)), gb.SET_B)
            if bt == 1.0:
                d["C_prime_4to6"] = gb.analyse(rr, bt, list(range(4, 7)), gb.SET_C)
            ok = lambda r: f"b{bt}_oracle_b" if r.get(f"b{bt}_oracle_b") is not None else f"b{bt}_oracle_a"
            d["win_share_vs_oracle"] = round(float(np.mean([r[f"b{bt}_quantum"] > r[ok(r)] for r in rr])), 3)
            d["r_log2_q_over_oracle_by_n"] = {n: round(float(np.mean([np.log2(r[f"b{bt}_quantum"] / r[ok(r)])
                                                                      for r in rr if r["n"] == n])), 3)
                                              for n in sorted({r["n"] for r in rr})}
            d["log2_s2_ratio_best_classical_over_quantum_by_n"] = {
                n: round(float(np.mean([np.log2(r[f"b{bt}_s2_best_classical"] / r[f"b{bt}_quantum_s2"])
                                        for r in rr if r["n"] == n])), 3) for n in sorted({r["n"] for r in rr})}
            d["median_lumped_over_full_by_chain"] = {
                ch: round(float(np.median([r[f"b{bt}_{ch}"] / max(r[f"b{bt}_{ch}_full"], 1e-300) for r in rr])), 3)
                for ch in CL + ["quantum", "qaoa"]}
            e[f"beta_tilde={bt}"] = d
        S["families"][fam] = e
    passed = [f for f in ("P3_roi_FC", "P3_roi_SC")
              if all(x and "ci95" in x and x["ci95"][0] > 0 for x in
                     (S["families"][f]["beta_tilde=1.0"]["A_prime_4to10"], S["families"][f]["beta_tilde=1.0"]["B_prime_4to8"]))]
    S["G1c_families_passing"] = passed
    # routing on partition space (same instances as G1/G1b: not out-of-sample)
    allr = [r for r in rows]
    ok1 = lambda r: "b1.0_oracle_b" if r.get("b1.0_oracle_b") is not None else "b1.0_oracle_a"
    F = np.array([r["F"] for r in allr]); y = np.array([r["b1.0_quantum"] > r[ok1(r)] for r in allr])
    S["win_share_by_F_bin_vs_oracle"] = {f"[{lo},{hi})": {"n": int(np.sum((F >= lo) & (F < hi))),
                                                          "win_share": round(float(np.mean(y[(F >= lo) & (F < hi)])), 3)}
                                         for lo, hi in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, 1.01))}
    S["canonical_rois_beta1"] = [{"ds": r["ds"], "roi": r["kind"], "n": r["n"], "F": round(r["F"], 3),
                                  "quantum_wins": bool(r["b1.0_quantum"] > r[ok1(r)]),
                                  "ratio_vs_oracle": round(float(r["b1.0_quantum"] / r[ok1(r)]), 3)}
                                 for r in fams["P3_canonical"]]
    # QAOA-MC angle ranking (class level) for the hardware pilot: P3_roi_{FC,SC}, n = 6, beta_tilde = 1
    for mod in ("FC", "SC"):
        g6 = [r for r in fams[f"P3_roi_{mod}"] if r["n"] == 6]
        m = [round(float(np.mean([np.log2(max(r["qaoa_angles_b1"][i], 1e-14)) for r in g6])), 3) for i in range(18)]
        S[f"qaoa_angle_mean_log2_gap_P3_roi_{mod}_n6"] = m
        S[f"qaoa_best_angle_index_P3_roi_{mod}_n6"] = int(np.argmax(m))
    json.dump(S, open(os.path.join(RES, "g1c_summary.json"), "w"), indent=1)

    def fmt(x):
        if not x or "ci95" not in x:
            return str(x)
        return f"dk {x['delta_k']:+.3f} CI {x['ci95']} kq {x['k']['quantum']} best {x['best_classical']} {x['k_best_classical']}"
    for fam, e in S["families"].items():
        print(f"\n== {fam} ({e['instances']})")
        for bt in BTS:
            d = e[f"beta_tilde={bt}"]
            for k, v in d.items():
                print(f"  b~{bt} {k}: {fmt(v) if isinstance(v, dict) and 'ci95' in (v or {}) else v}")
    for k in ("instances", "determinism_max_rel_err_full_gaps", "pt_errors", "G1c_families_passing",
              "win_share_by_F_bin_vs_oracle", "qaoa_best_angle_index_P3_roi_FC_n6", "qaoa_best_angle_index_P3_roi_SC_n6"):
        print(k, ":", S[k])
    print("canonical wins:", sum(x["quantum_wins"] for x in S["canonical_rois_beta1"]), "/", len(S["canonical_rois_beta1"]))


if __name__ == "__main__":
    main()
