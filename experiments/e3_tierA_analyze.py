"""E3 Tier A endpoints (docs/E3_scaling_protocol_2026-11-21.md, amendments 1-12), partition-space gaps.

Per instance, beta_tilde = 1:  r = log2(gap_quantum / gap_oracle),
  oracle = max(local, label_swap, uniform, sw, wolff, 2^-(n-1))   (all partition-space; PT excluded at every n).
Endpoint A, per modality (FC = FC200+FC400, SC = SC200+SC400 random ROI sets; canonical reported separately):
  (i)  slope of mean r vs n over n = 7..12, 95% bootstrap CI (1000 resamples of instances within n) > 0
  (ii) mean r at n = 12, 95% bootstrap CI > 0.          PASS iff (i) and (ii).
Secondary: same on full-space gaps; G1b min-exponent rule (n 7..12); asymptotic-variance ratio
log2(s2_best_classical / s2_quantum) (heat-bath i.i.d. at cost 2^(n-1)); beta_tilde 0.5, 2; QAOA-MC grid average;
PT-2 check at n = 7-8. Prospective routing test: one-sided Fisher exact, win share (r > 0) for F >= 0.05 vs F < 0.05.

Descriptive additions (22 Nov 2026, written while Tier A was still running and before this script was first run on
any Tier A data; none of them changes an endpoint, and they draw from a separate bootstrap stream so every frozen
number is reproduced exactly):
  - QAOA-MC grid angle 0 (pi/8, pi/8, 0.5) out of sample: the endpoint-A statistics with q = qaoa_a0, plus the best
    angle by mean r per n on the fresh instances (announced in G1c report section 5, before any E3 data);
  - win share (r > 0) by n; which classical chain is the oracle, by n; mean log2 partition-space gap per chain by n;
  - win share by F bin (G1c bins) and by ROI kind.
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
CL = ["local", "label_swap", "uniform", "sw", "wolff"]
NS = list(range(7, 13))
rng = np.random.default_rng(17)


def modality(fam):
    return None if fam == "P3_canonical" else fam.split("_")[-1][:2]


def oracle(r, bt, kind="lumped"):
    return max([r[f"b{bt}_{c}_{kind}"] for c in CL] + [2.0 ** -(r["n"] - (1 if kind == "lumped" else 0))])


def rvals(rows, bt, kind="lumped", q="quantum"):
    return [(r["n"], np.log2(max(r[f"b{bt}_{q}_{kind}"], 1e-300) / oracle(r, bt, kind))) for r in rows]


def s2vals(rows, bt):
    out = []
    for r in rows:
        s2c = [r[f"b{bt}_{c}_s2mean"] for c in CL] + [r[f"b{bt}_varmean"] * 2.0 ** (r["n"] - 1)]
        out.append((r["n"], np.log2(min(s2c) / r[f"b{bt}_quantum_s2mean"])))
    return out


def endpoint(pairs, ns=NS, rg=None):
    rg = rng if rg is None else rg
    by = defaultdict(list)
    for n, v in pairs:
        if n in ns:
            by[n].append(v)
    if any(len(by[n]) < 3 for n in ns):
        return None
    def stat(b):
        m = [np.mean(b[n]) for n in ns]
        return np.polyfit(ns, m, 1)[0], m[-1]
    s0, e0 = stat(by)
    boots = []
    for _ in range(1000):
        b = {n: [v[i] for i in rg.integers(0, len(v), len(v))] for n, v in by.items()}
        boots.append(stat(b))
    boots = np.array(boots)
    ci = lambda x: [round(float(v), 3) for v in np.percentile(x, [2.5, 97.5])]
    return {"slope": round(float(s0), 3), "slope_ci95": ci(boots[:, 0]), f"mean_r_at_n{ns[-1]}": round(float(e0), 3),
            f"mean_r_at_n{ns[-1]}_ci95": ci(boots[:, 1]), "mean_r_by_n": {n: round(float(np.mean(by[n])), 3) for n in ns},
            "instances_by_n": {n: len(by[n]) for n in ns},
            "passes": bool(np.percentile(boots[:, 0], 2.5) > 0 and np.percentile(boots[:, 1], 2.5) > 0)}


def main():
    rows = [json.loads(l) for l in open(os.path.join(RES, "e3_tierA_rows.jsonl"))]
    for r in rows:   # G1b-style keys for the min-exponent secondary
        for bt in (0.5, 1.0, 2.0):
            for c in CL + ["quantum"]:
                r[f"b{bt}_{c}"] = r[f"b{bt}_{c}_lumped"]
            r[f"b{bt}_oracle_a"] = oracle(r, bt)
    mods = defaultdict(list)
    for r in rows:
        mods[modality(r["family"])].append(r)
    S = {"protocol": "docs/E3_scaling_protocol_2026-11-21.md", "rule": __doc__, "instances": len(rows),
         "errors": sum(len(r.get("errors", [])) for r in rows), "modalities": {}}
    for mod in ("FC", "SC"):
        rr = mods[mod]
        e = {"instances": len(rr), "PRIMARY_endpoint_A_beta1": endpoint(rvals(rr, 1.0))}
        e["secondary_full_space_beta1"] = endpoint(rvals(rr, 1.0, "full"))
        e["secondary_asymptotic_variance_beta1"] = endpoint(s2vals(rr, 1.0))
        e["secondary_qaoa_grid_beta1"] = endpoint(rvals(rr, 1.0, q="qaoa"))
        for bt in (0.5, 2.0):
            e[f"secondary_endpoint_A_beta{bt}"] = endpoint(rvals(rr, bt))
        e["secondary_G1b_min_exponent_rule_7to12"] = gb.analyse(rr, 1.0, NS, ["local", "label_swap", "uniform", "heat_bath",
                                                                             "sw", "wolff", "oracle_a"])
        ptv = [(r, [v for k, v in r.items() if k.startswith("b1.0_pt2_r") and v is not None]) for r in rr if r["n"] <= 8]
        pt = [(r, max(v)) for r, v in ptv if v]          # guard (22 Nov): skip a row only if every PT ladder failed
        e["PT2_check_n7to8"] = {"rows": len(pt), "rows_all_ladders_failed": len(ptv) - len(pt),
                                "share_pt2_beats_oracle": round(float(np.mean([p > oracle(r, 1.0) for r, p in pt])), 3),
                                "mean_log2_pt2_over_oracle": round(float(np.mean([np.log2(p / oracle(r, 1.0)) for r, p in pt])), 3)}
        S["modalities"][mod] = e
    # ---- descriptive additions (separate bootstrap stream; see docstring) ----
    rx = np.random.default_rng(29)
    for mod in ("FC", "SC"):
        rr, d = mods[mod], {}
        d["qaoa_angle0_endpoint_beta1"] = endpoint(rvals(rr, 1.0, q="qaoa_a0"), rg=rx)
        ang = {n: [float(np.mean([np.log2(max(r[f"b1.0_qaoa_a{i}_lumped"], 1e-300) / oracle(r, 1.0))
                                  for r in rr if r["n"] == n])) for i in range(18)] for n in NS}
        d["qaoa_best_angle_by_n"] = {n: int(np.argmax(v)) for n, v in ang.items()}
        d["qaoa_angle0_rank_by_n"] = {n: int(1 + np.sum(np.array(v) > v[0])) for n, v in ang.items()}
        d["qaoa_best_angle_mean_r_by_n"] = {n: round(max(v), 3) for n, v in ang.items()}
        rv = rvals(rr, 1.0)
        d["win_share_by_n"] = {n: round(float(np.mean([v > 0 for m, v in rv if m == n])), 3) for n in NS}
        d["oracle_chain_counts_by_n"] = {}
        for n in NS:
            cnt = defaultdict(int)
            for r in rr:
                if r["n"] == n:
                    g = {c: r[f"b1.0_{c}_lumped"] for c in CL}
                    g["heat_bath"] = 2.0 ** -(n - 1)
                    cnt[max(g, key=g.get)] += 1
            d["oracle_chain_counts_by_n"][n] = dict(cnt)
        d["mean_log2_gap_by_n"] = {c: {n: round(float(np.mean([np.log2(max(r[f"b1.0_{c}_lumped"], 1e-300))
                                                               for r in rr if r["n"] == n])), 3) for n in NS}
                                   for c in CL + ["quantum", "qaoa", "qaoa_a0"]}
        d["mean_log2_gap_by_n"]["heat_bath"] = {n: float(-(n - 1)) for n in NS}
        kinds = sorted({r["kind"] for r in rr})
        d["mean_r_by_kind_and_n"] = {k: {n: round(float(np.mean([np.log2(r["b1.0_quantum_lumped"] / oracle(r, 1.0))
                                                                for r in rr if r["n"] == n and r["kind"] == k])), 3)
                                         for n in NS} for k in kinds}
        S["modalities"][mod]["descriptive"] = d
    Fa = np.array([r["F"] for r in rows]); ya = np.array([v > 0 for _, v in rvals(rows, 1.0)])
    S["win_share_by_F_bin"] = {f"[{lo},{hi})": {"n": int(np.sum((Fa >= lo) & (Fa < hi))),
                                                "win_share": round(float(np.mean(ya[(Fa >= lo) & (Fa < hi)])), 3)
                                                if np.any((Fa >= lo) & (Fa < hi)) else None}
                               for lo, hi in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, 1.01))}
    S["canonical_beta1"] = [{"ds": r["ds"], "roi": r["kind"], "n": r["n"], "F": round(r["F"], 3),
                             "r": round(float(np.log2(r["b1.0_quantum_lumped"] / oracle(r, 1.0))), 3)} for r in mods[None]]
    from scipy.stats import fisher_exact
    F = np.array([r["F"] for r in rows]); y = np.array([v > 0 for _, v in rvals(rows, 1.0)])
    for thr in (0.05, 0.10):
        a, b = int(np.sum(y & (F >= thr))), int(np.sum(~y & (F >= thr)))
        c, d = int(np.sum(y & (F < thr))), int(np.sum(~y & (F < thr)))
        S[f"routing_test_F>={thr}"] = {"routed_wins": a, "routed_losses": b, "unrouted_wins": c, "unrouted_losses": d,
                                       "win_share_routed": round(a / max(a + b, 1), 3), "win_share_unrouted": round(c / max(c + d, 1), 3),
                                       "fisher_one_sided_p": float(fisher_exact([[a, b], [c, d]], alternative="greater")[1])}
    S["E3_endpoint_A"] = {m: S["modalities"][m]["PRIMARY_endpoint_A_beta1"]["passes"] if S["modalities"][m]["PRIMARY_endpoint_A_beta1"] else None
                          for m in ("FC", "SC")}
    json.dump(S, open(os.path.join(RES, "e3_tierA_summary.json"), "w"), indent=1)
    print(json.dumps(S, indent=1)[:12000])


if __name__ == "__main__":
    main()
