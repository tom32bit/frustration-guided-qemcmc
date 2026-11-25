"""G1 analysis - pre-specified decision rule (written before results were inspected, 21 Nov 2026):

G1 PASSES iff, at the primary temperature beta_tilde = 1, at least 2 of the 6 pooled brain families
{P1_border, P2_split, P3_roi} x {FC, SC} show k_quantum < k_best-classical with the 95 % bootstrap CI of
Delta_k = k_best-classical - k_quantum strictly above 0, BOTH
  (a) over n = 4..10 against {local, label_swap, uniform, exact block heat-bath at cost 2^n (k = 1)}, AND
  (b) over n = 4..8  against the same set + parallel tempering (2 replicas),
using the UNTUNED quantum proposal (Layden grid average). QAOA-MC and class-tuned quench are secondary.
Exponents: k = -slope of mean(log2 gap) vs n (geometric mean; gaps floored at 1e-14).
"""
import json
import os
from collections import defaultdict

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FLOOR = 1e-14
B_BOOT = 1000
rng = np.random.default_rng(7)


def pooled(fam):
    if fam == "P3_canonical":
        return None
    kind, mod = fam.split("_")[0], fam.split("_")[-1][:2]
    return f"{ {'P1': 'P1_border', 'P2': 'P2_split', 'P3': 'P3_roi'}[kind] }_{mod}"


def lg(x):
    return np.log2(max(x, FLOOR)) if x is not None else np.nan


def kfit(by_n, ns):
    m = [np.nanmean(by_n[n]) for n in ns]
    return -np.polyfit(ns, m, 1)[0]


def analyse(rows, bt, ns, classical, quantum_key, tuned=False):
    """Returns point estimate and bootstrap CI of Delta_k, plus all exponents."""
    by_n = defaultdict(list)
    for r in rows:
        if r["n"] in ns:
            by_n[r["n"]].append(r)
    if "pt" in classical and any(r.get(f"b{bt}_pt") is None for v in by_n.values() for r in v):
        return None  # PT not computed for this family (P1)
    if len(by_n) < len(ns) or min(len(v) for v in by_n.values()) < 3:
        return None
    gidx = None
    if tuned:  # choose gamma on even-indexed instances, evaluate on odd-indexed (held out)
        train = [r for n in ns for r in by_n[n][0::2]]
        gidx = int(np.argmax([np.mean([lg(r[f"b{bt}_quantum_g{i}"]) for r in train]) for i in range(6)]))
        by_n = {n: v[1::2] for n, v in by_n.items()}
        if min(len(v) for v in by_n.values()) < 2:
            return None
    qkey = f"b{bt}_quantum_g{gidx}" if tuned else f"b{bt}_{quantum_key}"

    def exps(sample):
        ks = {}
        for name in classical:
            if name == "heat_bath":
                ks[name] = 1.0
            else:
                ks[name] = kfit({n: [lg(r[f"b{bt}_{name}"]) for r in sample[n]] for n in ns}, ns)
        ks["quantum"] = kfit({n: [lg(r[qkey]) for r in sample[n]] for n in ns}, ns)
        return ks

    point = exps(by_n)
    best = min(v for k, v in point.items() if k != "quantum")
    boots = []
    for _ in range(B_BOOT):
        s = {n: [v[i] for i in rng.integers(0, len(v), len(v))] for n, v in by_n.items()}
        e = exps(s)
        boots.append(min(v for k, v in e.items() if k != "quantum") - e["quantum"])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"k": {k: round(float(v), 3) for k, v in point.items()}, "k_best_classical": round(float(best), 3),
            "best_classical": min((v, k) for k, v in point.items() if k != "quantum")[1],
            "delta_k": round(float(best - point["quantum"]), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "instances": int(sum(len(v) for v in by_n.values())), "tuned_gamma_index": gidx}


def instance_wins(rows, bt):
    out = []
    for r in rows:
        cl = [r[f"b{bt}_local"], r[f"b{bt}_label_swap"], r[f"b{bt}_uniform"], 2.0 ** -r["n"]]
        if r.get(f"b{bt}_pt") is not None:
            cl.append(r[f"b{bt}_pt"])
        out.append((r, r[f"b{bt}_quantum"] > max(cl), r[f"b{bt}_quantum"] / max(cl)))
    return out


def main():
    data = json.load(open(os.path.join(RES, "g1_rows.json")))
    rows = data["rows"]
    fams = defaultdict(list)
    for r in rows:
        p = pooled(r["family"])
        if p:
            fams[p].append(r)
    CL_A = ["local", "label_swap", "uniform", "heat_bath"]
    CL_B = CL_A + ["pt"]
    summary = {"rule": __doc__, "families": {}}
    for fam in sorted(fams):
        rr = fams[fam]
        entry = {"instances": len(rr), "median_F": round(float(np.median([r["F"] for r in rr])), 3),
                 "median_field_ratio": round(float(np.median([r["field_ratio"] for r in rr])), 2)}
        for bt in data["betas_tilde"]:
            e = {"A_untuned_4to10": analyse(rr, bt, list(range(4, 11)), CL_A, "quantum"),
                 "A_qaoa_mc_4to10": analyse(rr, bt, list(range(4, 11)), CL_A, "qaoa_mc"),
                 "A_tuned_heldout_4to10": analyse(rr, bt, list(range(4, 11)), CL_A, None, tuned=True)}
            if bt in (1.0, 2.0):
                e["B_untuned_with_PT_4to8"] = analyse(rr, bt, list(range(4, 9)), CL_B, "quantum")
            w = instance_wins(rr, bt)
            e["instance_win_share"] = round(float(np.mean([x[1] for x in w])), 3)
            e["instance_median_ratio"] = round(float(np.median([x[2] for x in w])), 3)
            entry[f"beta_tilde={bt}"] = e
        summary["families"][fam] = entry

    # decision
    passed = []
    for fam, entry in summary["families"].items():
        e = entry["beta_tilde=1.0"]
        a, b = e["A_untuned_4to10"], e.get("B_untuned_with_PT_4to8")
        if a and b and a["ci95"][0] > 0 and b["ci95"][0] > 0:
            passed.append(fam)
    summary["G1_families_passing"] = passed
    summary["G1_PASS"] = len(passed) >= 2

    # canonical ROI networks (anatomical) - instance level
    can = [r for r in rows if r["family"] == "P3_canonical"]
    summary["canonical_rois_beta1"] = [{"ds": r["ds"], "roi": r["kind"], "n": r["n"], "F": round(r["F"], 3),
                                        "quantum_wins": bool(w), "ratio_q_over_best_classical": round(float(q), 3)}
                                       for r, w, q in instance_wins(can, 1.0)]

    # frustration criterion: leave-one-family-out logistic regression at beta_tilde = 1
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    allr = [r for r in rows if pooled(r["family"])]
    w = instance_wins(allr, 1.0)
    X = np.array([[np.log10(r["F"] + 1e-3), r["n"], np.log10(r["field_ratio"] + 1e-3)] for r, _, _ in w])
    y = np.array([int(b) for _, b, _ in w])
    famv = np.array([pooled(r["family"]) for r, _, _ in w])
    aucs = {}
    for f in sorted(set(famv)):
        tr, te = famv != f, famv == f
        if len(set(y[te])) < 2:
            aucs[f] = None
            continue
        m = LogisticRegression().fit(X[tr], y[tr])
        aucs[f] = round(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])), 3)
    m = LogisticRegression().fit(X[:, :2], y)
    fstar = 10 ** (-(m.intercept_[0] + m.coef_[0][1] * 10) / m.coef_[0][0]) - 1e-3
    summary["frustration_criterion"] = {"leave_one_family_out_AUC": aucs,
                                        "pooled_AUC_F_only": round(float(roc_auc_score(y, X[:, 0])), 3),
                                        "F_star_at_n10": round(float(fstar), 4),
                                        "win_share_by_F_bin": {
                                            f"[{lo},{hi})": round(float(np.mean(y[(10 ** X[:, 0] - 1e-3 >= lo) & (10 ** X[:, 0] - 1e-3 < hi)])), 3)
                                            for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1))
                                            if np.any((10 ** X[:, 0] - 1e-3 >= lo) & (10 ** X[:, 0] - 1e-3 < hi))}}
    with open(os.path.join(RES, "g1_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    # compact print
    for fam, entry in summary["families"].items():
        print(f"\n== {fam}: {entry['instances']} inst, median F {entry['median_F']}, field ratio {entry['median_field_ratio']}")
        for bt in data["betas_tilde"]:
            e = entry[f"beta_tilde={bt}"]
            def fmt(x):
                return "n/a" if not x else f"dk {x['delta_k']:+.3f} CI {x['ci95']} (kq {x['k']['quantum']}, best {x['best_classical']} {x['k_best_classical']})"
            print(f"  b~={bt}: untuned {fmt(e['A_untuned_4to10'])}")
            print(f"          qaoa_mc {fmt(e['A_qaoa_mc_4to10'])} | tuned(held-out) {fmt(e['A_tuned_heldout_4to10'])}")
            if "B_untuned_with_PT_4to8" in e:
                print(f"          +PT 4..8 {fmt(e['B_untuned_with_PT_4to8'])}")
            print(f"          instance wins {e['instance_win_share']}  median ratio {e['instance_median_ratio']}")
    print("\nG1 families passing:", passed, "=> G1_PASS =", summary["G1_PASS"])
    print("frustration criterion:", json.dumps(summary["frustration_criterion"]))
    print("canonical ROIs:", sum(x["quantum_wins"] for x in summary["canonical_rois_beta1"]), "/", len(summary["canonical_rois_beta1"]), "quantum wins")


if __name__ == "__main__":
    main()
