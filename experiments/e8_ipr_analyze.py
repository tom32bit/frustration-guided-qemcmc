"""E8 analysis, exactly as fixed in docs/E8_ipr_protocol_2026-11-24.md (P1-P5 and the secondary descriptives).
Reads results/e8_ipr_rows.jsonl, writes results/e8_ipr_summary.json."""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
NPERM, NBOOT, SEED = 10_000, 2_000, 2028
TOL = 1e-9


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def resid(y, x):
    X = np.column_stack([np.ones_like(x), x])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def cells_of(rows, mod):
    out = {}
    for r in rows:
        if r["modality"] == mod:
            out.setdefault(r["n"], []).append(r)
    return [out[n] for n in sorted(out)]


def strat_spearman(cells, fx, fy, rng):
    X = [rankdata([fx(r) for r in c]) for c in cells]
    Y = [rankdata([fy(r) for r in c]) for c in cells]
    obs = np.mean([pearson(x, y) for x, y in zip(X, Y)])
    perm = np.array([np.mean([pearson(rng.permutation(x), y) for x, y in zip(X, Y)]) for _ in range(NPERM)])
    p = (1 + np.sum(np.abs(perm) >= abs(obs))) / (1 + NPERM)
    boot = []
    for _ in range(NBOOT):
        vals = []
        for c in cells:
            idx = rng.integers(0, len(c), len(c))
            vals.append(pearson(rankdata([fx(c[i]) for i in idx]), rankdata([fy(c[i]) for i in idx])))
        boot.append(np.mean(vals))
    return {"rho": float(obs), "p": float(p), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def strat_partial(cells, fy, fx, fz, rng):
    """Mean within-cell partial Spearman rho(y, x | z): ranks of y and x regressed on ranks of z, residuals
    correlated; p by permuting the x residuals within cells."""
    EY, EX = [], []
    for c in cells:
        z = rankdata([fz(r) for r in c])
        EY.append(resid(rankdata([fy(r) for r in c]), z)); EX.append(resid(rankdata([fx(r) for r in c]), z))
    obs = np.mean([pearson(y, x) for y, x in zip(EY, EX)])
    perm = np.array([np.mean([pearson(y, rng.permutation(x)) for y, x in zip(EY, EX)]) for _ in range(NPERM)])
    p = (1 + np.sum(np.abs(perm) >= abs(obs))) / (1 + NPERM)
    boot = []
    for _ in range(NBOOT):
        vals = []
        for c in cells:
            idx = rng.integers(0, len(c), len(c))
            cc = [c[i] for i in idx]
            z = rankdata([fz(r) for r in cc])
            vals.append(pearson(resid(rankdata([fy(r) for r in cc]), z), resid(rankdata([fx(r) for r in cc]), z)))
        boot.append(np.mean(vals))
    return {"rho": float(obs), "p": float(p), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def slope_ci(cells, f, rng):
    ns = np.array([c[0]["n"] for c in cells], float)
    med = np.array([np.median([f(r) for r in c]) for c in cells])
    s = np.polyfit(ns, med, 1)[0]
    boot = []
    for _ in range(NBOOT):
        m = [np.median([f(c[i]) for i in rng.integers(0, len(c), len(c))]) for c in cells]
        boot.append(np.polyfit(ns, m, 1)[0])
    return {"slope": float(s), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "cell_medians": dict(zip([int(n) for n in ns], [float(x) for x in med]))}


def main():
    rows = [json.loads(l) for l in open(os.path.join(RES, "e8_ipr_rows.jsonl"))]
    assert len(rows) == 288 and len({tuple(r["key"]) for r in rows}) == 288, len(rows)
    rng = np.random.default_rng(SEED)
    ell, F, r_, bD = (lambda r: r["ell_pi"]), (lambda r: r["F"]), (lambda r: r["r"]), (lambda r: r["beta_D"])
    out = {"n_networks": len(rows), "settings": {"NPERM": NPERM, "NBOOT": NBOOT, "SEED": SEED}}
    # P3 first: the bounds are a known-answer check; a violation stops the analysis
    viol = {}
    for bt in ("b0.5", "b1.0", "b2.0"):
        v = [r["key"] for r in rows if not (r[bt]["delta_inf"] <= r[bt]["Lambda_star"] * (1 + TOL)
                                            and r[bt]["Lambda_star"] <= r[bt]["ipr_star"] * (1 + TOL))]
        viol[bt] = v
    out["P3_violations"] = {k: len(v) for k, v in viol.items()}
    if any(viol.values()):
        json.dump(out, open(os.path.join(RES, "e8_ipr_summary.json"), "w"), indent=1)
        sys.exit(f"bound violated: {viol} -- code error, see protocol section 4 (P3)")
    prim, labels = [], []
    for mod in ("FC", "SC"):
        cells = cells_of(rows, mod)
        m = {"n_networks": sum(len(c) for c in cells)}
        m["P1_F_vs_ell"] = strat_spearman(cells, F, ell, rng)
        m["P2_r_ell_given_F"] = strat_partial(cells, r_, ell, F, rng)
        m["P2_r_F_given_ell"] = strat_partial(cells, r_, F, ell, rng)
        for name in ("P1_F_vs_ell", "P2_r_ell_given_F", "P2_r_F_given_ell"):
            prim.append(m[name]["p"]); labels.append((mod, name))
        b1 = [r["b1.0"] for c in cells for r in c]
        dg = np.array([r["delta_grid"] for c in cells for r in c])
        di = np.array([b["delta_inf"] for b in b1]); lam = np.array([b["Lambda_star"] for b in b1])
        ipb = np.array([b["ipr_star"] for b in b1])
        m["P3_tightness_median_log2"] = {"ipr_star_over_delta_inf": float(np.median(np.log2(ipb / di))),
                                         "Lambda_star_over_delta_inf": float(np.median(np.log2(lam / di)))}
        lr = np.log2(dg / di)
        m["P4_log2_grid_over_inf"] = {"median": float(np.median(lr)),
                                      "iqr": [float(np.percentile(lr, 25)), float(np.percentile(lr, 75))],
                                      "n_grid_faster": int(np.sum(lr > 0))}
        m["P4_n_grid_above_ipr_star"] = int(np.sum(dg > ipb))
        m["P5_ell_slope_bits_per_node"] = slope_ci(cells, ell, rng)
        m["P5_ell_relative_to_ergodic_cell_medians"] = {str(c[0]["n"]): float(np.median([r["ell_pi"] + (r["n"] - 1)
                                                        for r in c])) for c in cells}
        # secondary, descriptive
        m["S_ell_vs_betaD"] = strat_spearman(cells, bD, ell, rng)["rho"]
        m["S_ell_vs_r"] = strat_spearman(cells, r_, ell, rng)["rho"]
        m["S_F_vs_r"] = strat_spearman(cells, F, r_, rng)["rho"]
        for alt in ("ell_ground", "ell_unif"):
            fa = (lambda key: (lambda r: r[key]))(alt)
            m[f"S_F_vs_{alt}"] = strat_spearman(cells, F, fa, rng)["rho"]
            m[f"S_r_{alt}_given_F"] = strat_partial(cells, r_, fa, F, rng)["rho"]
        out[mod] = m
    adj = holm(np.array(prim))
    for (mod, name), a in zip(labels, adj):
        out[mod][name]["p_holm"] = float(a)
        out[mod][name]["significant"] = bool(a < 0.05)
    json.dump(out, open(os.path.join(RES, "e8_ipr_summary.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("FC", "SC")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
