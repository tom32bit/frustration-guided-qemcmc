"""E5 proper (docs/E5_hardware_n8_protocol_2026-11-22.md): choose the n = 8 hardware instances by a fixed rule.

Pool: FC networks with n = 8:
  - the three 8-parcel canonical Yeo-7 x hemisphere ROIs of Schaefer-100 (LH_DorsAttn, RH_SomMot, RH_Vis): new,
    anatomically defined;
  - the 24 E3 Tier A random ROI node sets at n = 8 (FC-200, FC-400).
Eligibility (beta = 1/sigma_ROI, beta_tilde = 1):
  (1) two-factor routing rule of E3 amendment 13:  F >= 0.10 and beta*D >= 1;
  (2) non-trivial community uncertainty: n_eff = 1 / sum_c p_c^2 >= 2 over partitions (E5 pilot lesson);
  (3) the hardware circuit has a real ideal advantage: r_a0 = log2(gap_qaoa_a0 / gap_oracle) >= 1 (exact,
      partition space; oracle = best of local, label-swap, uniform, Swendsen-Wang, Wolff, heat-bath per cost).
Order: eligible canonical ROIs first, then eligible random ROIs in a seeded random order (seed 20261125); take 3.
Fallback: if fewer than 3 meet (1)-(3), fill with (1)-(2) and r_a0 > 0 in the same order.
Uses no hardware or noise-model information. Writes results/e5_selection.json (refuses to overwrite).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from qsr2 import exact, ising, networks, patches  # noqa: E402
from instances import RES, e3_instances, key, sigma_of  # noqa: E402
from e3_posthoc_barrier import critical_depth  # noqa: E402

CANON = ("LH_DorsAttn", "RH_SomMot", "RH_Vis")
CL = ("local", "label_swap", "uniform", "sw", "wolff")


def candidates():
    out = []
    W, labels = networks.load_connectome("FC100")
    rois = patches.canonical_rois(labels, 8, 8)
    for name in CANON:
        ix = rois[name]
        J, h = patches.standalone_patch(W, ix)
        out.append(dict(family="E5_canonical_FC100", ds="FC100", n=8, J=J, h=h, sigma=sigma_of(J), kind=name,
                        nodes=ix.tolist(), F=patches.frustration(J, h)))
    out += [d for d in e3_instances() if d["n"] == 8 and d["family"] in ("P3_roi_FC200", "P3_roi_FC400")]
    return out


def features(d):
    n, J = d["n"], d["J"]
    E = ising.energies(J)
    beta = 1.0 / d["sigma"]
    M = 2 ** (n - 1)
    p = np.exp(-beta * (E[:M] - E[:M].min())); p /= p.sum()
    row = exact.compute_exact(J, d["sigma"], betas_tilde=(1.0,), full=False, asym=False, pt=False, qaoa_angles=True,
                              chains=CL + ("quantum",))
    orc = max([row[f"b1.0_{c}_lumped"] for c in CL] + [2.0 ** -(n - 1)])
    best = max(list(CL) + ["heat_bath"], key=lambda c: row[f"b1.0_{c}_lumped"] if c != "heat_bath" else 2.0 ** -(n - 1))
    return {"F": d["F"], "beta_D": critical_depth(E, n) * beta, "n_eff": float(1 / np.sum(p ** 2)),
            "p_max": float(p.max()), "gap_qaoa_a0": row["b1.0_qaoa_a0_lumped"], "gap_quench": row["b1.0_quantum_lumped"],
            "gap_oracle": orc, "oracle_chain": best,
            "gaps_classical": {c: row[f"b1.0_{c}_lumped"] for c in CL} | {"heat_bath_per_cost": 2.0 ** -(n - 1)},
            "r_a0": float(np.log2(row["b1.0_qaoa_a0_lumped"] / orc)),
            "r_quench": float(np.log2(row["b1.0_quantum_lumped"] / orc))}


def main():
    path = os.path.join(RES, "e5_selection.json")
    if os.path.exists(path):
        sys.exit(f"{path} exists; the selection is fixed and is not recomputed")
    tierA = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(os.path.join(RES, "e3_tierA_rows.jsonl"))}
    rows = []
    for d in candidates():
        f = features(d)
        k = list(key(d))
        ta = tierA.get(tuple(json.loads(json.dumps(k))))
        if ta is not None:                      # consistency with the Tier A rows (same code, same instance)
            f["tierA_check_max_rel_diff"] = max(abs(f["gaps_classical"][c] - ta[f"b1.0_{c}_lumped"]) / ta[f"b1.0_{c}_lumped"]
                                                for c in CL)
            f["tierA_check_qaoa_a0_rel_diff"] = abs(f["gap_qaoa_a0"] - ta["b1.0_qaoa_a0_lumped"]) / ta["b1.0_qaoa_a0_lumped"]
        f.update({"key": k, "family": d["family"], "ds": d["ds"], "kind": d["kind"], "n": d["n"],
                  "sigma": d["sigma"], "nodes": d["nodes"],
                  "rule_routing": bool(d["F"] >= 0.10 and f["beta_D"] >= 1.0), "rule_uncertainty": bool(f["n_eff"] >= 2.0)})
        f["eligible"] = bool(f["rule_routing"] and f["rule_uncertainty"] and f["r_a0"] >= 1.0)
        f["eligible_fallback"] = bool(f["rule_routing"] and f["rule_uncertainty"] and f["r_a0"] > 0.0)
        rows.append(f)
    canon = [r for r in rows if r["family"] == "E5_canonical_FC100"]
    rand = [r for r in rows if r["family"] != "E5_canonical_FC100"]
    perm = np.random.default_rng(20261125).permutation(len(rand))
    order = canon + [rand[i] for i in perm]
    chosen = [r for r in order if r["eligible"]][:3]
    if len(chosen) < 3:
        chosen += [r for r in order if r["eligible_fallback"] and r not in chosen][: 3 - len(chosen)]
    for i, r in enumerate(chosen):
        r["selected_rank"] = i + 1
    S = {"rule": __doc__, "candidates": rows, "selected": [r["key"] for r in chosen],
         "n_eligible": sum(r["eligible"] for r in rows), "n_eligible_fallback": sum(r["eligible_fallback"] for r in rows)}
    json.dump(S, open(path, "x"), indent=1)
    for r in rows:
        print(f"{r['family']:20s} {r['kind']:12s} F={r['F']:.3f} bD={r['beta_D']:5.2f} neff={r['n_eff']:5.2f} "
              f"r_a0={r['r_a0']:+.2f} r_q={r['r_quench']:+.2f} elig={int(r['eligible'])} "
              f"sel={r.get('selected_rank', '')} chk={r.get('tierA_check_max_rel_diff', float('nan')):.1e}")
    print("selected:", [(r["family"], r["kind"], r["nodes"]) for r in chosen])


if __name__ == "__main__":
    main()
