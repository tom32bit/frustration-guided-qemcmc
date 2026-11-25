"""POST HOC consistency check (not pre-registered): the 8 canonical 13-parcel ROIs that both E3c (qsr2/exact_ps.py,
canonical extension) and E4 (qsr2/roi_exact.py) computed at n = 13. The two modules share the lumped proposal kernels
(lumped_local, qaoa_lumped, cluster_lumped) but compute gaps and asymptotic variances with separate code.
Compared, at beta_tilde = 1 and 2, for local, label_swap, uniform, quantum, qaoa_a0, sw, wolff: the gap, and the mean
and maximum over pairs of the asymptotic variance s2; for heat-bath, the mean Gibbs variance (E3c varmean).
Writes results/e4_e3c_consistency_n13.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from instances import RES  # noqa: E402

CH = ("local", "label_swap", "uniform", "quantum", "qaoa_a0", "sw", "wolff")


def main():
    e4 = {(r["ds"], r["roi"]): r for r in (json.loads(l) for l in open(os.path.join(RES, "e4_exact_rows.jsonl")))
          if r["n"] == 13}
    e3 = {(r["ds"], r["kind"]): r for r in (json.loads(l) for l in open(os.path.join(RES, "e3c_canon_rows.jsonl")))}
    out, worst = [], {}
    for k in sorted(set(e4) & set(e3)):
        a, b = e4[k], e3[k]
        assert abs(a["sigma"] - b["sigma"]) < 1e-12 and abs(a["F"] - b["F"]) < 1e-12 and a["nodes"] == b["nodes"]
        rec = {"ds": k[0], "roi": k[1], "rel": {}}
        for bt in ("1.0", "2.0"):
            c4 = a[f"b{bt}"]["chains"]
            for c in CH:
                pairs = {"gap": (c4[c]["gap"], b[f"b{bt}_{c}_lumped"]),
                         "s2mean": (float(np.mean(c4[c]["s2"])), b[f"b{bt}_{c}_s2mean"]),
                         "s2max": (float(np.max(c4[c]["s2"])), b[f"b{bt}_{c}_s2max"])}
                for q, (x, y) in pairs.items():
                    rel = abs(x - y) / abs(y)
                    rec["rel"][f"b{bt}_{c}_{q}"] = rel
                    worst[q] = max(worst.get(q, 0.0), rel)
            x, y = float(np.mean(a[f"b{bt}"]["var"])), b[f"b{bt}_varmean"]
            rec["rel"][f"b{bt}_varmean"] = abs(x - y) / abs(y)
            worst["varmean"] = max(worst.get("varmean", 0.0), rec["rel"][f"b{bt}_varmean"])
        rec["max_rel"] = max(rec["rel"].values())
        rec["max_key"] = max(rec["rel"], key=rec["rel"].get)
        out.append(rec)
    S = {"note": __doc__, "rois": len(out), "keys_per_roi": len(out[0]["rel"]) if out else 0,
         "max_rel_by_quantity": worst, "max_rel_overall": max(r["max_rel"] for r in out), "per_roi": out}
    json.dump(S, open(os.path.join(RES, "e4_e3c_consistency_n13.json"), "w"), indent=1)
    for r in out:
        print(f"{r['ds']:6s} {r['roi']:12s} max rel {r['max_rel']:.2e} ({r['max_key']})")
    print("by quantity:", {q: f"{v:.1e}" for q, v in worst.items()})


if __name__ == "__main__":
    main()
