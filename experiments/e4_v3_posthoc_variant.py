"""POST HOC sensitivity analysis (not pre-registered; does NOT change the pre-registered E4-B verdict).
V3' = V3 with the near-certain exclusion that the protocol applies to V1 and V2: on each network the binding pair is
the pair with the largest predicted variance among pairs with min(p, 1 - p) >= 0.01; networks with no such pair are
skipped (none of their pairs enters V1 or V2 either). Threshold unchanged (kappa <= 1.3).
Writes results/e4_v3_posthoc_variant.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from instances import RES  # noqa: E402
from e4_instances import PRIMARY  # noqa: E402
from e4_analyze import KAPPA_BINDING, P_MIN, oracle  # noqa: E402


def main():
    rows = [json.loads(l) for l in open(os.path.join(RES, "e4_exact_rows.jsonl"))]
    P = json.load(open(os.path.join(RES, "e4_routing_predictions.json")))
    R1 = {tuple(x["key"]): x["routed_b1.0"] for x in P["instances"]}
    runs = {(tuple(x["key"]), x["chain"]): x for x in map(json.loads, open(os.path.join(RES, "e4_run_rows.jsonl")))}
    out = {"note": __doc__, "tilings": {}}
    for ds in PRIMARY:
        res = {}
        for arm in ("H", "C"):
            binding, skipped = [], []
            for r in (r for r in rows if r["ds"] == ds):
                k = tuple(r["key"])
                ch = "quantum" if (arm == "H" and R1[k]) else oracle(r["b1.0"])
                x = runs[(k, ch)]
                p = np.array(x["p"]); keep = np.minimum(p, 1 - p) >= P_MIN
                if not keep.any():
                    skipped.append(r["roi"])
                    continue
                s2 = np.array(r["b1.0"]["chains"][ch]["s2"] if ch != "heat_bath" else r["b1.0"]["var"])
                ib = int(np.flatnonzero(keep)[np.argmax(s2[keep])])
                binding.append({"roi": r["roi"], "chain": ch, "pair_min_p": float(min(p[ib], 1 - p[ib])),
                                "kappa": float(x["kappa"][ib])})
            mk = max(b["kappa"] for b in binding)
            res[arm] = {"binding": binding, "skipped_all_pairs_near_certain": skipped, "max_binding_kappa": mk,
                        "V3_variant": bool(mk <= KAPPA_BINDING)}
        out["tilings"][ds] = res
        print(ds, {a: (round(res[a]["max_binding_kappa"], 3), res[a]["V3_variant"], res[a]["skipped_all_pairs_near_certain"]) for a in ("H", "C")})
    json.dump(out, open(os.path.join(RES, "e4_v3_posthoc_variant.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
