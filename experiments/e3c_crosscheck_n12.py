"""E3c cross-check (post hoc, 22 Nov 2026; not part of any endpoint).

E3c differences n = 13 values from qsr2.exact_ps.compute_ps (partition space) against n = 12 Tier A values from
qsr2.exact.compute_exact (full space, then lumped). The unit tests check that the two paths agree at n = 5-7. This
script checks them at the size actually differenced: it recomputes the Tier A n = 12 parents of the first FC and the
first SC study instance with compute_ps and compares every shared gap / variance key with the stored Tier A row.

Usage: e3c_crosscheck_n12.py    -> results/e3c_crosscheck_n12.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import exact_ps  # noqa: E402
from instances import RES, e3_instances, key  # noqa: E402


def norm(k):
    return tuple(json.loads(json.dumps(list(k))))


def main():
    rows = [json.loads(l) for l in open(os.path.join(RES, "e3c_rows.jsonl"))]
    par = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(os.path.join(RES, "e3_tierA_rows.jsonl"))}
    picks = [next(r for r in rows if r["ds"].startswith(m)) for m in ("FC", "SC")]
    inst = {norm(key(d)): d for d in e3_instances() if d["n"] == 12}
    out = []
    for r in picks:
        pk = tuple(r["parent_key"])
        d, ref = inst[pk], par[pk]
        t = time.time()
        new = exact_ps.compute_ps(d["J"], d["sigma"], betas_tilde=(1.0, 2.0))
        shared = sorted(k for k, v in new.items() if k.startswith("b") and isinstance(v, float) and k in ref)
        rel = {k: abs(new[k] - ref[k]) / abs(ref[k]) for k in shared}
        worst = max(rel, key=rel.get)
        out.append({"parent_key": list(pk), "keys_compared": len(shared), "max_rel_diff": rel[worst], "worst_key": worst,
                    "seconds": round(time.time() - t, 1)})
        print(out[-1], flush=True)
    json.dump({"note": "post hoc cross-check, not pre-registered", "checks": out}, open(os.path.join(RES, "e3c_crosscheck_n12.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
