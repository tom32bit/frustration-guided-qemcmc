"""E6 runner (docs/E6_generality_protocol_2026-11-22.md): exact partition-space gaps and asymptotic variances
(qsr2.exact_ps.compute_ps, beta_tilde = 1 and 2, all chains incl. QAOA-MC angle 0) for the E6 instances.

Usage: e6_run.py fit|test     (set OPENBLAS_NUM_THREADS / NUMBA_NUM_THREADS in the environment)
  fit  -> sizes FIT_SIZES (7-10)
  test -> sizes TEST_SIZES (11-12); refused unless the sealed projection results/e6_projection.json exists
Appends one JSON row per instance to results/e6_rows.jsonl; instances already present (by key) are skipped.
Order: by size, then networks in turn, so a partial run stays balanced.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import exact_ps  # noqa: E402
from instances import RES, key  # noqa: E402
from e6_instances import NETS, e6_instances  # noqa: E402

FIT_SIZES, TEST_SIZES = (7, 8, 9, 10), (11, 12)


def main():
    stage = sys.argv[1]
    sizes = {"fit": FIT_SIZES, "test": TEST_SIZES}[stage]
    if stage == "test" and not os.path.exists(os.path.join(RES, "e6_projection.json")):
        sys.exit("the sealed projection results/e6_projection.json does not exist; run e6_project.py first")
    path = os.path.join(RES, "e6_rows.jsonl")
    done = {tuple(json.loads(l)["key"]) for l in open(path)} if os.path.exists(path) else set()
    inst = [d for d in e6_instances() if d["n"] in sizes]
    pos, cnt = [], {}
    for d in inst:
        c = (d["n"], d["ds"])
        pos.append(cnt.get(c, 0))
        cnt[c] = cnt.get(c, 0) + 1
    order = sorted(range(len(inst)), key=lambda i: (inst[i]["n"], pos[i], NETS.index(inst[i]["ds"])))
    inst = [inst[i] for i in order if tuple(json.loads(json.dumps(list(key(inst[i]))))) not in done]
    print(f"E6 {stage}: {len(inst)} instances to run ({len(done)} rows already present)", flush=True)
    t0 = time.time()
    with open(path, "a") as f:
        for i, d in enumerate(inst):
            row = exact_ps.compute_ps(d["J"], d["sigma"], betas_tilde=(1.0, 2.0))
            row.update({"key": list(key(d)), "family": d["family"], "ds": d["ds"], "kind": d["kind"], "nodes": d["nodes"],
                        "names": d["names"], "F": d["F"], "sigma": d["sigma"]})
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"{i + 1}/{len(inst)} done ({d['ds']} {d['kind']} n={d['n']}, {row['timing']['total']:.0f}s, "
                  f"elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
