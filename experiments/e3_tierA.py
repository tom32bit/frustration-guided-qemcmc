"""E3 Tier A (docs/E3_scaling_protocol_2026-11-21.md, amendments 1-11): exact gaps on FRESH instances, n = 7-12.
Usage: e3_tierA.py SIZES WORKERS    e.g.  e3_tierA.py 12 2   |   e3_tierA.py 7-11 7
(set OPENBLAS_NUM_THREADS per run: 4 for n = 12 with 2 workers, 1 otherwise). Appends to results/e3_tierA_rows.jsonl."""
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import exact  # noqa: E402
from instances import RES, e3_instances, key  # noqa: E402


def run(d):
    row = exact.compute_exact(d["J"], d["sigma"], pt=d["n"] <= 8)
    row.update({"key": list(key(d)), "nodes": d["nodes"], "F": d["F"], "family": d["family"], "ds": d["ds"],
                "kind": d["kind"], "sigma": d["sigma"]})
    return row


def main():
    lo, _, hi = sys.argv[1].partition("-")
    sizes = range(int(lo), int(hi or lo) + 1)
    workers = int(sys.argv[2])
    path = os.path.join(RES, "e3_tierA_rows.jsonl")
    done = set()
    if os.path.exists(path):
        done = {tuple(json.loads(l)["key"]) for l in open(path)}
    inst = [d for d in e3_instances() if d["n"] in sizes and key(d) not in done]
    inst.sort(key=lambda d: -d["n"])
    print(f"{len(inst)} instances to run for n in {list(sizes)} ({len(done)} already done)", flush=True)
    t0 = time.time()
    with Pool(workers) as pool, open(path, "a") as f:
        for i, row in enumerate(pool.imap_unordered(run, inst, chunksize=1)):
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"{i + 1}/{len(inst)} done ({row['family']} n={row['n']}, {row['timing']['total']:.0f}s, "
                  f"elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
