"""E4-A runner (docs/E4_whole_cortex_protocol_2026-11-22.md): exact per-ROI quantities (qsr2.roi_exact, beta_tilde
= 1 and 2) for every ROI of the E4 tilings, primary tilings first. Appends one JSON row per ROI to
results/e4_exact_rows.jsonl; ROIs already present (by key) are skipped, so a run can be resumed.
Usage: e4_exact.py   (set OPENBLAS_NUM_THREADS / NUMBA_NUM_THREADS in the environment)"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import roi_exact  # noqa: E402
from instances import RES, key  # noqa: E402
from e4_instances import all_instances  # noqa: E402


def main():
    path = os.path.join(RES, "e4_exact_rows.jsonl")
    done = {tuple(json.loads(l)["key"]) for l in open(path)} if os.path.exists(path) else set()
    inst = [d for d in all_instances() if tuple(json.loads(json.dumps(list(key(d))))) not in done]
    print(f"E4 exact: {len(inst)} ROIs to run ({len(done)} already done)", flush=True)
    t0 = time.time()
    with open(path, "a") as f:
        for i, d in enumerate(inst):
            row = roi_exact.roi_exact(d["J"], d["sigma"], betas_tilde=(1.0, 2.0))
            row.update({"key": list(key(d)), "ds": d["ds"], "roi": d["kind"], "nodes": d["nodes"], "labels": d["labels"],
                        "F": d["F"], "sigma": d["sigma"]})
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"{i + 1}/{len(inst)} done ({d['ds']} {d['kind']} n={d['n']}, {row['timing']['total']:.0f}s, "
                  f"elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
