"""E3c runner (docs/E3c_exact_n13_protocol_2026-11-22.md): exact partition-space gaps and asymptotic variances
(qsr2.exact_ps.compute_ps, beta_tilde = 1 and 2) for the nested-growth instances of experiments/e3c_instances.py.

Usage: e3c_run.py pilot|study WORKERS     (set OPENBLAS_NUM_THREADS / NUMBA_NUM_THREADS in the environment)
  pilot -> design pilot, Tier A n0 = 10 sets grown to n = 11 -> results/e3c_pilot_rows.jsonl
  study -> Tier A n0 = 12 sets grown to n = 13              -> results/e3c_rows.jsonl
  canon -> optional: canonical 13-parcel ROIs (descriptive)  -> results/e3c_canon_rows.jsonl
Appends one JSON row per instance; instances already present (by key) are skipped, so a run can be resumed.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import exact_ps  # noqa: E402
from instances import RES, key  # noqa: E402
from e3c_instances import canonical13, nested  # noqa: E402

STAGES = {"pilot": (10, "e3c_pilot_rows.jsonl"), "study": (12, "e3c_rows.jsonl"), "canon": (None, "e3c_canon_rows.jsonl")}


def run(d):
    row = exact_ps.compute_ps(d["J"], d["sigma"], betas_tilde=(1.0, 2.0))
    row.update({"key": list(key(d)), "nodes": d["nodes"], "F": d["F"], "family": d["family"], "ds": d["ds"],
                "kind": d["kind"], "sigma": d["sigma"], "parent_key": d["parent_key"], "added_node": d["added_node"],
                "pool_size": d["pool_size"]})
    return row


def main():
    stage, workers = sys.argv[1], int(sys.argv[2])
    n0, fname = STAGES[stage]
    path = os.path.join(RES, fname)
    done = {tuple(json.loads(l)["key"]) for l in open(path)} if os.path.exists(path) else set()
    inst = nested(n0) if n0 is not None else canonical13()
    # round-robin over datasets (FC-200, FC-400, SC-200, SC-400) so that a partial run stays balanced by modality
    by = {}
    for d in inst:
        by.setdefault(d["ds"], []).append(d)
    if n0 is not None:
        inst = [d for grp in zip(*by.values()) for d in grp]
    inst = [d for d in inst if tuple(json.loads(json.dumps(list(key(d))))) not in done]
    print(f"{stage}: {len(inst)} instances to run ({len(done)} already done)", flush=True)
    t0 = time.time()
    it = map(run, inst) if workers == 1 else Pool(workers).imap_unordered(run, inst, chunksize=1)
    with open(path, "a") as f:
        for i, row in enumerate(it):
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"{i + 1}/{len(inst)} done ({row['family']} {row['kind']} n={row['n']}, {row['timing']['total']:.0f}s, "
                  f"elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
