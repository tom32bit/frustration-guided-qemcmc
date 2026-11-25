"""G1c (docs/G1c_partition_space_protocol_2026-11-21.md): partition-space gaps on the 858 field-free G1 instances."""
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import exact  # noqa: E402
from instances import RES, g1_field_free_matched, key  # noqa: E402


def run(d):
    row = exact.compute_exact(d["J"], d["sigma"])
    row.update({"key": list(key(d)), "g1_line": d["g1_line"]})
    return row


def main():
    t0 = time.time()
    inst = g1_field_free_matched()
    print(f"{len(inst)} instances matched to G1 ({time.time() - t0:.0f}s)", flush=True)
    inst.sort(key=lambda d: -d["n"])
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool, open(os.path.join(RES, "g1c_rows.jsonl"), "w") as f:
        for i, row in enumerate(pool.imap_unordered(run, inst, chunksize=1)):
            f.write(json.dumps(row) + "\n"); f.flush()
            if i % 50 == 0:
                print(f"{i + 1}/{len(inst)} done (n={row['n']}, {row['timing']['total']:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
