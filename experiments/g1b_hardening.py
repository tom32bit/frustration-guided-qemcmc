"""G1b (protocol: docs/G1b_hardening_protocol_2026-11-21.md): stronger classical baselines on the field-free G1 instances.

Rebuilds the G1 instances with the same seeded code, matches each to its G1 row (quantum gaps are reused, not
recomputed), aborts if the recomputed local-move gap disagrees, then adds exact gaps of:
  sw, wolff                      (all n, beta_tilde in 0.5, 1, 2)
  pt2_r{r}                       (n <= 8, beta_tilde in 1, 2; replicas at beta, r*beta)
  pt3_r{r}                       (n <= 6, beta_tilde = 1;   replicas at beta, (1+r)beta/2, r*beta)
and records the off-diagonal density of J (Houdayer is a replica exchange when it is 1).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import cluster, ising, mcmc, patches, proposals  # noqa: E402
import g1_regime_map as g1  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results")
R2 = [0.5, 1 / 3, 0.2, 0.1, 1 / 30, 0.0]
R3 = [0.3, 0.1, 0.0]


def key(d):
    """Instance fingerprint: sigma alone is shared by all P2 modules of one network/gamma, so F is included."""
    return (d["family"], d["ds"], d["n"], d["kind"], round(float(d["sigma"]), 10), round(float(d["F"]), 10))


def run(d):
    t0 = time.time()
    n, J = d["n"], d["J"]
    E = ising.energies(J)
    row = {"key": list(key(d)), "g1_line": d["g1_line"], "density": cluster.offdiag_density(J)}
    for bt in g1.BETAS_T:
        beta = bt / d["sigma"]
        lp = -beta * (E - E.min())
        row[f"b{bt}_sw"] = mcmc.spectral_gap(cluster.sw_transition(J, beta), logpi=lp)
        row[f"b{bt}_wolff"] = mcmc.spectral_gap(cluster.wolff_transition(J, beta), logpi=lp)
        if n <= 8 and bt in (1.0, 2.0):
            for r in R2:
                try:
                    row[f"b{bt}_pt2_r{r:.4f}"] = mcmc.pt_gap(E, n, [beta, r * beta])
                except Exception as e:
                    row[f"b{bt}_pt2_r{r:.4f}"] = None
                    row.setdefault("errors", []).append(f"pt2 {bt} {r}: {str(e)[:60]}")
        if n <= 6 and bt == 1.0:
            for r in R3:
                try:
                    row[f"b{bt}_pt3_r{r:.4f}"] = mcmc.pt_gap(E, n, [beta, (1 + r) / 2 * beta, r * beta])
                except Exception as e:
                    row[f"b{bt}_pt3_r{r:.4f}"] = None
                    row.setdefault("errors", []).append(f"pt3 {r}: {str(e)[:60]}")
    row["seconds"] = time.time() - t0
    return row


def main():
    g1rows = [json.loads(l) for l in open(os.path.join(OUT, "g1_rows.jsonl"))]
    idx = {}
    for line, r in enumerate(g1rows):
        if not r["family"].startswith("P1"):
            idx.setdefault(key(r), []).append(line)
    inst = [d for d in g1.build_instances() if not d["family"].startswith("P1")]
    # one-to-one matching: same fingerprint AND same local-move gap (rel. 1e-9) -> this is also the determinism check
    unmatched = []
    for d in inst:
        d["F"] = patches.frustration(d["J"], d["h"])
        loc = mcmc.gap_of(proposals.local(d["n"]), ising.energies(d["J"]), 1.0 / d["sigma"])
        cands = idx.get(key(d), [])
        hit = [L for L in cands if abs(g1rows[L]["b1.0_local"] - loc) <= 1e-9 * max(loc, 1e-300)]
        if not hit:
            unmatched.append(key(d))
            continue
        d["g1_line"] = hit[0]
        cands.remove(hit[0])
    left = sum(len(v) for v in idx.values())
    print(f"{len(inst)} field-free instances; {len(unmatched)} unmatched; {left} G1 rows left over", flush=True)
    if unmatched or left:
        raise SystemExit(f"instance rebuild does not match G1: {unmatched[:3]}")
    inst.sort(key=lambda d: -d["n"])
    path = os.path.join(OUT, "g1b_rows.jsonl")
    with Pool(8) as pool, open(path, "w") as f:
        for i, row in enumerate(pool.imap_unordered(run, inst, chunksize=1)):
            f.write(json.dumps(row) + "\n"); f.flush()
            if i % 50 == 0:
                print(f"{i + 1}/{len(inst)} done ({row['key'][0]} n={row['key'][2]}, {row['seconds']:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
