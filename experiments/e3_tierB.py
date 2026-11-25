"""E3 Tier B (docs/E3_scaling_protocol_2026-11-21.md, amendment 10): simulated chains, co-assignment accuracy.

Per instance (beta = 1/sigma_ROI): C independent chains of T steps from uniform random starts (same starts for every
chain type). Metric: RMSE over pairs i<j of the POOLED estimate of <z_i z_j> (mean over chains and steps, no burn-in
discarded) against exact enumeration, recorded at checkpoints. Chains: quantum quench (Chebyshev propagator; Layden
grid, untuned), local, label_swap, uniform, Swendsen-Wang, Wolff (one proposal or one cluster update per step), and
i.i.d. heat-bath at cost 2^(n-1) steps per sample (analytic RMSE). Sensitivity: every classical chain also gets n*T.

Usage: e3_tierB.py SIZES WORKERS C T     e.g.  e3_tierB.py 14 8 8 2000     (appends to results/e3_tierB_rows.jsonl)
Instances: n = 12 -> the first Tier A n = 12 instances (2 per dataset and kind) so simulated and exact metrics can be
compared; n >= 13 -> fresh seed 20261123, same generator.
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
from qsr2 import chains, ising  # noqa: E402
from instances import RES, e3_instances, key  # noqa: E402


def tierB_instances(sizes, per_kind_12=2):
    """Instance sets are generated with FIXED size lists, so they never depend on which sizes a run requests:
    n = 12 -> the Tier A instances (e3_instances() with its default sizes 7..12), first 2 per (dataset, kind);
    n = 14 -> e3_instances(sizes=[14, 16], seed=20261123, per_kind=2): all 2 per (dataset, kind);
    n = 16 -> the same call, first 1 per (dataset, kind)."""
    out = []
    if 12 in sizes:
        cnt = {}
        for d in e3_instances():
            if d["n"] != 12 or d["family"] == "P3_canonical":
                continue
            k = (d["ds"], d["kind"])
            if cnt.get(k, 0) < per_kind_12:
                cnt[k] = cnt.get(k, 0) + 1
                out.append(d)
    if 14 in sizes or 16 in sizes:
        cnt = {}
        for d in e3_instances(sizes=[14, 16], seed=20261123, per_kind=2):
            if d["family"] == "P3_canonical" or d["n"] not in sizes:
                continue
            k = (d["n"], d["ds"], d["kind"])
            cap = 2 if d["n"] == 14 else 1
            if cnt.get(k, 0) < cap:
                cnt[k] = cnt.get(k, 0) + 1
                out.append(d)
    return out


def checkpoints(T):
    cps = sorted({int(round(x)) for x in np.geomspace(25, T, 12)} | {T})
    return [c for c in cps if c <= T]


def simulate(step, starts, n, exact, T, cps):
    s = starts.copy()
    rc = chains.RunningCorr(len(starts), n, exact)
    iu = rc.iu
    out = {}
    for t in range(1, T + 1):
        s = step(s)
        rc.add(s)
        if t in cps:
            pooled = rc.S.sum(0)[iu[0], iu[1]] / (len(starts) * t)
            out[t] = float(np.sqrt(np.mean((pooled - rc.exact) ** 2)))
    return out


def run(args):
    d, C, T, seed = args
    t0 = time.time()
    n, J = d["n"], d["J"]
    E = ising.energies(J)
    beta = 1.0 / d["sigma"]
    exact = chains.exact_corr(J, E, beta)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, 2 ** n, C)
    cps = checkpoints(T)
    cps_n = sorted(set(cps) | {c * n for c in cps})
    N = 2 ** n
    row = {"key": list(key(d)), "family": d["family"], "ds": d["ds"], "kind": d["kind"], "n": n, "F": d["F"],
           "sigma": d["sigma"], "C": C, "T": T, "rmse": {}, "seconds": {}}
    kinds = {
        "local": lambda s: chains.local_step(s, n, E, beta, rng),
        "label_swap": lambda s: chains.label_swap_step(s, n, E, beta, rng),
        "uniform": lambda s: chains._mh(s, rng.integers(0, N, len(s)), E, beta, rng),
        "sw": lambda s: chains.sw_step(s, n, J, beta, rng),
        "wolff": lambda s: chains.wolff_step(s, n, J, beta, rng),
    }
    for name, step in kinds.items():
        ts = time.time()
        row["rmse"][name] = simulate(step, starts, n, exact, T * n, cps_n)   # includes the per-step checkpoints
        row["seconds"][name] = time.time() - ts
    ts = time.time()
    q = chains.ChebQuenchSampler(J, E, beta)
    row["rmse"]["quantum"] = simulate(lambda s: q.step(s, rng), starts, n, exact, T, cps)
    row["seconds"]["quantum"] = time.time() - ts
    w = np.exp(-beta * (E - E.min())); w /= w.sum()
    Z = ising.spins(n).astype(float)
    iu = np.triu_indices(n, 1)
    zz = Z[:, iu[0]] * Z[:, iu[1]]
    var = w @ (zz * zz) - (w @ zz) ** 2
    row["rmse"]["heat_bath"] = {t: (float(np.sqrt(np.mean(var) / (C * t / 2 ** (n - 1)))) if C * t >= 2 ** (n - 1) else None)
                                for t in cps_n}
    row["seconds"]["total"] = time.time() - t0
    return row


def main():
    lo, _, hi = sys.argv[1].partition("-")
    sizes = list(range(int(lo), int(hi or lo) + 1))
    workers, C, T = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    path = os.path.join(RES, "e3_tierB_rows.jsonl")
    done = {tuple(json.loads(l)["key"]) for l in open(path)} if os.path.exists(path) else set()
    inst = [d for d in tierB_instances(sizes) if key(d) not in done]
    print(f"{len(inst)} Tier B instances for n in {sizes}, C={C}, T={T} ({len(done)} already done)", flush=True)
    t0 = time.time()
    jobs = [(d, C, T, 1000 + i) for i, d in enumerate(inst)]
    with Pool(workers) as pool, open(path, "a") as f:
        for i, row in enumerate(pool.imap_unordered(run, jobs, chunksize=1)):
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"{i + 1}/{len(inst)} done ({row['family']} n={row['n']}, quantum {row['seconds']['quantum']:.0f}s, "
                  f"elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
