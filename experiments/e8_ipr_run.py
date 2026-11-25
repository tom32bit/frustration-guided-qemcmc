"""E8 (docs/E8_ipr_protocol_2026-11-24.md): participation ratios, long-time quench chains and the Orfi-Sels bounds
for the 288 ROI sub-networks of the scaling study (E3 Tier A, n = 7-12).
Usage: OPENBLAS_NUM_THREADS=1 python3 experiments/e8_ipr_run.py WORKERS   (appends to results/e8_ipr_rows.jsonl;
networks already present are skipped, so an interrupted run can be resumed)."""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import ipr, ising  # noqa: E402
from instances import RES, e3_instances, key  # noqa: E402
from e3c_analyze import r_of  # noqa: E402

BETAS_TILDE = (0.5, 1.0, 2.0)


def run(job):
    d, stored = job
    t0 = time.time()
    J, n = d["J"], d["n"]
    E = ising.energies(J)
    M = 2 ** (n - 1)
    Er = E[:M]
    iprs, Q = ipr.ipr_and_qinf(E, J)
    out = {"key": list(key(d)), "n": n, "family": d["family"], "modality": d["ds"][:2], "kind": d["kind"],
           "F": stored["F"], "beta_D": stored["beta_D"], "r": stored["r"], "delta_grid": stored["delta_grid"]}
    for bt in BETAS_TILDE:
        beta = bt / d["sigma"]
        gap = ipr.chain_gap(Q, Er, beta)
        lam, ipb, klam, kipb = ipr.threshold_bounds(Q, Er, beta, iprs)
        out[f"b{bt}"] = {"delta_inf": gap, "Lambda_star": lam, "ipr_star": ipb, "cut_Lambda": klam, "cut_ipr": kipb}
    beta = 1.0 / d["sigma"]
    pi = np.exp(ipr.gibbs_log(Er, beta))
    ibar = iprs.mean(0)
    out["ell_pi"] = float(np.log2(pi @ ibar))
    out["ell_unif"] = float(np.log2(ibar.mean()))
    out["ell_ground"] = float(np.log2(ibar[int(np.argmin(Er))]))
    out["ell_pi_by_gamma"] = [float(np.log2(pi @ x)) for x in iprs]
    out["seconds"] = time.time() - t0
    return out


def main():
    workers = int(sys.argv[1])
    post = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(os.path.join(RES, "e3_posthoc_barrier_rows.jsonl"))}
    tier = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(os.path.join(RES, "e3_tierA_rows.jsonl"))}
    path = os.path.join(RES, "e8_ipr_rows.jsonl")
    done = {tuple(json.loads(l)["key"]) for l in open(path)} if os.path.exists(path) else set()
    jobs = []
    for d in e3_instances():
        k = key(d)
        if d["family"] == "P3_canonical" or k in done:
            continue
        row = tier[k]
        stored = {"F": post[k]["F"], "beta_D": post[k]["beta_D"], "r": r_of(row), "delta_grid": row["b1.0_quantum_lumped"]}
        assert np.isclose(stored["r"], post[k]["r"], atol=1e-9)          # same r as the barrier analysis
        jobs.append((d, stored))
    jobs.sort(key=lambda j: -j[0]["n"])
    print(f"{len(jobs)} networks to run ({len(done)} already done)", flush=True)
    t0 = time.time()
    with Pool(workers) as pool, open(path, "a") as f:
        for i, out in enumerate(pool.imap_unordered(run, jobs, chunksize=1)):
            f.write(json.dumps(out) + "\n"); f.flush()
            print(f"{i + 1}/{len(jobs)} n={out['n']} {out['seconds']:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
