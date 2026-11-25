"""E7 runner (docs/E7_direct_sampling_protocol_2026-11-23.md).

For every E7 network: the exact output distribution of every direct sampler (qsr2/direct.py) and its metrics against
the exact Gibbs co-assignments at beta_tilde = 1 and 2; and the QeMCMC reference. For QeMCMC the QAOA-MC angle-0
kernel is rebuilt and its stationarity checked on every network; its gap and asymptotic variances are recomputed and
compared with the stored E4 / E3c rows where n <= 11; the quench kernel's gap and variances are taken from those rows.
Appends one JSON row per network to results/e7_rows.jsonl; networks already present (by key) are skipped, so a run
can be resumed. Usage: e7_run.py [workers]   (with OPENBLAS_NUM_THREADS=1)
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
from qsr2 import direct, ising, mcmc, proposals  # noqa: E402
from qsr2.exact_ps import qaoa_lumped  # noqa: E402
from qsr2.roi_exact import node_uncertainty  # noqa: E402
from instances import RES, key  # noqa: E402
from e7_instances import all_networks  # noqa: E402

SEED = 20261130
P_MAX = 4
T_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
DT, DT_CHECK = 0.01, 0.02
BTS = (1.0, 2.0)
ETAS = (0.2, 0.5)
OBJ_ID = {"energy": 0, "kl": 1, "oracle": 2}
RECOMPUTE_MAX_N = 11
DEPLOYABLE = (["optimiser", "greedy"] + [f"anneal_T{T:g}" for T in T_GRID] + [f"qaoa_energy_p{p}" for p in range(1, P_MAX + 1)]
              + [f"qaoa_kl_p{p}" for p in range(1, P_MAX + 1)])
ORACLE = [f"qaoa_oracle_p{p}" for p in range(1, P_MAX + 1)]
OUT = os.path.join(RES, "e7_rows.jsonl")


def skey(d):
    return tuple(json.loads(json.dumps(list(key(d)))))


def stored_rows():
    out = {}
    for l in open(os.path.join(RES, "e4_exact_rows.jsonl")):
        r = json.loads(l)
        out[tuple(r["key"])] = ("E4", r)
    for l in open(os.path.join(RES, "e3c_canon_rows.jsonl")):
        r = json.loads(l)
        out.setdefault(tuple(r["key"]), ("E3c", r))
    return out


STORED = stored_rows()


def rng(idx, obj, bt, k):
    return np.random.default_rng([SEED, idx, OBJ_ID[obj], int(round(10 * bt)), k])


def strip(res):
    return {str(p): rec for p, rec in res.items()}


def state_checks(full, n):
    return {"norm": float(abs(full.sum() - 1.0)), "flip": direct.flip_asymmetry(full, n)}


def qaoa_family(aE, n, res, F):
    D, Dfirst, chk = {}, {}, {}
    for p in range(1, P_MAX + 1):
        th = np.array(res[p]["theta"])
        full = np.abs(direct.qaoa_state(aE, th[:p], th[p:])) ** 2
        chk[p] = state_checks(full, n)
        D[p] = direct.lump(full, n)
        if "theta_first" in res[p]:
            tf = np.array(res[p]["theta_first"])
            Dfirst[p] = direct.qaoa_probs(aE, tf[:p], tf[p:])
    return D, Dfirst, chk


def stored_chain(src, st, bt, c):
    if src == "E4":
        ch = st[f"b{bt}"]["chains"][c]
        return float(ch["gap"]), float(np.max(ch["s2"]))
    return float(st[f"b{bt}_{c}_lumped"]), float(st[f"b{bt}_{c}_s2max"])


def run_one(job):
    idx, d = job
    t0 = time.time()
    J = d["J"]
    n = len(J)
    M = 2 ** (n - 1)
    E = ising.energies(J)
    Er = E[:M]
    a = proposals.layden_alpha(J, np.zeros(n))
    aE = np.ascontiguousarray(a * E)
    F = mcmc.coassignment_observables(n)
    src, st = STORED[skey(d)]
    row = {"key": list(skey(d)), "index": idx, "set": d["e7_set"], "ds": d["ds"], "roi": d["kind"], "n": n,
           "sigma": float(d["sigma"]), "F": float(d["F"]), "nodes": [int(i) for i in d["nodes"]],
           "labels": d.get("labels"), "alpha": float(a),
           "source": src, "checks": {}, "timing": {}}
    dists, first = {"optimiser": direct.optimiser_probs(Er), "greedy": direct.greedy_probs(J)}, {}
    ts = time.time()
    for T in T_GRID:
        name = f"anneal_T{T:g}"
        full = np.abs(direct.anneal_state(aE, T, DT)) ** 2
        P = direct.lump(full, n)
        P2 = direct.anneal_probs(aE, T, DT_CHECK)
        row["checks"][name] = dict(state_checks(full, n),
                                   trotter=float(np.max(np.abs(direct.coassign(P, F) - direct.coassign(P2, F)))))
        dists[name] = P
    row["timing"]["anneal"] = time.time() - ts
    ts = time.time()
    res = direct.optimise_qaoa(direct.make_objective("energy", aE, n), P_MAX, rng(idx, "energy", 0, 1), rng(idx, "energy", 0, 2))
    row["opt_energy"] = strip(res)
    D, Df, chk = qaoa_family(aE, n, res, F)
    for p in D:
        dists[f"qaoa_energy_p{p}"], first[f"qaoa_energy_p{p}"] = D[p], Df[p]
        row["checks"][f"qaoa_energy_p{p}"] = chk[p]
    row["timing"]["qaoa_energy"] = time.time() - ts
    for bt in BTS:
        beta = bt / d["sigma"]
        logpi, pi = direct.gibbs(Er, beta)
        p_t = direct.coassign(pi, F)
        out = {"beta": beta, "p": p_t.tolist(), "u": node_uncertainty(p_t, n).tolist(), "samplers": {},
               "first_set_floor": {}, "noisy_floor": {}, "checks": {}, "timing": {}}
        if src == "E4":
            out["checks"]["p_vs_stored"] = float(np.max(np.abs(p_t - np.array(st[f"b{bt}"]["p"]))))
        Db, Dbf = dict(dists), dict(first)
        ts = time.time()
        seeds_kl = {p: [row["opt_energy"][str(p)]["theta"]] + ([row["b1.0"]["opt_kl"][str(p)]["theta"]] if bt != BTS[0] else [])
                    for p in range(1, P_MAX + 1)}
        res_kl = direct.optimise_qaoa(direct.make_objective("kl", aE, n, logpi=logpi), P_MAX, rng(idx, "kl", bt, 1),
                                      rng(idx, "kl", bt, 2), extra=seeds_kl)
        out["opt_kl"] = strip(res_kl)
        D, Df, chk = qaoa_family(aE, n, res_kl, F)
        for p in D:
            Db[f"qaoa_kl_p{p}"], Dbf[f"qaoa_kl_p{p}"] = D[p], Df[p]
            out["checks"][f"qaoa_kl_p{p}"] = chk[p]
        out["timing"]["qaoa_kl"] = time.time() - ts
        ts = time.time()
        pol = direct.make_objective("oracle_max", aE, n, p_target=p_t, Fobs=F)
        extra = {p: [row["opt_energy"][str(p)]["theta"], res_kl[p]["theta"]] for p in range(1, P_MAX + 1)}
        res = direct.optimise_qaoa(direct.make_objective("oracle_smooth", aE, n, p_target=p_t, Fobs=F), P_MAX,
                                   rng(idx, "oracle", bt, 1), rng(idx, "oracle", bt, 2), polish=pol, extra=extra)
        out["opt_oracle"] = strip(res)
        D, _, chk = qaoa_family(aE, n, res, F)
        for p in D:
            Db[f"qaoa_oracle_p{p}"] = D[p]
            out["checks"][f"qaoa_oracle_p{p}"] = chk[p]
        out["timing"]["qaoa_oracle"] = time.time() - ts
        for name, P in Db.items():
            out["samplers"][name] = direct.metrics(P, logpi, p_t, F, n)
        for name, P in Dbf.items():
            out["first_set_floor"][name] = float(np.max(np.abs(direct.coassign(P, F) - p_t)))
        for name in DEPLOYABLE:
            pD = np.array(out["samplers"][name]["pD"])
            for eta in ETAS:
                out["noisy_floor"][f"{name}|{eta}"] = float(np.max(np.abs((1 - eta) * pD + eta / 2 - p_t)))
        ts = time.time()
        lp = -beta * (Er - Er.min())
        Q = qaoa_lumped(E, a, *proposals.QAOA_GRID[0])
        Pk = mcmc.mh_matrix(Q, Er, beta)
        del Q
        qe = {"a0_stationarity": float(np.max(np.abs(pi @ Pk - pi))), "a0_row_sum": float(np.max(np.abs(Pk.sum(1) - 1)))}
        if n <= RECOMPUTE_MAX_N:
            ev = mcmc._sym_eigs(Pk, lp)
            s2, _ = mcmc.asymptotic_variances(Pk, lp, F)
            qe["a0_recomputed"] = {"gap": float(max(1.0 - max(abs(ev[-2]), abs(ev[0])), 0.0)), "s2max": float(np.max(s2))}
        del Pk
        for c in ("quantum", "qaoa_a0"):
            g, s2m = stored_chain(src, st, bt, c)
            qe[c] = {"gap": g, "s2max": s2m, "calls": direct.qemcmc_calls(g, s2m)}
        out["qemcmc"] = qe
        out["timing"]["qemcmc"] = time.time() - ts
        row[f"b{bt}"] = out
    row["timing"]["total"] = time.time() - t0
    return row


def warm_up():
    J = np.array([[0.0, 1.0, -0.5], [1.0, 0.0, 0.3], [-0.5, 0.3, 0.0]])
    aE = np.ascontiguousarray(ising.energies(J))
    direct.qaoa_probs(aE, [0.1], [0.2])
    direct.anneal_probs(aE, 0.1, 0.05)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    nets = all_networks()
    assert len(nets) == 44 and sum(d["e7_set"] == "primary" for d in nets) == 26
    for d in nets:
        assert skey(d) in STORED, f"no stored QeMCMC row for {d['ds']} {d['kind']}"
    done = {tuple(json.loads(l)["key"]) for l in open(OUT)} if os.path.exists(OUT) else set()
    jobs = sorted([(i, d) for i, d in enumerate(nets) if skey(d) not in done], key=lambda j: -j[1]["n"])
    print(f"E7: {len(jobs)} networks to run ({len(done)} already done), {workers} workers", flush=True)
    warm_up()
    t0 = time.time()
    with open(OUT, "a") as f, Pool(workers) as pool:
        for k, row in enumerate(pool.imap_unordered(run_one, jobs)):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{k + 1}/{len(jobs)} done ({row['ds']} {row['roi']} n={row['n']}, {row['timing']['total']:.0f} s, "
                  f"elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
