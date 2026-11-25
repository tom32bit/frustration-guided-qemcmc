"""G1 experiment (plan E2): exact spectral gaps on real-connectome partition landscapes at PHYSICALLY FIXED beta.

beta = beta_tilde / sigma, with sigma the energy scale of the NETWORK that defines the conditional (whole connectome
for P1/P2; the ROI network itself for P3) - never rescaled per patch. beta_tilde in {0.5, 1 (primary), 2}.

Families (2-label decisions, n = 4..10):
  P1_border_<DS>   : ambiguous nodes between two adjacent communities, conditioned on the rest (fields)   [FC100, SC100]
  P2_split_<DS>    : whole modules (4..10 nodes) found by weighted Leiden at gamma in 1.5..5, exact zero-field split
                                                                                    [FC100, FC200, FC400, SC100, SC200]
  P3_roi_<DS>      : standalone ROI sub-networks (random node sets: ambiguous border sets and within-module sets)
                                                                                    [FC100, FC200, SC100, SC200]
  P3_canonical     : anatomically defined Yeo-7 network x hemisphere with 4..10 parcels (Schaefer-100/200, FC & SC)
Proposals (per-step spectral gap): local, label_swap, uniform, quantum (Layden grid average, untuned),
  quantum_g{0..5} (each gamma, t-averaged; used for class-level tuning on held-out instances), qaoa_mc (palindromic,
  grid average, untuned). Exact block heat-bath = gap 1 at cost 2^n (added analytically in the analysis).
Parallel tempering (2 replicas, beta and beta/3, local+swap moves; exact sparse gap) for n <= 8 on P2/P3
(not on field-pinned P1, where classical local moves already win and ARPACK converges very slowly).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qsr2 import ising, mcmc, networks, patches, proposals  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
BETAS_T = [0.5, 1.0, 2.0]
SIZES = list(range(4, 11))
RNG = np.random.default_rng(20261121)


def sigma_of(J):
    v = J[np.triu_indices(len(J), 1)]
    return float(np.std(v)) if len(v) > 1 else 1.0


def build_instances():
    inst = []
    cache = {}
    for ds in ("FC100", "FC200", "FC400", "SC100", "SC200"):
        W, labels = networks.load_connectome(ds)
        cache[ds] = (W, labels)
    # P1 border (conditioned) and P3 random ROI sets
    for ds in ("FC100", "FC200", "SC100", "SC200"):
        W, _ = cache[ds]
        c = networks.best_partition(W, 1.0, runs=30)
        B = networks.modularity_matrix(W, 1.0)
        sig_net = networks.network_scale(W, 1.0)
        comms = np.unique(c)
        inter = sorted(((W[np.ix_(c == a, c == b)].sum(), a, b) for i, a in enumerate(comms) for b in comms[i + 1:]), reverse=True)
        pairs = [(a, b) for _, a, b in inter[:4]]
        for n in SIZES:
            for r in range(12):
                a, b = pairs[r % len(pairs)]
                cand = np.where((c == a) | (c == b))[0]
                hA = np.array([B[i, c == a].sum() for i in cand]); hB = np.array([B[i, c == b].sum() for i in cand])
                amb = cand[np.argsort(np.abs(hA - hB))[: max(2 * n, n + 2)]]
                U = np.sort(RNG.choice(amb, n, replace=False))
                J, h = patches.standalone_patch(W, U)
                inst.append(dict(family=f"P3_roi_{ds}", ds=ds, n=n, J=J, h=h, sigma=sigma_of(J), kind="border-set"))
                big = [cc for cc in comms if np.sum(c == cc) >= n]
                M = np.sort(RNG.choice(np.where(c == big[r % len(big)])[0], n, replace=False))
                J, h = patches.standalone_patch(W, M)
                inst.append(dict(family=f"P3_roi_{ds}", ds=ds, n=n, J=J, h=h, sigma=sigma_of(J), kind="within-set"))
                if ds in ("FC100", "SC100") and r < 8:
                    J, h = patches.border_patch(W, c, U, a, b, 1.0)
                    inst.append(dict(family=f"P1_border_{ds}", ds=ds, n=n, J=J, h=h, sigma=sig_net, kind="border"))
    # P2 exact splits
    for ds in ("FC100", "FC200", "FC400", "SC100", "SC200"):
        W, _ = cache[ds]
        seen = set()
        for g in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
            c = networks.best_partition(W, g, runs=10)
            sig_net = networks.network_scale(W, g)
            for m in np.unique(c):
                M = np.where(c == m)[0]
                if 4 <= len(M) <= 10 and tuple(M) not in seen:
                    seen.add(tuple(M))
                    J, h = patches.split_patch(W, M, g)
                    inst.append(dict(family=f"P2_split_{ds}", ds=ds, n=len(M), J=J, h=h, sigma=sig_net, kind=f"gamma={g}"))
    # P3 canonical ROI networks
    for ds in ("FC100", "FC200", "SC100", "SC200"):
        W, labels = cache[ds]
        for name, idx in patches.canonical_rois(labels, 4, 10).items():
            J, h = patches.standalone_patch(W, idx)
            inst.append(dict(family="P3_canonical", ds=ds, n=len(idx), J=J, h=h, sigma=sigma_of(J), kind=name))
    return inst


def run(d):
    t0 = time.time()
    n, J, h = d["n"], d["J"], d["h"]
    E = ising.energies(J, h)
    Qg = proposals.quench_by_gamma(E, J, h)
    props = {"local": proposals.local(n), "label_swap": proposals.label_swap(n), "uniform": proposals.uniform(n),
             "quantum": sum(Qg) / len(Qg), "qaoa_mc": proposals.qaoa_mc(E, J, h)}
    for i, Q in enumerate(Qg):
        props[f"quantum_g{i}"] = Q
    row = {k: d[k] for k in ("family", "ds", "n", "sigma", "kind")}
    row["F"] = patches.frustration(J, h)
    row["field_ratio"] = patches.field_to_coupling_ratio(J, h) if np.any(h) else 0.0
    for bt in BETAS_T:
        beta = bt / d["sigma"]
        for name, Q in props.items():
            row[f"b{bt}_{name}"] = mcmc.gap_of(Q, E, beta)
        if n <= 8 and bt in (1.0, 2.0) and not d["family"].startswith("P1"):
            try:
                row[f"b{bt}_pt"] = mcmc.pt_gap(E, n, [beta, beta / 3])
            except Exception as e:  # ARPACK non-convergence for extremely small gaps
                row[f"b{bt}_pt"] = None
                row[f"b{bt}_pt_err"] = str(e)[:80]
    row["seconds"] = time.time() - t0
    return row


def main():
    inst = build_instances()
    inst.sort(key=lambda d: -d["n"])
    print(f"{len(inst)} instances", flush=True)
    path = os.path.join(OUT, "g1_rows.jsonl")
    rows = []
    with Pool(8) as pool, open(path, "w") as f:
        for i, row in enumerate(pool.imap_unordered(run, inst, chunksize=1)):
            f.write(json.dumps(row) + "\n"); f.flush()
            rows.append(row)
            if i % 50 == 0:
                print(f"{i + 1}/{len(inst)} done", flush=True)
    with open(os.path.join(OUT, "g1_rows.json"), "w") as f:
        json.dump({"date": "2026-11-21", "betas_tilde": BETAS_T, "rows": rows}, f)
    from collections import Counter
    print(Counter((r["family"]) for r in rows))


if __name__ == "__main__":
    main()
