"""Instance sets. (1) The G1 field-free instances, rebuilt from seed and matched one-to-one to their G1 rows
(fingerprint + local-move gap at rel. 1e-9, as in G1b). (2) Fresh E3 instances (seed 20261122, E3 amendment 1)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import ising, mcmc, networks, patches, proposals  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")


def sigma_of(J):
    v = J[np.triu_indices(len(J), 1)]
    return float(np.std(v)) if len(v) > 1 else 1.0


def key(d):
    return (d["family"], d["ds"], d["n"], d["kind"], round(float(d["sigma"]), 10), round(float(d["F"]), 10))


def g1_field_free_matched():
    import g1_regime_map as g1
    g1rows = [json.loads(l) for l in open(os.path.join(RES, "g1_rows.jsonl"))]
    idx = {}
    for line, r in enumerate(g1rows):
        if not r["family"].startswith("P1"):
            idx.setdefault(key(r), []).append(line)
    inst = [d for d in g1.build_instances() if not d["family"].startswith("P1")]
    for d in inst:
        d["F"] = patches.frustration(d["J"], d["h"])
        loc = mcmc.gap_of(proposals.local(d["n"]), ising.energies(d["J"]), 1.0 / d["sigma"])
        hit = [L for L in idx.get(key(d), []) if abs(g1rows[L]["b1.0_local"] - loc) <= 1e-9 * max(loc, 1e-300)]
        if not hit:
            raise SystemExit(f"G1 instance not reproduced: {key(d)}")
        d["g1_line"] = hit[0]
        idx[key(d)].remove(hit[0])
    if sum(len(v) for v in idx.values()):
        raise SystemExit("G1 rows left unmatched")
    return inst


def e3_instances(sizes=range(7, 13), seed=20261122, per_kind=6,
                 datasets=("FC200", "FC400", "SC200", "SC400")):
    """P3 ROI node sets drawn exactly as in G1 (ambiguous border sets and within-module sets) with a fresh seed,
    plus canonical Yeo-7 x hemisphere ROIs whose parcel count is in `sizes`.
    NOTE: one random stream runs over all sizes, so the instance set depends on the `sizes` argument. Callers must
    always use the same size list for the same set (Tier A: the default 7..12; Tier B: [14, 16])."""
    rng = np.random.default_rng(seed)
    inst = []
    for ds in datasets:
        W, labels = networks.load_connectome(ds)
        c = networks.best_partition(W, 1.0, runs=30)
        B = networks.modularity_matrix(W, 1.0)
        comms = np.unique(c)
        inter = sorted(((W[np.ix_(c == a, c == b)].sum(), a, b) for i, a in enumerate(comms) for b in comms[i + 1:]),
                       reverse=True)
        pairs = [(a, b) for _, a, b in inter[:4]]
        for n in sizes:
            for r in range(per_kind):
                a, b = pairs[r % len(pairs)]
                cand = np.where((c == a) | (c == b))[0]
                hA = np.array([B[i, c == a].sum() for i in cand]); hB = np.array([B[i, c == b].sum() for i in cand])
                amb = cand[np.argsort(np.abs(hA - hB))[: max(2 * n, n + 2)]]
                U = np.sort(rng.choice(amb, n, replace=False))
                J, h = patches.standalone_patch(W, U)
                inst.append(dict(family=f"P3_roi_{ds}", ds=ds, n=n, J=J, h=h, sigma=sigma_of(J), kind="border-set",
                                 nodes=U.tolist()))
                big = [cc for cc in comms if np.sum(c == cc) >= n]
                M = np.sort(rng.choice(np.where(c == big[r % len(big)])[0], n, replace=False))
                J, h = patches.standalone_patch(W, M)
                inst.append(dict(family=f"P3_roi_{ds}", ds=ds, n=n, J=J, h=h, sigma=sigma_of(J), kind="within-set",
                                 nodes=M.tolist()))
        if labels is not None:
            for name, ix in patches.canonical_rois(labels, min(sizes), max(sizes)).items():
                J, h = patches.standalone_patch(W, ix)
                inst.append(dict(family="P3_canonical", ds=ds, n=len(ix), J=J, h=h, sigma=sigma_of(J), kind=name,
                                 nodes=ix.tolist()))
    for d in inst:
        d["F"] = patches.frustration(d["J"], d["h"])
    return inst
