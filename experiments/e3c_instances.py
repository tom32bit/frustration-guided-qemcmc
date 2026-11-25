"""E3c nested-growth instances (docs/E3c_exact_n13_protocol_2026-11-22.md).

Every E3 Tier A random ROI node set of size n0 is grown by ONE node, drawn uniformly from the candidate pool that
the set itself was drawn from:
  border sets      -> the max(2 n0, n0 + 2) most ambiguous nodes of its community pair (instances.e3_instances)
  within-module    -> its module
Growth seeds: n0 = 12 (the study, n = 13): 20261124.  n0 = 10 (design pilot, n = 11): 20261126.
The Tier A generator is replayed exactly (seed 20261122, sizes 7..12, 6 per kind) and every replayed node set is
asserted equal to instances.e3_instances(); the canonical ROIs consume no random numbers and are not replayed.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import networks, patches  # noqa: E402
from instances import e3_instances, key, sigma_of  # noqa: E402

GROWTH_SEED = {12: 20261124, 10: 20261126}
DATASETS = ("FC200", "FC400", "SC200", "SC400")


def tierA_with_pools(sizes=range(7, 13), seed=20261122, per_kind=6):
    rng = np.random.default_rng(seed)
    out, Ws = [], {}
    for ds in DATASETS:
        W, _ = networks.load_connectome(ds)
        Ws[ds] = W
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
                out.append(dict(ds=ds, n=n, kind="border-set", nodes=U.tolist(), pool=sorted(amb.tolist())))
                big = [cc for cc in comms if np.sum(c == cc) >= n]
                mod = np.where(c == big[r % len(big)])[0]
                M = np.sort(rng.choice(mod, n, replace=False))
                out.append(dict(ds=ds, n=n, kind="within-set", nodes=M.tolist(), pool=mod.tolist()))
    ref = [d for d in e3_instances() if d["family"] != "P3_canonical"]
    assert len(ref) == len(out) and all(r["nodes"] == o["nodes"] and r["ds"] == o["ds"] and r["kind"] == o["kind"]
                                        for r, o in zip(ref, out)), "Tier A replay mismatch"
    for o, r in zip(out, ref):
        o["parent_key"] = list(key(r))
    return out, Ws


def nested(n0):
    """Instances of size n0 + 1 grown from the Tier A sets of size n0 (order = Tier A generation order)."""
    base, Ws = tierA_with_pools()
    rng = np.random.default_rng(GROWTH_SEED[n0])
    out = []
    for d in base:
        if d["n"] != n0:
            continue
        rest = sorted(set(d["pool"]) - set(d["nodes"]))
        if not rest:
            continue
        new = int(rng.choice(rest))
        U = np.array(sorted(d["nodes"] + [new]))
        J, h = patches.standalone_patch(Ws[d["ds"]], U)
        e = dict(family=f"E3c_nested_{d['ds']}", ds=d["ds"], n=n0 + 1, J=J, h=h, sigma=sigma_of(J), kind=d["kind"],
                 nodes=U.tolist(), added_node=new, parent_key=d["parent_key"], pool_size=len(d["pool"]))
        e["F"] = patches.frustration(J, h)
        out.append(e)
    return out


def canonical13():
    """Optional descriptive extension: every canonical Yeo-7 x hemisphere ROI with exactly 13 parcels (Schaefer-100,
    -200, -400; FC and SC). No parent, so no paired change; r(13) is reported per instance."""
    out = []
    for ds in ("FC100", "FC200", "FC400", "SC100", "SC200", "SC400"):
        W, labels = networks.load_connectome(ds)
        for name, ix in sorted(patches.canonical_rois(labels, 13, 13).items()):
            J, h = patches.standalone_patch(W, ix)
            d = dict(family="E3c_canonical", ds=ds, n=13, J=J, h=h, sigma=sigma_of(J), kind=name, nodes=ix.tolist(),
                     added_node=None, parent_key=None, pool_size=None)
            d["F"] = patches.frustration(J, h)
            out.append(d)
    return out
