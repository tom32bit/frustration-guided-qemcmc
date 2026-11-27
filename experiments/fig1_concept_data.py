"""Drawing data for Fig. 1 (how FG-QeMCMC works). No test and no new claim: every quantity uses the definitions of the
analyses, for one real network, so the figure shows data rather than an illustration.

  cortex   the Schaefer-100 FC connectome split into the 13 canonical networks of the whole-cortex maps (E4 tiling),
           with each network's within-network connections (weights normalised to the network's strongest) and
           whether the sealed rule routes it to the quantum proposal (results/e4_roi_table.csv)
  example  the right control network (FC, 9 regions; 85% of the classical cost in E4): partition energies in units of
           temperature (beta = 1 / sigma), the disconnectivity tree of its single-flip landscape (valleys and the
           ridges between them), the best partition z*, the best partition of the second valley z', and the
           probability that the hardware circuit (QAOA-MC, angles pi/8, pi/8, 0.5) proposes z' from z*
Self-checks: beta * D from the tree equals e3_posthoc_barrier.critical_depth; F equals the stored E4 value; the
proposal is symmetric with unit row sums.
Usage (from the repository root): python3 experiments/fig1_concept_data.py  ->  results/fig1_concept_data.json
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import ising, networks, patches, proposals  # noqa: E402
from e4_instances import tiling  # noqa: E402
from e3_posthoc_barrier import critical_depth  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
DS, EXAMPLE = "FC100", "RH_Cont"
QAOA_ANGLES = (np.pi / 8, np.pi / 8, 0.5)


def disconnectivity(e, n):
    """Merge tree of the single-flip landscape on partition space (representatives r < 2^(n-1)).
    Partitions are added in order of energy; a partition with no lower neighbour starts a valley (a leaf), and one
    that joins two valleys is the ridge (saddle) between them (an internal node). Returns leaves, merges, D."""
    M, N = len(e), 2 ** n
    order = np.argsort(e, kind="stable")
    parent = np.arange(M)
    node_of = {}                               # union-find root -> tree node id
    leaves, merges = [], []
    active = np.zeros(M, bool)
    lowest = e.copy()                          # lowest energy in each valley (by root)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    climb = 0.0
    for s in order:
        active[s] = True
        roots = set()
        for i in range(n):
            t = s ^ (1 << i)
            if t >= M:
                t ^= N - 1
            if active[t]:
                roots.add(find(t))
        if not roots:                          # a new valley
            node_of[s] = len(leaves) + len(merges)
            leaves.append({"id": node_of[s], "state": int(s), "energy": float(e[s])})
            continue
        roots = sorted(roots, key=lambda r: lowest[r])
        base = roots[0]
        parent[s] = base
        for r in roots[1:]:                    # s is the ridge between valley `base` and valley `r`
            climb = max(climb, e[s] - lowest[r])
            nid = len(leaves) + len(merges)
            merges.append({"id": nid, "energy": float(e[s]), "children": [node_of[base], node_of[r]]})
            parent[r] = base
            node_of[base] = nid
    return leaves, merges, float(climb)


def main():
    rois = {d["kind"]: d for d in tiling(DS)}
    W, labels = networks.load_connectome(DS)
    routed = {r["roi"]: r["routed"] == "True" for r in csv.DictReader(open(os.path.join(RES, "e4_roi_table.csv")))
              if r["ds"] == DS}
    cortex = []
    for name, d in sorted(rois.items()):
        ix = d["nodes"]
        w = W[np.ix_(ix, ix)]
        wmax = w.max()
        edges = [[labels[ix[i]], labels[ix[j]], float(w[i, j] / wmax)] for i in range(len(ix)) for j in range(i + 1, len(ix))
                 if w[i, j] > 0]
        cortex.append({"roi": name, "routed": routed[name], "labels": [labels[i] for i in ix], "edges": edges})

    d = rois[EXAMPLE]
    J, n, sigma = d["J"], d["n"], d["sigma"]
    beta = 1.0 / sigma
    E = ising.energies(J)
    M = 2 ** (n - 1)
    e = beta * (E[:M] - E[:M].min())          # energy in units of temperature, ground partition at 0
    leaves, merges, climb = disconnectivity(e, n)
    assert np.isclose(climb, beta * critical_depth(E, n)), (climb, beta * critical_depth(E, n))
    e4 = next(json.loads(l) for l in open(os.path.join(RES, "e4_exact_rows.jsonl"))
              if json.loads(l)["ds"] == DS and json.loads(l)["roi"] == EXAMPLE)
    assert np.isclose(patches.frustration(J), e4["F"])
    zstates = 1 - 2 * ((np.arange(M)[:, None] >> np.arange(n)[None, :]) & 1)
    best = int(np.argmin(e))
    other = sorted((l for l in leaves if l["state"] != best), key=lambda l: l["energy"])[0]
    z1, z2 = zstates[best], zstates[other["state"]]
    ham = int(min(np.sum(z1 != z2), np.sum(z1 == z2)))
    a = proposals.layden_alpha(J, np.zeros(n))
    Qf = proposals.qaoa_mc_proposal(E, a, *QAOA_ANGLES)
    N = 2 ** n
    r = np.arange(M)
    Q = Qf[np.ix_(r, r)] + Qf[np.ix_(r, r ^ (N - 1))]
    assert np.allclose(Q.sum(1), 1) and np.allclose(Q, Q.T, atol=1e-10)
    out = {"cortex": {"ds": DS, "networks": cortex},
           "example": {"ds": DS, "roi": EXAMPLE, "n": n, "labels": d["labels"], "F": float(e4["F"]),
                       "beta_D": climb, "J_over_sigma": (J / sigma).tolist(), "z_best": z1.tolist(),
                       "z_other": z2.tolist(), "hamming": ham, "dE_beta": float(e[other["state"]] - e[best]),
                       "accept_best_to_other": float(np.exp(-(e[other["state"]] - e[best]))),
                       "q_qaoa_best_to_other": float(Q[best, other["state"]]),
                       "q_local_best_to_other": 0.0 if ham > 1 else 1.0 / n,
                       "energies_beta": e.tolist(), "tree": {"leaves": leaves, "merges": merges}}}
    json.dump(out, open(os.path.join(RES, "fig1_concept_data.json"), "w"), indent=1)
    x = out["example"]
    print(f"{EXAMPLE}: n={n}, F={x['F']:.3f}, beta*D={climb:.2f}, valleys={len(leaves)}, hamming(z*, z')={ham}, "
          f"beta*dE={x['dE_beta']:.2f}, accept={x['accept_best_to_other']:.2f}, q_qaoa={x['q_qaoa_best_to_other']:.4f}")


if __name__ == "__main__":
    main()
