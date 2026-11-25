"""E6 instances (docs/E6_generality_protocol_2026-11-22.md): sub-networks of NON-brain networks.

Networks (no downloads; the two real ones ship with networkx 3.4.2):
  karate     Zachary karate club, 34 nodes, weighted (interaction counts)
  lesmis     Les Miserables character co-occurrence, 77 nodes, weighted
  lfr_mu0.1, lfr_mu0.3, lfr_mu0.5
             LFR benchmarks (networkx.LFR_benchmark_graph): 250 nodes, degree exponent 2.5, community-size exponent
             1.5, average degree 10, minimum community 20, mixing mu as named; the first seed from 20261128 upward
             that the generator accepts; edge weights lognormal(0, 0.5) drawn in sorted edge order from a stream seeded
             with the accepted seed.
Node sets follow the E3 Tier A design (border sets across a community pair, within-module sets), adapted to sparse
networks. In dense brain connectomes a uniform random subset is connected; here most are not, so every set is grown
as a CONNECTED snowball: start at a seed node, then repeatedly add a uniformly chosen node of the pool adjacent to
the current set, until n nodes.
  Leiden partition (gamma = 1, 30 runs); the 4 most strongly connected community pairs, used in turn.
  border set: pool = the union of the pair's two communities; seed = a uniform choice among the max(2n, n + 2) most
              ambiguous nodes of the pair (as in E3); accepted only with at least 2 nodes from each community.
  within set: pool = one module with at least n nodes (modules used in turn); seed uniform in the module.
Sizes 7-12, 4 slots per kind and size; one random stream (seed 20261129) over networks in the order above, sizes
ascending. Up to 50 attempts per slot; a slot is skipped, and recorded, when no attempt gives an accepted set that
differs from the sets already drawn from the same network (e.g. when the pool's connected part has fewer than n
nodes).
Each set is analysed as its own field-free network (patches.standalone_patch), as in E3.
"""
import os
import sys

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import networks, patches  # noqa: E402
from instances import sigma_of  # noqa: E402

NETS = ("karate", "lesmis", "lfr_mu0.1", "lfr_mu0.3", "lfr_mu0.5")
SIZES = tuple(range(7, 13))
PER_KIND = 4
SEED_SETS, SEED_LFR = 20261129, 20261128


def load(name):
    meta = {"name": name}
    if name == "karate":
        G = nx.karate_club_graph()
    elif name == "lesmis":
        G = nx.les_miserables_graph()
    else:
        mu = float(name.split("mu")[1])
        G = None
        for s in range(SEED_LFR, SEED_LFR + 200):
            try:
                G = nx.LFR_benchmark_graph(250, 2.5, 1.5, mu, average_degree=10, min_community=20, seed=s)
                break
            except (nx.ExceededMaxIterations, ValueError):
                continue
        if G is None:
            raise RuntimeError(f"LFR generator failed for {name}")
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        w = np.random.default_rng(s)
        for u, v in sorted(G.edges()):
            G[u][v]["weight"] = float(w.lognormal(0.0, 0.5))
        meta["lfr_seed"] = s
    nodes = sorted(G.nodes(), key=str)
    W = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    W = np.clip((W + W.T) / 2, 0, None)
    np.fill_diagonal(W, 0)
    meta.update(nodes=len(nodes), edges=G.number_of_edges())
    return W, [str(x) for x in nodes], meta


def connected(A):
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(A[i]):
            if j not in seen:
                seen.add(int(j)); stack.append(int(j))
    return len(seen) == len(A)


def e6_instances(with_skipped=False):
    rng = np.random.default_rng(SEED_SETS)
    inst, skipped, metas = [], [], {}
    for net in NETS:
        W, names, meta = load(net)
        c = networks.best_partition(W, 1.0, runs=30)
        meta["communities"] = int(len(np.unique(c)))
        metas[net] = meta
        B = networks.modularity_matrix(W, 1.0)
        comms = np.unique(c)
        inter = sorted(((W[np.ix_(c == a, c == b)].sum(), a, b) for i, a in enumerate(comms) for b in comms[i + 1:]),
                       reverse=True)
        pairs = [(a, b) for _, a, b in inter[:4]]
        seen = set()

        def draw(pool, seeds, n, kind, sides=None):
            inpool = np.zeros(len(W), bool); inpool[pool] = True
            for _ in range(50):
                cur = [int(rng.choice(seeds))]
                member = np.zeros(len(W), bool); member[cur[0]] = True
                while len(cur) < n:
                    front = np.flatnonzero((W[cur].sum(0) > 0) & inpool & ~member)
                    if len(front) == 0:
                        break
                    j = int(rng.choice(front)); cur.append(j); member[j] = True
                if len(cur) < n:
                    continue
                U = np.sort(np.array(cur))
                if tuple(U.tolist()) in seen:
                    continue
                if sides is not None and min(np.sum(c[U] == sides[0]), np.sum(c[U] == sides[1])) < 2:
                    continue
                return U
            skipped.append({"net": net, "n": n, "kind": kind, "reason": "no new accepted connected set in 50 attempts"})
            return None

        def add(U, n, kind):
            if U is None:
                return
            seen.add(tuple(U.tolist()))
            J, h = patches.standalone_patch(W, U)
            d = dict(family=f"E6_{net}", ds=net, n=n, kind=kind, nodes=U.tolist(), names=[names[i] for i in U],
                     J=J, h=h, sigma=sigma_of(J))
            d["F"] = patches.frustration(J, h)
            inst.append(d)

        for n in SIZES:
            for r in range(PER_KIND):
                a, b = pairs[r % len(pairs)]
                cand = np.where((c == a) | (c == b))[0]
                if len(cand) >= n:
                    hA = np.array([B[i, c == a].sum() for i in cand]); hB = np.array([B[i, c == b].sum() for i in cand])
                    amb = cand[np.argsort(np.abs(hA - hB), kind="stable")[: max(2 * n, n + 2)]]
                    add(draw(cand, amb, n, "border-set", (a, b)), n, "border-set")
                else:
                    skipped.append({"net": net, "n": n, "kind": "border-set", "reason": f"pool {len(cand)} < n"})
                big = [cc for cc in comms if np.sum(c == cc) >= n]
                if big:
                    mod = np.where(c == big[r % len(big)])[0]
                    add(draw(mod, mod, n, "within-set"), n, "within-set")
                else:
                    skipped.append({"net": net, "n": n, "kind": "within-set", "reason": "no module with >= n nodes"})
    return (inst, skipped, metas) if with_skipped else inst


if __name__ == "__main__":
    inst, skipped, metas = e6_instances(with_skipped=True)
    for net, m in metas.items():
        g = [d for d in inst if d["ds"] == net]
        print(net, m, "| instances", len(g), "by n", {n: sum(d["n"] == n for d in g) for n in SIZES})
    print("total", len(inst), "| skipped", len(skipped), skipped[:6])
