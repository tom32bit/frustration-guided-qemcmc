"""POST HOC (22 Nov 2026, after the Tier A results): what separates Tier A instances where the quantum proposal wins
from those where it does not?

Motivation: on SC the per-instance r is bimodal at n = 10-12 (r ~ 0 when the local chain is already fast, r ~ +2.5..+5
otherwise) and the frustration index F does not separate the two groups well at every n.

Candidate mechanism: the critical depth D of the single-node-move energy landscape on partition space (the largest
energy barrier, over local minima, that must be crossed to reach a lower-energy partition; Metropolis-type chains
with single-node moves relax on a time scale ~ exp(beta * D) at low temperature). Computed exactly with union-find
over the 2^(n-1) partitions (single-node flips; the top-bit flip maps through the global flip).

Not a pre-registered test: reported as descriptive and as a hypothesis for a pre-registered follow-up. Cheap to
compute only because n <= 12 (exponential in n), so it is a mechanism, not a practical routing rule.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import ising  # noqa: E402
from instances import RES, e3_instances, key  # noqa: E402

CL = ["local", "label_swap", "uniform", "sw", "wolff"]


def critical_depth(E, n):
    """Max over basins of (saddle energy at which the basin first joins a basin with a lower minimum - its minimum),
    for single-node moves on partition space (representatives s < 2^(n-1))."""
    N, M = 2 ** n, 2 ** (n - 1)
    e = E[:M]
    order = np.argsort(e, kind="stable")
    parent = np.arange(M)
    mn = e.copy()
    active = np.zeros(M, bool)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    D = 0.0
    for s in order:
        active[s] = True
        for i in range(n):
            t = s ^ (1 << i)
            if t >= M:
                t ^= N - 1
            if not active[t]:
                continue
            a, b = find(s), find(t)
            if a == b:
                continue
            hi, lo = (a, b) if mn[a] > mn[b] else (b, a)
            D = max(D, e[s] - mn[hi])
            parent[hi] = lo
    return float(D)


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    return float(mannwhitneyu(np.asarray(score)[y], np.asarray(score)[~y]).statistic / (y.sum() * (~y).sum()))


def main():
    rows = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(os.path.join(RES, "e3_tierA_rows.jsonl"))}
    out = []
    for d in e3_instances():
        r = rows.get(tuple(json.loads(json.dumps(list(key(d))))))
        if r is None:
            continue
        n, J = d["n"], d["J"]
        E = ising.energies(J)
        beta = 1.0 / d["sigma"]
        M = 2 ** (n - 1)
        lp = -beta * (E[:M] - E[:M].min())
        p = np.exp(lp); p /= p.sum()
        top = np.argsort(-p)[:2]
        ham = bin(int(top[0]) ^ int(top[1])).count("1")
        orc = max([r[f"b1.0_{c}_lumped"] for c in CL] + [2.0 ** -(n - 1)])
        out.append({"key": r["key"], "family": d["family"], "n": n, "F": d["F"],
                    "beta_D": beta * critical_depth(E, n), "p_max": float(p.max()),
                    "n_eff": float(1.0 / np.sum(p ** 2)), "hamming_top2": int(min(ham, n - ham)),
                    "r": float(np.log2(r["b1.0_quantum_lumped"] / orc)),
                    "log2_local": float(np.log2(r["b1.0_local_lumped"])),
                    "log2_quantum": float(np.log2(r["b1.0_quantum_lumped"]))})
    S = {"note": __doc__, "instances": len(out), "by_modality": {}}
    for m in ("FC", "SC"):
        rr = [x for x in out if x["family"] != "P3_canonical" and x["family"].split("_")[-1][:2] == m]
        e = {}
        for n in range(7, 13):
            g = [x for x in rr if x["n"] == n]
            bD = [x["beta_D"] for x in g]; F = [x["F"] for x in g]; R = [x["r"] for x in g]; L = [x["log2_local"] for x in g]
            y = [v > 0 for v in R]
            e[n] = {"spearman_r_vs_betaD": round(float(spearmanr(R, bD)[0]), 2),
                    "spearman_r_vs_F": round(float(spearmanr(R, F)[0]), 2),
                    "spearman_log2local_vs_betaD": round(float(spearmanr(L, bD)[0]), 2),
                    "auc_win_betaD": None if auc(bD, y) is None else round(auc(bD, y), 2),
                    "auc_win_F": None if auc(F, y) is None else round(auc(F, y), 2),
                    "median_betaD": round(float(np.median(bD)), 2)}
        allR = [x["r"] for x in rr]; ally = [v > 0 for v in allR]
        e["pooled_7to12"] = {"auc_win_betaD": round(auc([x["beta_D"] for x in rr], ally), 3),
                             "auc_win_F": round(auc([x["F"] for x in rr], ally), 3),
                             "spearman_r_vs_betaD": round(float(spearmanr(allR, [x["beta_D"] for x in rr])[0]), 3),
                             "spearman_r_vs_F": round(float(spearmanr(allR, [x["F"] for x in rr])[0]), 3)}
        # win share by beta*D bin, n = 10..12
        big = [x for x in rr if x["n"] >= 10]
        bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 1e9)]
        e["win_share_by_betaD_bin_n10to12"] = {f"[{lo},{hi if hi < 1e9 else 'inf'})": {
            "n": sum(lo <= x["beta_D"] < hi for x in big),
            "win_share": (round(float(np.mean([x["r"] > 0 for x in big if lo <= x["beta_D"] < hi])), 2)
                          if any(lo <= x["beta_D"] < hi for x in big) else None),
            "mean_r": (round(float(np.mean([x["r"] for x in big if lo <= x["beta_D"] < hi])), 2)
                       if any(lo <= x["beta_D"] < hi for x in big) else None)} for lo, hi in bins}
        S["by_modality"][m] = e
    json.dump(S, open(os.path.join(RES, "e3_posthoc_barrier.json"), "w"), indent=1)
    with open(os.path.join(RES, "e3_posthoc_barrier_rows.jsonl"), "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    print(json.dumps(S["by_modality"], indent=1))


if __name__ == "__main__":
    main()
