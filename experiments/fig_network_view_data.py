"""Drawing data for the network-view figure: the right control network (9 regions) in FC and SC, side by side.
No test and no new claim. For each modality: the exact Gibbs distribution over the network's partitions
(beta = 1 / sigma, as in the whole-cortex maps), the co-assignment probabilities p_ij and node uncertainties u_i, and
the most likely partitions.
Self-checks: p_ij and u_i equal the stored whole-cortex results (results/e4_exact_rows.jsonl) to 1e-9.
Usage (from the repository root): python3 experiments/fig_network_view_data.py  ->  results/fig_network_view_data.json
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import ising  # noqa: E402
from e4_instances import tiling  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
ROI, TOP = "RH_Cont", 6


def h2(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main():
    stored = {}
    for line in open(os.path.join(RES, "e4_exact_rows.jsonl")):
        r = json.loads(line)
        stored[(r["ds"], r["roi"])] = r
    routed = {(r["ds"], r["roi"]): r["routed"] == "True" for r in csv.DictReader(open(os.path.join(RES, "e4_roi_table.csv")))}
    out = {"roi": ROI, "views": {}}
    for ds in ("FC100", "SC100"):
        d = {x["kind"]: x for x in tiling(ds)}[ROI]
        J, n, sigma = d["J"], d["n"], d["sigma"]
        M = 2 ** (n - 1)
        e = ising.energies(J)[:M]
        lp = -(e - e.min()) / sigma
        pi = np.exp(lp - lp.max())
        pi /= pi.sum()
        Z = 1 - 2 * ((np.arange(M)[:, None] >> np.arange(n)[None, :]) & 1)
        iu = np.triu_indices(n, 1)
        p = pi @ (Z[:, iu[0]] == Z[:, iu[1]]).astype(float)
        st = stored[(ds, ROI)]
        assert st["labels"] == d["labels"]
        assert np.allclose(p, st["b1.0"]["p"], atol=1e-9)
        H = np.zeros((n, n))
        H[iu] = h2(p)
        H = H + H.T
        u = H.sum(1) / (n - 1)
        assert np.allclose(u, st["b1.0"]["u"], atol=1e-9)
        top = []
        for r in np.argsort(-pi)[:TOP]:
            z = Z[r].copy()
            if (z > 0).sum() < (z < 0).sum():        # display convention: filled = the larger community
                z = -z
            top.append({"z": z.tolist(), "prob": float(pi[r]), "energy_beta": float(-lp[r])})
        srt = np.sort(pi)[::-1]
        out["views"][ds] = {"labels": d["labels"], "F": float(st["F"]), "routed": routed[(ds, ROI)],
                            "p": p.tolist(), "u": u.tolist(), "mean_u": float(u.mean()), "top": top,
                            "rest_prob": float(1 - srt[:TOP].sum()),
                            "partitions_for_90pct": int(np.searchsorted(np.cumsum(srt), 0.9) + 1),
                            "entropy_bits": float(-(pi[pi > 0] * np.log2(pi[pi > 0])).sum())}
        v = out["views"][ds]
        print(f"{ds} {ROI}: routed={v['routed']}, F={v['F']:.3f}, mean u={v['mean_u']:.3f} bits, "
              f"top probs={[round(x['prob'], 3) for x in top]}, 90% needs {v['partitions_for_90pct']} partitions, "
              f"entropy={v['entropy_bits']:.2f} bits")
    assert out["views"]["FC100"]["labels"] == out["views"]["SC100"]["labels"]
    json.dump(out, open(os.path.join(RES, "fig_network_view_data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
