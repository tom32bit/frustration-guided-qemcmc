"""POST HOC (not pre-registered), second part of the E4-B V3 investigation: are the two V3 failures (FC100 LH_Limbic
and LH_DorsAttn, Wolff) consistent with the exact finite-run law of the estimator?
  LH_Limbic (n = 3, N = 2): the exact law of p_hat is discrete; the law of the realised kappa over R = 200 replicates
    is computed exactly by convolution, giving P(kappa >= observed).
  LH_DorsAttn (n = 8, N = 281): an INDEPENDENT numpy sampler (grouped inverse-CDF draws, numpy Generator, not the Numba
    code) runs 200,000 single replicates from the same start/burn/N. Its mean and MSE are compared with the exact
    finite-run law (experiments/e4_v3_investigation.finite_law); then 200,000 resampled groups of R = 200 give
    P(kappa >= observed).
Writes results/e4_v3_rare_events.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import ising, mcmc  # noqa: E402
from instances import RES, key  # noqa: E402
from e4_instances import tiling  # noqa: E402
from e4_v3_investigation import finite_law, kernel  # noqa: E402


def setup(roi):
    r = [json.loads(l) for l in open(os.path.join(RES, "e4_run_rows.jsonl"))]
    r = [x for x in r if x["ds"] == "FC100" and x["roi"] == roi and x["chain"] == "wolff"][0]
    d = [d for d in tiling("FC100") if d["kind"] == roi][0]
    ex = [json.loads(l) for l in open(os.path.join(RES, "e4_exact_rows.jsonl")) if json.loads(l)["key"] == r["key"]][0]
    n = r["n"]; M = 2 ** (n - 1)
    Er = ising.energies(d["J"])[:M]
    P = kernel(d["J"], "wolff", Er, ex["b1.0"]["beta"], n)
    F = mcmc.coassignment_observables(n)
    p = np.array(r["p"]); pred = np.array(r["pred"]); b = int(np.argmax(pred))
    return r, n, M, Er, P, F, p, pred, b


def limbic():
    r, n, M, Er, P, F, p, pred, b = setup("LH_Limbic")
    s0 = int(np.argmin(Er)); N, burn, R = r["steps"], r["burn"], r["R"]
    mu = np.zeros(M); mu[s0] = 1
    for _ in range(burn + 1):
        mu = mu @ P
    law = {}                                     # exact law of p_hat for the binding pair (N = 2)
    for x1 in range(M):
        for x2 in range(M):
            ph = (1 + (F[x1, b] + F[x2, b]) / 2) / 2
            law[ph] = law.get(ph, 0) + mu[x1] * P[x1, x2]
    vals = np.array(sorted(law)); probs = np.array([law[v] for v in vals])
    err2 = (vals - p[b]) ** 2
    mse_exact = float(probs @ err2)
    # exact law of the realised MSE = mean of R iid err2 draws: multinomial counts over the 3 values
    from scipy.stats import multinomial
    obs = r["kappa"][b]
    thr = (obs * pred[b]) ** 2 * R               # realised sum of err2 at the observed kappa
    tail = 0.0
    for k1 in range(R + 1):
        for k2 in range(R + 1 - k1):
            k0 = R - k1 - k2
            s = k0 * err2[0] + k1 * err2[1] + k2 * err2[2]
            if s >= thr * (1 - 1e-12):
                tail += multinomial.pmf([k0, k1, k2], R, probs)
    return {"roi": "LH_Limbic", "n": n, "burn": burn, "N": N, "R": R, "binding_pair_p": float(p[b]),
            "p_hat_values": vals.tolist(), "p_hat_probs": probs.tolist(),
            "kappa_observed": float(obs), "kappa_exact_finite": float(np.sqrt(mse_exact) / pred[b]),
            "P_kappa_ge_observed": float(tail)}


def dorsattn(B=200000, G=200000, seed=7):
    r, n, M, Er, P, F, p, pred, b = setup("LH_DorsAttn")
    s0 = int(np.argmin(Er)); N, burn, R = r["steps"], r["burn"], r["R"]
    cdf = np.cumsum(P, 1); cdf /= cdf[:, -1:]
    rng = np.random.default_rng(seed)
    s = np.full(B, s0)
    acc = np.zeros((B, F.shape[1]))
    for t in range(burn + N):
        u = rng.random(B)
        nxt = np.empty(B, dtype=np.int64)
        for st in np.unique(s):
            ix = np.nonzero(s == st)[0]
            nxt[ix] = np.minimum(np.searchsorted(cdf[st], u[ix], side="right"), M - 1)
        s = nxt
        if t >= burn:
            acc += F[s]
    ph = (1 + acc / N) / 2
    bias, mse = finite_law(P, s0, burn, N, F, 2 * p - 1)
    e2 = (ph[:, b] - p[b]) ** 2
    groups = rng.integers(0, B, (G, R))
    kap = np.sqrt(e2[groups].mean(1)) / pred[b]
    obs = r["kappa"][b]
    excursion = ph[:, b] < 1 - 1e-12 if p[b] > 0.5 else ph[:, b] > 1e-12
    se_mse = float(e2.std(ddof=1) / np.sqrt(B))
    return {"roi": "LH_DorsAttn", "n": n, "burn": burn, "N": N, "R": R, "binding_pair_p": float(p[b]),
            "independent_sampler_replicates": B,
            "mse_independent": float(e2.mean()), "mse_independent_se": se_mse, "mse_exact": float(mse[b]),
            "mse_z_independent_vs_exact": float((e2.mean() - mse[b]) / se_mse),
            "mean_error_se": float(ph[:, b].std(ddof=1) / np.sqrt(B)),
            "mean_error_independent": float(ph[:, b].mean() - p[b]), "mean_error_exact": float(bias[b]),
            "rmse_independent": float(np.sqrt(e2.mean())), "rmse_exact": float(np.sqrt(mse[b])),
            "rmse_independent_all_pairs_over_exact": [float(x) for x in np.percentile(np.sqrt(((ph - p) ** 2).mean(0)) / np.sqrt(mse), [0, 50, 100])],
            "share_of_replicates_with_excursion": float(excursion.mean()),
            "kappa_observed": float(obs), "kappa_exact_finite": float(np.sqrt(mse[b]) / pred[b]),
            "kappa_over_R200_groups_percentiles_5_50_95_99": [float(x) for x in np.percentile(kap, [5, 50, 95, 99])],
            "P_kappa_ge_observed": float((kap >= obs).mean())}


def main():
    out = {"note": __doc__, "LH_Limbic": limbic(), "LH_DorsAttn": dorsattn()}
    json.dump(out, open(os.path.join(RES, "e4_v3_rare_events.json"), "w"), indent=1)
    for k in ("LH_Limbic", "LH_DorsAttn"):
        print(k, json.dumps({a: v for a, v in out[k].items() if a not in ("p_hat_values",)}, indent=None))


if __name__ == "__main__":
    main()
