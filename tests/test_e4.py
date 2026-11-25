"""E4 code paths: qsr2/roi_exact.py (exact per-pair quantities) and qsr2/sampler_numba.py (end-to-end chain runs).
Known answers: the validated partition-space path (exact_ps.compute_ps), brute-force enumeration over all 2^n
labelled states, the MH mixture identity, and long-run statistics of the sampler against the exact kernels."""
import numba
import numpy as np
import pytest

from qsr2 import cluster, exact_ps, ising, mcmc, roi_exact, sampler_numba as smp

RNG = np.random.default_rng(2027)
SHARED = ("local", "label_swap", "uniform", "quantum", "qaoa_a0", "sw", "wolff")


def rand_J(n):
    J = np.triu(RNG.standard_normal((n, n)), 1)
    return J + J.T


def sigma_of(J):
    v = J[np.triu_indices(len(J), 1)]
    return float(np.std(v))


@pytest.mark.parametrize("n", [5, 6, 7])
def test_roi_exact_matches_compute_ps(n):
    J = rand_J(n)
    s = sigma_of(J)
    row = roi_exact.roi_exact(J, s, betas_tilde=(1.0, 2.0))
    ref = exact_ps.compute_ps(J, s, betas_tilde=(1.0, 2.0))
    for bt in (1.0, 2.0):
        ch = row[f"b{bt}"]["chains"]
        for c in SHARED:
            for mine, theirs in ((ch[c]["gap"], ref[f"b{bt}_{c}_lumped"]), (np.mean(ch[c]["s2"]), ref[f"b{bt}_{c}_s2mean"]),
                                 (np.max(ch[c]["s2"]), ref[f"b{bt}_{c}_s2max"])):
                assert abs(mine - theirs) <= 1e-9 * abs(theirs), (n, bt, c)
        assert abs(np.mean(row[f"b{bt}"]["var"]) - ref[f"b{bt}_varmean"]) <= 1e-9 * ref[f"b{bt}_varmean"]


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_coassignment_bruteforce(n):
    J = rand_J(n)
    beta = 0.8 / sigma_of(J) if n > 2 else 0.8
    E = ising.energies(J)
    w = np.exp(-beta * (E - E.min())); w /= w.sum()
    x = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
    iu = np.triu_indices(n, 1)
    p_bf = np.array([w @ (x[:, i] == x[:, j]) for i, j in zip(*iu)])
    lp, pi = roi_exact.gibbs(E[: 2 ** (n - 1)], beta)
    p, var = roi_exact.coassignment(pi, mcmc.coassignment_observables(n))
    assert np.max(np.abs(p - p_bf)) < 1e-12
    assert np.max(np.abs(var - 4 * p_bf * (1 - p_bf))) < 1e-12        # Var(z_i z_j) = 4 p (1 - p)


def test_noisy_mixture_is_mixture_of_mh_kernels():
    n, eps, beta = 5, 0.3, 1.1
    J = rand_J(n)
    Er = ising.energies(J)[: 2 ** (n - 1)]
    Q = roi_exact.lumped_proposals(J, ("uniform", "quantum"))
    lhs = mcmc.mh_matrix((1 - eps) * Q["quantum"] + eps * Q["uniform"], Er, beta)
    rhs = (1 - eps) * mcmc.mh_matrix(Q["quantum"], Er, beta) + eps * mcmc.mh_matrix(Q["uniform"], Er, beta)
    assert np.max(np.abs(lhs - rhs)) < 1e-14


def test_heat_bath_accounting_and_node_uncertainty():
    n = 6
    J = rand_J(n)
    row = roi_exact.roi_exact(J, sigma_of(J), betas_tilde=(1.0,))
    b = row["b1.0"]
    assert np.allclose(b["chains"]["heat_bath"]["s2"], np.array(b["var"]) * 2 ** (n - 1), rtol=1e-14)
    assert np.allclose(roi_exact.node_uncertainty(np.full(6, 0.5), 4), 1.0)
    assert np.allclose(roi_exact.node_uncertainty(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]), 4), 0.0)


@pytest.mark.parametrize("n", [2, 3])
def test_small_rois(n):
    J = rand_J(n)
    row = roi_exact.roi_exact(J, sigma_of(J) if n > 2 else 1.0, betas_tilde=(1.0, 2.0))
    for bt in (1.0, 2.0):
        for c, v in row[f"b{bt}"]["chains"].items():
            assert all(np.isfinite(v["s2"])) and min(v["s2"]) > 0, (n, c)
            if v["gap"] is not None:
                assert 0 < v["gap"] <= 1 + 1e-12, (n, c)
    SW, WO = cluster.sw_wolff_transitions(J, 0.9)
    SWb, WOb = exact_ps.cluster_lumped(J, 0.9)
    assert np.max(np.abs(SWb - mcmc.lump(SW, n))) < 1e-12 and np.max(np.abs(WOb - mcmc.lump(WO, n))) < 1e-12


def _setup(n=4, beta=1.0, seed=5):
    J = np.triu(np.random.default_rng(seed).standard_normal((n, n)), 1)
    J = J + J.T
    Er = ising.energies(J)[: 2 ** (n - 1)]
    lp, pi = roi_exact.gibbs(Er, beta)
    Q = roi_exact.lumped_proposals(J)
    SW, WO = exact_ps.cluster_lumped(J, beta)
    K = {c: mcmc.mh_matrix(Q[c], Er, beta) for c in ("local", "label_swap", "uniform", "quantum")}
    K["sw"], K["wolff"] = SW, WO
    return J, Er, lp, pi, Q, K


def _cdf_for(chain, Q, K, pi):
    if chain == "heat_bath":
        return smp.row_cdf(pi)
    if chain in ("sw", "wolff"):
        return smp.row_cdf(K[chain])
    if chain == "quantum":
        return smp.row_cdf(Q[chain])
    return np.ones((1, 1))


@pytest.mark.parametrize("chain", ["local", "label_swap", "uniform", "quantum", "sw", "wolff", "heat_bath"])
def test_sampler_stationary_and_variance(chain):
    n, beta = 4, 1.0
    J, Er, lp, pi, Q, K = _setup(n, beta)
    Fobs = mcmc.coassignment_observables(n)
    cdf = _cdf_for(chain, Q, K, pi)
    seeds = np.arange(1, 257, dtype=np.int64)
    steps = 20000
    H = smp.run_chains(smp.KINDS[chain], cdf, Er, beta, n, int(np.argmin(Er)), 500, steps, seeds)
    occ = H.sum(0) / H.sum()
    assert 0.5 * np.abs(occ - pi).sum() < 0.01, chain
    est = H @ Fobs / steps
    if chain == "heat_bath":
        s2 = pi @ (Fobs - pi @ Fobs) ** 2
    else:
        s2, _ = mcmc.asymptotic_variances(K[chain], lp, Fobs)
    ratio = est.var(0, ddof=1) * steps / s2
    assert 0.8 < np.mean(ratio) < 1.2, (chain, ratio)
    assert np.max(np.abs(est.mean(0) - pi @ Fobs) / np.sqrt(s2 / steps / len(seeds))) < 4.0, chain


def test_sampler_reproducible_across_threads():
    n, beta = 4, 1.0
    J, Er, lp, pi, Q, K = _setup(n, beta)
    cdf = smp.row_cdf(Q["quantum"])
    seeds = np.arange(100, 116, dtype=np.int64)
    old = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        a = smp.run_chains(0, cdf, Er, beta, n, 0, 10, 3000, seeds)
        numba.set_num_threads(old)
        b = smp.run_chains(0, cdf, Er, beta, n, 0, 10, 3000, seeds)
    finally:
        numba.set_num_threads(old)
    assert np.array_equal(a, b) and a.sum() == 16 * 3000


def test_e4_pipeline_dry(tmp_path, monkeypatch):
    """Whole E4 pipeline (sealed routing -> exact rows -> E4-B runs -> analysis) on SYNTHETIC ROIs in a temporary
    results folder: no connectome data are touched. One ROI per tiling is forced to be routed."""
    import json
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    import e4_analyze
    import e4_exact
    import e4_instances
    import e4_routing_predict
    import e4_runs
    from qsr2 import patches

    def fake_tiling(ds):
        g = np.random.default_rng(sum(map(ord, ds)))
        out = []
        for j, n in enumerate((3, 4, 5, 6)):
            J = np.triu(g.standard_normal((n, n)), 1); J = J + J.T
            d = dict(family="E4_tiling", ds=ds, n=n, kind=f"ROI{j}", nodes=list(range(10 * j, 10 * j + n)),
                     labels=[f"p{10 * j + i}" for i in range(n)], J=J, h=np.zeros(n), sigma=sigma_of(J), n_parcels_total=16)
            d["F"] = patches.frustration(J, None)
            out.append(d)
        return out

    monkeypatch.setattr(e4_instances, "tiling", fake_tiling)
    monkeypatch.setattr(e4_runs, "tiling", fake_tiling)
    for mod in (e4_routing_predict, e4_exact, e4_runs, e4_analyze):
        monkeypatch.setattr(mod, "RES", str(tmp_path))
    e4_routing_predict.main()
    pp = tmp_path / "e4_routing_predictions.json"
    P = json.loads(pp.read_text())
    for x in P["instances"]:
        if x["roi"] == "ROI3":
            x["routed_b1.0"] = True
    pp.write_text(json.dumps(P))
    e4_exact.main()
    e4_runs.main()
    monkeypatch.setattr(sys, "argv", ["e4_analyze.py", "full"])
    e4_analyze.main()
    S = json.loads((tmp_path / "e4_summary.json").read_text())
    for ds in ("FC100", "SC100"):
        T = S["tilings"][ds]
        assert "ROI3" in T["routed"] and 0 < T["rho"] < 10
        V = T["E4B"]
        assert V["H"]["V1"] and V["C"]["V1"], V
        assert V["H"]["V2"] and V["C"]["V2"], V
        assert V["H"]["V3"] and V["C"]["V3"], V
    assert (tmp_path / "e4_parcel_map.csv").exists() and (tmp_path / "e4_roi_table.csv").exists()
