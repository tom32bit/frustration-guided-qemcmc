"""E7 direct samplers (qsr2/direct.py). Known answers: dense-matrix references for QAOA and for the annealing
evolution (scipy expm / solve_ivp), flip symmetry and normalisation, limits (gamma = 0 or T -> 0 give uniform; a slow
anneal reaches the ground state), greedy end states are local minima and match a plain-loop implementation,
objective and metric definitions against brute force."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from qsr2 import direct, ising, mcmc, proposals, roi_exact

RNG = np.random.default_rng(7077)


def rand_J(n):
    J = np.triu(RNG.standard_normal((n, n)), 1)
    return J + J.T


def setup(n):
    J = rand_J(n)
    E = ising.energies(J)
    a = proposals.layden_alpha(J, np.zeros(n))
    return J, E, a


@pytest.mark.parametrize("n,p", [(4, 1), (5, 3), (6, 4)])
def test_qaoa_matches_dense(n, p):
    J, E, a = setup(n)
    g, b = RNG.uniform(-2, 2, p), RNG.uniform(0, 1.5, p)
    X = proposals.mixer(n)
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for gk, bk in zip(g, b):
        psi = expm(-1j * bk * X) @ (np.exp(-1j * gk * a * E) * psi)
    ref = direct.lump(np.abs(psi) ** 2, n)
    P = direct.qaoa_probs(a * E, g, b)
    assert np.max(np.abs(P - ref)) < 1e-12
    full = np.abs(direct.qaoa_state(a * E, g, b)) ** 2
    assert abs(full.sum() - 1) < 1e-12 and direct.flip_asymmetry(full, n) < 1e-13


def test_qaoa_limits():
    n = 5
    J, E, a = setup(n)
    M = 2 ** (n - 1)
    assert np.allclose(direct.qaoa_probs(a * E, [0.0, 0.0], [0.3, 1.1]), 1 / M, atol=1e-14)
    assert np.allclose(direct.qaoa_probs(a * E, [0.7, -1.2], [0.0, 0.0]), 1 / M, atol=1e-14)
    # mixer period pi/2 on flip-even states
    P1 = direct.qaoa_probs(a * E, [0.7, 0.4], [0.3, 0.9])
    P2 = direct.qaoa_probs(a * E, [0.7, 0.4], [0.3 + np.pi / 2, 0.9 - np.pi / 2])
    assert np.max(np.abs(P1 - P2)) < 1e-12


def test_anneal_matches_ode():
    n, T = 4, 3.0
    J, E, a = setup(n)
    X = proposals.mixer(n)
    D = np.diag(a * E)

    def rhs(t, y):
        s = t / T
        return -1j * ((-(1 - s)) * (X @ y) + s * (D @ y))

    y0 = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    sol = solve_ivp(rhs, (0, T), y0, rtol=1e-11, atol=1e-12, method="DOP853")
    ref = direct.lump(np.abs(sol.y[:, -1]) ** 2, n)
    P = direct.anneal_probs(a * E, T, 0.002)
    assert np.max(np.abs(P - ref)) < 2e-6
    # second order: halving dt cuts the error about 4x
    e1 = np.max(np.abs(direct.anneal_probs(a * E, T, 0.04) - ref))
    e2 = np.max(np.abs(direct.anneal_probs(a * E, T, 0.02) - ref))
    assert 3.0 < e1 / e2 < 5.0


def test_anneal_limits():
    n = 5
    J, E, a = setup(n)
    M = 2 ** (n - 1)
    assert np.allclose(direct.anneal_probs(a * E, 1e-9, 1e-9), 1 / M, atol=1e-8)
    Jf = np.ones((n, n)) - np.eye(n)                    # ferromagnet: unique ground bipartition (all together)
    Ef = ising.energies(Jf)
    af = proposals.layden_alpha(Jf, np.zeros(n))
    P = direct.anneal_probs(af * Ef, 200.0, 0.02)
    assert P[0] > 0.99 and abs(P.sum() - 1) < 1e-12


def _greedy_loop(J):
    n = len(J)
    M = 2 ** (n - 1)
    Z = ising.spins(n)[:M].astype(float)
    out = np.zeros(M)
    for z in Z:
        z = z.copy()
        while True:
            dE = 2 * z * (J @ z)
            k = int(np.argmin(dE))
            if dE[k] < -1e-12 * np.abs(J).max():
                z[k] *= -1
            else:
                break
        s = int(((1 - z) / 2).astype(int) @ (1 << np.arange(n)))
        out[s if s < M else 2 ** n - 1 - s] += 1
    return out / M


@pytest.mark.parametrize("n", [3, 5, 7])
def test_greedy(n):
    J = rand_J(n)
    P = direct.greedy_probs(J)
    assert np.allclose(P, _greedy_loop(J), atol=1e-15)
    Er = ising.energies(J)[: 2 ** (n - 1)]
    Z = ising.spins(n)[: 2 ** (n - 1)].astype(float)
    for s in np.where(P > 0)[0]:                         # every end state is a single-flip local minimum
        assert np.all(2 * Z[s] * (J @ Z[s]) >= -1e-12 * np.abs(J).max())
    Jf = np.ones((n, n)) - np.eye(n)
    assert direct.greedy_probs(Jf)[0] == 1.0


def test_optimiser_and_coassign():
    n = 6
    J, E, a = setup(n)
    Er = E[: 2 ** (n - 1)]
    P = direct.optimiser_probs(Er)
    assert P.sum() == 1 and P[np.argmin(Er)] == 1
    F = mcmc.coassignment_observables(n)
    pD = direct.coassign(P, F)
    z = ising.spins(n)[np.argmin(Er)]
    iu = np.triu_indices(n, 1)
    assert np.array_equal(pD, (z[iu[0]] == z[iu[1]]).astype(float))


def test_gibbs_matches_roi_exact_and_kl():
    n = 6
    J, E, a = setup(n)
    Er = E[: 2 ** (n - 1)]
    beta = 0.9
    logpi, pi = direct.gibbs(Er, beta)
    _, pi2 = roi_exact.gibbs(Er, beta)
    assert np.max(np.abs(pi - pi2)) < 1e-15
    assert abs(direct.kl(pi, logpi)) < 1e-13
    Q = RNG.dirichlet(np.ones(len(pi)))
    assert direct.kl(Q, logpi) > 0
    assert abs(direct.kl(Q, logpi) - np.sum(Q * np.log(Q / pi))) < 1e-12


def test_calls_and_metrics():
    d = np.array([0.0, 0.01, -0.015])
    v = np.array([0.25, 0.09, 0.01])
    S = direct.calls_to_eps(d, v, 0.02)
    assert S == np.ceil(max(v / (0.02 ** 2 - d ** 2)))
    assert np.all(d ** 2 + v / S <= 0.02 ** 2 + 1e-15)
    assert direct.calls_to_eps(np.array([0.02, 0.0]), v[:2], 0.02) == float("inf")
    assert direct.calls_to_eps(np.array([0.001, -0.019]), np.zeros(2), 0.02) == 1.0     # deterministic, within eps (amendment 1)
    assert direct.qemcmc_calls(0.1, 0.4, 0.02) == 100 + 250
    n = 5
    J, E, a = setup(n)
    Er = E[: 2 ** (n - 1)]
    F = mcmc.coassignment_observables(n)
    logpi, pi = direct.gibbs(Er, 1.0)
    p = direct.coassign(pi, F)
    m = direct.metrics(pi, logpi, p, F, n)
    assert m["floor"] < 1e-14 and m["tv"] < 1e-14 and abs(m["suppression_ratio"] - 1) < 1e-12 and m["du_abs_mean"] < 1e-12
    mo = direct.metrics(direct.optimiser_probs(Er), logpi, p, F, n)
    assert mo["kl_pi_P_bits"] == float("inf") and mo["u_mean"] == 0.0


def test_objectives_and_optimiser_monotone():
    n = 5
    J, E, a = setup(n)
    Er = E[: 2 ** (n - 1)]
    F = mcmc.coassignment_observables(n)
    logpi, pi = direct.gibbs(Er, 1.0 / np.std(J[np.triu_indices(n, 1)]))
    p = direct.coassign(pi, F)
    th = np.array([0.4, -0.3, 0.2, 0.9])
    P = direct.qaoa_probs(a * E, th[:2], th[2:])
    assert abs(direct.make_objective("energy", a * E, n)(th) - P @ (a * Er)) < 1e-12
    assert abs(direct.make_objective("kl", a * E, n, logpi=logpi)(th) - direct.kl(P, logpi)) < 1e-12
    mx = direct.make_objective("oracle_max", a * E, n, p_target=p, Fobs=F)(th)
    sm = direct.make_objective("oracle_smooth", a * E, n, p_target=p, Fobs=F)(th)
    assert abs(mx - np.max(np.abs(direct.coassign(P, F) - p))) < 1e-14 and mx <= sm <= mx * len(p) ** (1 / 16) + 1e-14
    rng, rng2 = np.random.default_rng(1), np.random.default_rng(2)
    for kind, pol in (("kl", None), ("oracle_smooth", direct.make_objective("oracle_max", a * E, n, p_target=p, Fobs=F))):
        f = direct.make_objective(kind, a * E, n, logpi=logpi, p_target=p, Fobs=F)
        res = direct.optimise_qaoa(f, 3, rng, rng2, n_random=2, n_second=2, polish=pol)
        vals = [res[q]["value"] for q in (1, 2, 3)]
        assert all(vals[i + 1] <= vals[i] + 1e-12 for i in range(2)), (kind, vals)
        th = np.array(res[3]["theta"])
        chk = pol(th) if pol is not None else f(th)
        assert abs(chk - res[3]["value"]) < 1e-12


def test_runner_dry_run(monkeypatch):
    """experiments/e7_run.run_one end to end on a random model that is not an E7 network, with a stored reference
    row produced by roi_exact (the E4 code): every check field present and passing, the oracle floor never above
    the energy or KL QAOA floor at the same depth, and depth monotonicity of the KL and oracle objectives."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    import e7_run
    from qsr2 import patches
    n = 5
    J = rand_J(n)
    sig = float(np.std(J[np.triu_indices(n, 1)]))
    d = dict(family="E7_dry", ds="TEST", n=n, kind="random5", J=J, h=np.zeros(n), sigma=sig,
             F=patches.frustration(J, np.zeros(n)), nodes=list(range(n)), e7_set="test")
    monkeypatch.setitem(e7_run.STORED, e7_run.skey(d), ("E4", roi_exact.roi_exact(J, sig, betas_tilde=(1.0, 2.0))))
    row = e7_run.run_one((0, d))
    for name, c in row["checks"].items():
        assert c["norm"] < 1e-10 and c["flip"] < 1e-10 and c.get("trotter", 0.0) < 1e-3, name
    for bt in (1.0, 2.0):
        b = row[f"b{bt}"]
        assert b["checks"]["p_vs_stored"] < 1e-12
        for name, c in b["checks"].items():
            if name != "p_vs_stored":
                assert c["norm"] < 1e-10 and c["flip"] < 1e-10, name
        qe = b["qemcmc"]
        assert qe["a0_stationarity"] < 1e-12 and qe["a0_row_sum"] < 1e-12
        for k in ("gap", "s2max"):
            assert abs(qe["a0_recomputed"][k] - qe["qaoa_a0"][k]) <= 1e-9 * qe["qaoa_a0"][k]
        assert set(e7_run.DEPLOYABLE + e7_run.ORACLE) <= set(b["samplers"])
        S = b["samplers"]
        for p in range(1, e7_run.P_MAX + 1):
            fo = S[f"qaoa_oracle_p{p}"]["floor"]
            assert fo <= S[f"qaoa_kl_p{p}"]["floor"] + 1e-12 and fo <= S[f"qaoa_energy_p{p}"]["floor"] + 1e-12
            if p > 1:
                assert fo <= S[f"qaoa_oracle_p{p - 1}"]["floor"] + 1e-12
                assert b["opt_kl"][str(p)]["value"] <= b["opt_kl"][str(p - 1)]["value"] + 1e-12
        assert np.isfinite(qe["quantum"]["calls"]) and np.isfinite(qe["qaoa_a0"]["calls"])


def test_analysis_runs_on_synthetic_rows(tmp_path, monkeypatch):
    """experiments/e7_analyze.main on 44 synthetic rows (two runner rows from random models, not E7 networks,
    replicated with distinct keys): the gate passes, the endpoints and tables are produced, and the counts are
    consistent with the per-network floors."""
    import json
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    import e7_analyze
    import e7_run
    from qsr2 import patches
    import e7_sensitivity
    base, based = [], []
    for n in (4, 5):
        J = rand_J(n)
        sig = float(np.std(J[np.triu_indices(n, 1)]))
        d = dict(family="E7_dry", ds="TEST", n=n, kind=f"random{n}", J=J, h=np.zeros(n), sigma=sig,
                 F=patches.frustration(J, np.zeros(n)), nodes=list(range(n)), e7_set="test")
        monkeypatch.setitem(e7_run.STORED, e7_run.skey(d), ("E4", roi_exact.roi_exact(J, sig, betas_tilde=(1.0, 2.0))))
        base.append(e7_run.run_one((0, d)))
        based.append(d)
    fakes = []
    with open(tmp_path / "e7_rows.jsonl", "w") as f:
        for i in range(44):
            r = json.loads(json.dumps(base[i % 2]))
            r["key"] = r["key"][:3] + [f"syn{i}"] + r["key"][4:]
            r["index"], r["set"], r["roi"] = i, ("primary" if i < 26 else "secondary"), f"syn{i}"
            r["ds"] = "FCSYN" if i % 2 else "SCSYN"
            if i == 0:                                   # force one primary row into the sensitivity band
                for nm in e7_run.DEPLOYABLE:
                    r["b1.0"]["samplers"][nm]["floor"] = 0.025 if nm == "greedy" else max(0.03, r["b1.0"]["samplers"][nm]["floor"])
            fakes.append(dict(based[i % 2], kind=f"syn{i}", e7_set=r["set"]))
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr(e7_analyze, "RES", str(tmp_path))
    monkeypatch.setattr(e7_sensitivity, "RES", str(tmp_path))
    monkeypatch.setattr(e7_sensitivity, "all_networks", lambda: fakes)
    e7_sensitivity.main()
    sens = json.load(open(tmp_path / "e7_sensitivity.json"))
    assert len(sens["networks"]) >= 1 and sens["networks"][0]["network"] == "SCSYN syn0"
    assert sens["networks"][0]["new_best_floor"] <= 0.025
    e7_analyze.main()
    S = json.load(open(tmp_path / "e7_summary.json"))
    assert S["gate"]["passes"] and S["E7_A_sensitivity"]["networks_researched"] == len(sens["networks"])
    fl = S["E7_A"]["floors"]
    assert S["E7_A"]["above_eps"] == sum(v > 0.02 for v in fl.values()) and len(fl) == 26
    assert S["E7_A"]["passes"] == (S["E7_A"]["above_eps"] >= 14)
    assert (tmp_path / "e7_parcel_map.csv").exists()
