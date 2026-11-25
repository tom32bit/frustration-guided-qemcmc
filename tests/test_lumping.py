"""Partition-space (lumped) chains: the exact reduction of field-free chains onto bipartitions {x, NOT x}."""
import numpy as np
import pytest

from qsr2 import cluster, ising, mcmc, proposals

RNG = np.random.default_rng(21)


def rand_J(n):
    J = np.triu(RNG.standard_normal((n, n)), 1)
    return J + J.T


def kernels(J, beta):
    n = len(J); E = ising.energies(J); z = np.zeros(n)
    SW, WO = cluster.sw_wolff_transitions(J, beta)
    return E, {"quantum": mcmc.mh_matrix(sum(proposals.quench_by_gamma(E, J, z)) / 6, E, beta),
               "qaoa": mcmc.mh_matrix(proposals.qaoa_mc(E, J, z), E, beta),
               "local": mcmc.mh_matrix(proposals.local(n), E, beta),
               "label_swap": mcmc.mh_matrix(proposals.label_swap(n), E, beta),
               "uniform": mcmc.mh_matrix(proposals.uniform(n), E, beta), "sw": SW, "wolff": WO}


def sym_eigs(P, logpi):
    S = np.exp(np.log(np.where(P > 0, P, 1)) + 0.5 * (logpi[:, None] - logpi[None, :])) * (P > 0)
    return np.sort(np.linalg.eigvalsh((S + S.T) / 2))


@pytest.mark.parametrize("name", ["quantum", "qaoa", "local", "label_swap", "uniform", "sw", "wolff"])
def test_lumped_spectrum_is_even_part_of_full(name):
    n, beta = 6, 1.2
    J = rand_J(n)
    E, K = kernels(J, beta)
    P = K[name]
    assert mcmc.flip_symmetry_error(P, n) < 1e-12
    L = mcmc.lump(P, n)
    lp = -beta * (E - E.min())
    assert np.allclose(L.sum(1), 1, atol=1e-12)
    pib = np.exp(lp[:32]); pib /= pib.sum()
    assert np.max(np.abs(pib @ L - pib)) < 1e-12
    full, lum = sym_eigs(P, lp), sym_eigs(L, lp[:32])
    assert all(np.min(np.abs(full - x)) < 1e-9 for x in lum)          # lumped spectrum within the full spectrum
    assert mcmc.lumped_gap(P, E, beta, n) >= mcmc.spectral_gap(P, logpi=lp) - 1e-12


def test_lumping_removes_label_flip_mode_of_local_moves():
    n, beta = 8, 1.0
    J = np.ones((n, n)) - np.eye(n)                                    # ferromagnet: the slow mode IS the label flip
    E = ising.energies(J)
    P = mcmc.mh_matrix(proposals.local(n), E, beta)
    assert mcmc.lumped_gap(P, E, beta, n) > 1e3 * mcmc.spectral_gap(P, logpi=-beta * (E - E.min()))


def test_lumped_parallel_tempering_spectrum():
    n, betas = 4, [1.3, 0.4]
    E = ising.energies(rand_J(n))
    Pf, lpf = mcmc.pt_matrix(E, n, betas)
    Pl, lpl = mcmc.pt_matrix(E, n, betas, lumped=True)
    Pf, Pl = Pf.toarray(), Pl.toarray()
    assert np.allclose(Pl.sum(1), 1, atol=1e-12)
    full, lum = sym_eigs(Pf, lpf), sym_eigs(Pl, lpl)
    assert all(np.min(np.abs(full - x)) < 1e-9 for x in lum)
    g = mcmc.pt_gap(E, n, betas, lumped=True)
    assert np.isclose(g, 1 - max(abs(lum[-2]), abs(lum[0])), atol=1e-8)


def test_asymptotic_variance_matches_spectral_formula_and_iid():
    n, beta = 5, 0.9
    J = rand_J(n)
    E, K = kernels(J, beta)
    lp = -beta * (E[:16] - E.min())
    F = mcmc.coassignment_observables(n)
    L = mcmc.lump(K["quantum"], n)
    s2, var = mcmc.asymptotic_variances(L, lp, F)
    pi = np.exp(lp - lp.max()); pi /= pi.sum()
    S = np.sqrt(pi)[:, None] * L / np.sqrt(pi)[None, :]
    lam, V = np.linalg.eigh((S + S.T) / 2)
    fc = F - pi @ F
    coef = V.T @ (np.sqrt(pi)[:, None] * fc)                           # <f, phi_k>_pi
    keep = lam < 1 - 1e-10
    s2_spec = np.sum(((1 + lam[keep]) / (1 - lam[keep]))[:, None] * coef[keep] ** 2, axis=0)
    assert np.allclose(s2, s2_spec, rtol=1e-8)
    s2_iid, var_iid = mcmc.asymptotic_variances(np.outer(np.ones(16), pi), lp, F)
    assert np.allclose(s2_iid, var_iid, rtol=1e-10)


def test_quench_mean_equals_grid_average():
    n = 6
    J = rand_J(n); E = ising.energies(J); z = np.zeros(n)
    assert np.allclose(proposals.quench_mean(E, J), sum(proposals.quench_by_gamma(E, J, z)) / 6, atol=1e-12)


def test_quench_sectors_equal_full_quench_and_its_lumping():
    for n in (5, 7):
        J = rand_J(n); E = ising.energies(J)
        Qf = proposals.quench_mean(E, J)
        Qbar, Q = proposals.quench_mean_sectors(E, J)
        assert np.allclose(Q, Qf, atol=1e-12)
        r = np.arange(2 ** (n - 1))
        assert np.allclose(Qbar, Qf[np.ix_(r, r)] + Qf[np.ix_(r, r ^ (2 ** n - 1))], atol=1e-12)


@pytest.mark.parametrize("name", ["quantum", "local", "label_swap", "uniform", "sw", "wolff"])
def test_sector_gaps_equal_direct_gaps(name):
    n, beta = 6, 1.1
    J = rand_J(n)
    E, K = kernels(J, beta)
    lum, full = mcmc.sector_gaps(K[name], E, beta, n)
    assert np.isclose(lum, mcmc.lumped_gap(K[name], E, beta, n), atol=1e-10)
    assert np.isclose(full, mcmc.spectral_gap(K[name], logpi=-beta * (E - E.min())), atol=1e-10)


def test_numba_cluster_tables_equal_numpy():
    pytest.importorskip("numba")
    J = rand_J(7)
    for beta in (0.4, 1.5):
        a = cluster.sw_wolff_transitions(J, beta, use_numba=True)
        b = cluster.sw_wolff_transitions(J, beta, use_numba=False)
        assert np.allclose(a[0], b[0], atol=1e-13) and np.allclose(a[1], b[1], atol=1e-13)


def test_exact_row_matches_direct_definitions():
    from qsr2 import exact
    n = 6
    J = rand_J(n); E = ising.energies(J); sigma = float(np.std(J[np.triu_indices(n, 1)]))
    row = exact.compute_exact(J, sigma, betas_tilde=(1.0,), pt=False)
    beta = 1.0 / sigma
    _, K = kernels(J, beta)
    for name in ("quantum", "qaoa", "local", "label_swap", "uniform", "sw", "wolff"):
        assert np.isclose(row[f"b1.0_{name}_lumped"], mcmc.lumped_gap(K[name], E, beta, n), atol=1e-10)
        assert np.isclose(row[f"b1.0_{name}_full"], mcmc.spectral_gap(K[name], logpi=-beta * (E - E.min())), atol=1e-10)
    a0 = proposals.QAOA_GRID[0]
    Q0 = np.abs(proposals.qaoa_mc_unitary(E, proposals.layden_alpha(J, np.zeros(n)), *a0)) ** 2
    assert np.isclose(row["b1.0_qaoa_a0_lumped"], mcmc.lumped_gap(mcmc.mh_matrix(Q0, E, beta), E, beta, n), atol=1e-10)


def test_qaoa_half_column_proposal_equals_full_unitary():
    n = 7
    J = rand_J(n); E = ising.energies(J); a = proposals.layden_alpha(J, np.zeros(n))
    for ang in (proposals.QAOA_GRID[0], proposals.QAOA_GRID[11]):
        Qfull = (np.abs(proposals.qaoa_mc_unitary(E, a, *ang)) ** 2).T
        assert np.allclose(proposals.qaoa_mc_proposal(E, a, *ang), Qfull, atol=1e-13)
