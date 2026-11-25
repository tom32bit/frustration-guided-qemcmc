"""Known-answer tests for qsr2/ipr.py (E8: participation ratios and the Orfi-Sels bound)."""
import numpy as np

from qsr2 import ipr, ising, mcmc, proposals


def _net(n, seed):
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(n, n))
    J = np.triu(J, 1)
    return J + J.T


def test_identities():
    n, g = 6, 0.4
    J = _net(n, 1)
    E = ising.energies(J)
    iprs, Q = ipr.ipr_and_qinf(E, J, gammas=[g])
    M = 2 ** (n - 1)
    assert np.allclose(Q, Q.T, atol=1e-12)                       # symmetric proposal
    assert np.allclose(Q.sum(1), 1.0, atol=1e-12)                # rows sum to 1
    assert np.allclose(np.diag(Q), iprs[0], atol=1e-12)          # staying probability = IPR
    assert np.all(iprs >= 1.0 / M - 1e-12) and np.all(iprs <= 1.0 + 1e-12)


def test_gamma_zero_is_localised():
    n = 5
    J = _net(n, 2)
    E = ising.energies(J)
    assert len(np.unique(np.round(E[: 2 ** (n - 1)], 10))) == 2 ** (n - 1)   # distinct partition energies
    iprs, Q = ipr.ipr_and_qinf(E, J, gammas=[0.0])
    assert np.allclose(iprs, 1.0) and np.allclose(Q, np.eye(2 ** (n - 1)))


def test_matches_full_space():
    """Sector algebra: the lumped long-time proposal from the full 2^n eigenbasis equals the sector result, and the
    labelled-state IPR is half the partition-space IPR."""
    n, g = 5, 0.45
    J = _net(n, 3)
    E = ising.energies(J)
    N, M = 2 ** n, 2 ** (n - 1)
    a = proposals.layden_alpha(J, np.zeros(n))
    H = g * proposals.mixer(n)
    H[np.diag_indices_from(H)] += (1 - g) * a * E
    w, V = np.linalg.eigh(H)
    assert np.min(np.diff(w)) > 1e-8                             # non-degenerate: dephased formula applies
    W = V * V
    Qfull = W @ W.T
    r = np.arange(M)
    Qlump = Qfull[np.ix_(r, r)] + Qfull[np.ix_(r, r ^ (N - 1))]
    iprs, Q = ipr.ipr_and_qinf(E, J, gammas=[g])
    assert np.allclose(Q, Qlump, atol=1e-10)
    assert np.allclose(iprs[0], 2 * (W[:M] ** 2).sum(1), atol=1e-10)


def test_long_time_average():
    """The dephased proposal is the long-time average of the finite-time quench (Orfi & Sels, Eq. 11)."""
    n, g = 4, 0.5
    J = _net(n, 4)
    E = ising.energies(J)
    _, Qinf = ipr.ipr_and_qinf(E, J, gammas=[g])
    Qt, _ = proposals.quench_mean_sectors(E, J, gammas=[g], times=np.linspace(0.0, 4000.0, 20001), full=False)
    assert np.max(np.abs(Qt - Qinf)) < 0.01


def test_bounds_order_and_bruteforce():
    n = 6
    J = _net(n, 5)
    E = ising.energies(J)
    M = 2 ** (n - 1)
    Er = E[:M]
    iprs, Q = ipr.ipr_and_qinf(E, J)
    beta = 1.0 / np.std(J[np.triu_indices(n, 1)])
    gap = ipr.chain_gap(Q, Er, beta)
    lam, ipb, _, _ = ipr.threshold_bounds(Q, Er, beta, iprs)
    assert gap <= lam * (1 + 1e-9) and lam <= ipb * (1 + 1e-9)
    # brute force over the same energy-level cuts
    P = mcmc.mh_matrix(Q, Er, beta)
    pi = np.exp(ipr.gibbs_log(Er, beta))
    levels = np.unique(Er)
    best_lam, best_ipb = np.inf, np.inf
    for lv in levels[:-1]:
        Bc = Er <= lv
        B = ~Bc
        flow = (pi[B][:, None] * P[np.ix_(B, Bc)]).sum()
        best_lam = min(best_lam, flow / (pi[B].sum() * pi[Bc].sum()))
        pibar = pi[Bc].sum() / Bc.sum()
        vals = [np.sqrt((pi[B] * iprs[i][B]).sum() / pi[B].sum() * iprs[i][Bc].mean()) / pibar
                for i in range(len(iprs))]
        best_ipb = min(best_ipb, np.mean(vals))
    assert np.isclose(lam, best_lam, rtol=1e-9) and np.isclose(ipb, best_ipb, rtol=1e-9)


def _net_with_isolated(n, seed, isolated=2):
    """A random network whose last `isolated` regions have no couplings to any other region. Their symmetry makes some
    levels of H(g) degenerate within a flip sector, as in 30 structural sub-networks of E8 (amendment 1)."""
    J = _net(n, seed)
    J[-isolated:, :] = 0.0
    J[:, -isolated:] = 0.0
    return J


def test_degenerate_levels_long_time_average():
    """With degenerate levels the projector form is still the long-time average of the finite-time quench."""
    n, g = 5, 0.5
    J = _net_with_isolated(n, 6)
    E = ising.energies(J)
    M = 2 ** (n - 1)
    a = proposals.layden_alpha(J, np.zeros(n))
    spacing = []
    for X in proposals._sector_mixers(n):
        H = g * X
        H[np.diag_indices_from(H)] += (1 - g) * a * E[:M]
        spacing.append(np.min(np.diff(np.linalg.eigvalsh(H))))
    assert min(spacing) < 1e-10                                     # a sector has a degenerate level
    iprs, Qinf = ipr.ipr_and_qinf(E, J, gammas=[g])
    assert np.allclose(Qinf, Qinf.T, atol=1e-12) and np.allclose(Qinf.sum(1), 1.0, atol=1e-12)
    assert np.allclose(np.diag(Qinf), iprs[0], atol=1e-12)
    Qt, _ = proposals.quench_mean_sectors(E, J, gammas=[g], times=np.linspace(0.0, 4000.0, 20001), full=False)
    assert np.max(np.abs(Qt - Qinf)) < 0.01
