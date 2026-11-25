"""Known-answer tests (plan E1). Run from the repository root:  make test   (or: python3 -m pytest -q)"""
import os
import sys

import numpy as np
import pytest

from qsr2 import ising, legacy, mcmc, networks, patches, proposals

RNG = np.random.default_rng(0)


def rand_J(n, rng=RNG):
    J = rng.standard_normal((n, n)); J = np.triu(J, 1)
    return J + J.T


# ---------- Ising / frustration ----------
def test_z2_symmetry_without_field():
    E = ising.energies(rand_J(6))
    idx = np.arange(64)
    assert np.allclose(E, E[idx ^ 63])


def test_frustration_known_values():
    ferro = np.ones((5, 5)) - np.eye(5)
    assert patches.frustration(ferro) == 0.0
    af_triangle = -(np.ones((3, 3)) - np.eye(3))
    assert np.isclose(patches.frustration(af_triangle), 1 / 3)


# ---------- proposals ----------
@pytest.mark.parametrize("n", [3, 5, 7])
def test_proposals_stochastic_and_quantum_symmetric(n):
    J, h = rand_J(n), RNG.standard_normal(n)
    E = ising.energies(J, h)
    Qs = [proposals.local(n), proposals.label_swap(n), proposals.uniform(n),
          sum(proposals.quench_by_gamma(E, J, h)) / 6, proposals.qaoa_mc(E, J, h)]
    for Q in Qs:
        assert np.allclose(Q.sum(1), 1, atol=1e-12)
        assert np.allclose(Q, Q.T, atol=1e-12)          # Proposition 4 for the quantum ones


def test_qaoa_unitary_is_unitary_and_symmetric():
    J, h = rand_J(4), RNG.standard_normal(4)
    E = ising.energies(J, h)
    U = proposals.qaoa_mc_unitary(E, 0.7, 0.3, 0.9, 1.1)
    assert np.allclose(U @ U.conj().T, np.eye(16), atol=1e-12)
    assert np.allclose(U, U.T, atol=1e-12)


# ---------- MCMC ----------
def test_mh_is_reversible_and_stationary():
    n = 6
    J, h = rand_J(n), RNG.standard_normal(n)
    E = ising.energies(J, h)
    Q = sum(proposals.quench_by_gamma(E, J, h)) / 6
    gap, chk = mcmc.spectral_gap(mcmc.mh_matrix(Q, E, 1.0), ising.gibbs(E, 1.0), checks=True)
    assert 0 < gap <= 1 and max(chk.values()) < 1e-12


def test_heat_bath_gap_is_one():
    n = 5
    J = rand_J(n); E = ising.energies(J); pi = ising.gibbs(E, 1.0)
    assert np.isclose(mcmc.spectral_gap(proposals.heat_bath_transition(pi), pi), 1.0, atol=1e-10)


def test_pt_sparse_gap_matches_dense():
    n = 3
    J = rand_J(n); E = ising.energies(J)
    betas = [1.5, 0.5]
    P, logpi = mcmc.pt_matrix(E, n, betas)
    Pd = P.toarray()
    pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
    assert np.allclose(Pd.sum(1), 1) and np.allclose(pi @ Pd, pi, atol=1e-12)
    dense = mcmc.spectral_gap(Pd, pi)
    assert np.isclose(mcmc.pt_gap(E, n, betas), dense, atol=1e-8)


def test_layden_sk_known_answer_small():
    """Quantum quench proposals beat local/uniform on SK at T=1 (Layden et al. 2023), n = 4..8, 15 instances."""
    rng = np.random.default_rng(1)
    ks = {}
    for name in ("quantum", "local", "uniform"):
        means = []
        for n in range(4, 9):
            g = []
            for _ in range(15):
                J, h = rand_J(n, rng), rng.standard_normal(n)
                E = ising.energies(J, h)
                Q = {"quantum": lambda: sum(proposals.quench_by_gamma(E, J, h)) / 6,
                     "local": lambda: proposals.local(n), "uniform": lambda: proposals.uniform(n)}[name]()
                g.append(mcmc.gap_of(Q, E, 1.0))
            means.append(np.mean(g))
        ks[name] = -np.polyfit(range(4, 9), np.log2(means), 1)[0]
    assert ks["quantum"] < 0.5 < min(ks["local"], ks["uniform"])


# ---------- exact patch conditionals ----------
def _full_energy(B, lab):
    return -np.sum(B * (lab[:, None] == lab[None, :]))


def test_split_and_border_propositions_on_real_connectome():
    W, _ = networks.load_connectome("FC100")
    c = networks.best_partition(W, 2.0, runs=5)
    B = networks.modularity_matrix(W, 2.0)
    m = next(m for m in np.unique(c) if 4 <= np.sum(c == m) <= 10)
    M = np.where(c == m)[0]
    J, h = patches.split_patch(W, M, 2.0)
    new = c.max() + 1
    Z = ising.spins(len(M))
    diffs = []
    for z in Z:
        lab = c.copy(); lab[M] = np.where(z == 1, m, new)
        diffs.append(_full_energy(B, lab) - (-(np.triu(J, 1) * np.outer(z, z)).sum()))
    assert np.ptp(diffs) < 1e-9                      # differs from the Ising energy only by a constant
    # border: two adjacent communities a, b at gamma = 1
    c1 = networks.best_partition(W, 1.0, runs=5)
    a, b = np.unique(c1)[:2]
    U = np.where((c1 == a) | (c1 == b))[0][:6]
    J, h = patches.border_patch(W, c1, U, a, b, 1.0)
    B1 = networks.modularity_matrix(W, 1.0)
    diffs = []
    for z in ising.spins(6):
        lab = c1.copy(); lab[U] = np.where(z == 1, a, b)
        e_ising = -(np.triu(J, 1) * np.outer(z, z)).sum() - h @ z
        diffs.append(_full_energy(B1, lab) - e_ising)
    assert np.ptp(diffs) < 1e-9


# ---------- fixed QSR v1 pieces ----------
def test_weighted_leiden_recovers_planted_partition_where_v1_failed():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures", "qsr_v1"))
    import Phase_1 as P1
    from sklearn.metrics import normalized_mutual_info_score as nmi
    W, truth = P1.generate_brain_like_network(n_nodes=80, n_communities=4, inter_prob=0.4, intra_prob=0.5,
                                              noise_level=0.03, random_state=1000)
    np.fill_diagonal(W, 0)
    assert nmi(networks.leiden(W, 1.0, 0), truth) > 0.95


def test_v1_hamiltonian_ising_mapping_matches_v1_code():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures", "qsr_v1"))
    import Phase_2 as P2
    n = 6
    rng = np.random.default_rng(3)
    Wp = rng.random((n, n)); Wp = (Wp + Wp.T) / 2
    Hs = rng.random((n, n)); Hs = (Hs + Hs.T) / 2; np.fill_diagonal(Hs, 0)
    d = rng.random(n) * 5 + 1; m = 20.0
    q = P2.QuantumEnhancedQSR(lambda_mod=1.5)
    v1 = q._construct_hamiltonian_binary(Wp, Hs, d, m)
    J, h, const = legacy.ising_from_v1_hamiltonian(Wp, Hs, d, m, lam=1.5)
    E = ising.energies(J, h) + const
    X = (1 - ising.spins(n)) // 2
    assert np.allclose([v1(x) for x in X], E, atol=1e-10)


def test_qaoa_sampler_returns_low_energy_state():
    J = rand_J(6); h = np.zeros(6)
    z, e = legacy.qaoa_sample(J, h, [0.4, 0.6], [0.6, 0.3], shots=2000, rng=np.random.default_rng(0))
    assert np.isclose(e, ising.energies(J, h).min())     # 2000 shots on 64 states find the optimum


def test_pennylane_circuit_matches_exact_matrix():
    pytest.importorskip("pennylane")
    from qsr2 import pl_circuits
    n = 4
    J, h = rand_J(n), RNG.standard_normal(n)
    E = ising.energies(J, h)
    a = proposals.layden_alpha(J, h)
    U = proposals.qaoa_mc_unitary(E, a, 0.4, 0.9, 1.3)
    for x in (0, 5, 11):
        assert np.allclose(pl_circuits.proposal_probs(J, h, x, a, 0.4, 0.9, 1.3), np.abs(U[:, x]) ** 2, atol=1e-10)
