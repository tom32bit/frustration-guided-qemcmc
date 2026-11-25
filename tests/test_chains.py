"""E3 Tier B simulators must reproduce the exact one-step kernels used in G1/G1b (n = 4, fixed start)."""
import numpy as np
import pytest

from qsr2 import chains, cluster, ising, mcmc, proposals

RNG = np.random.default_rng(5)
n, beta, C, s0 = 4, 0.8, 40000, 6


def J4():
    J = np.triu(np.random.default_rng(3).standard_normal((n, n)), 1)
    return J + J.T


def check(states, exact_row):
    freq = np.bincount(states, minlength=2 ** n) / len(states)
    se = np.sqrt(exact_row * (1 - exact_row) / len(states)) + 1e-3
    assert np.all(np.abs(freq - exact_row) < 5 * se), (freq, exact_row)


@pytest.mark.parametrize("kind", ["quench", "local", "label_swap", "sw", "wolff"])
def test_one_step_kernel_matches_exact(kind):
    J = J4(); E = ising.energies(J)
    start = np.full(C, s0)
    rng = np.random.default_rng(1)
    if kind == "quench":
        out = chains.QuenchSampler(J, E, beta).step(start, rng)
        P = mcmc.mh_matrix(sum(proposals.quench_by_gamma(E, J, np.zeros(n))) / 6, E, beta)
    elif kind == "local":
        out = chains.local_step(start, n, E, beta, rng); P = mcmc.mh_matrix(proposals.local(n), E, beta)
    elif kind == "label_swap":
        out = chains.label_swap_step(start, n, E, beta, rng); P = mcmc.mh_matrix(proposals.label_swap(n), E, beta)
    elif kind == "sw":
        out = chains.sw_step(start, n, J, beta, rng); P = cluster.sw_transition(J, beta)
    else:
        out = chains.wolff_step(start, n, J, beta, rng); P = cluster.wolff_transition(J, beta)
    check(out, P[s0])


def test_pt_step_matches_exact_matrix():
    J = J4(); E = ising.energies(J); betas = [beta, beta / 3]
    P, _ = mcmc.pt_matrix(E, n, betas)
    s = s0 + 16 * 9                                    # joint index x0 + N x1
    reps = np.tile([s0, 9], (C, 1))
    out = chains.pt_step(reps, n, E, betas, np.random.default_rng(2))
    joint = out[:, 0] + 16 * out[:, 1]
    freq = np.bincount(joint, minlength=256) / C
    row = P[s].toarray().ravel()
    assert np.all(np.abs(freq - row) < 5 * np.sqrt(row * (1 - row) / C) + 1e-3)


def test_running_corr_converges_for_exact_samples():
    J = J4(); E = ising.energies(J)
    ex = chains.exact_corr(J, E, beta)
    pi = ising.gibbs(E, beta)
    rc = chains.RunningCorr(8, n, ex)
    rng = np.random.default_rng(4)
    for _ in range(4000):
        rc.add(rng.choice(16, 8, p=pi))
    assert rc.rmse() < 0.03


def test_chebyshev_propagator_equals_expm_multiply():
    from scipy.sparse.linalg import expm_multiply
    n = 8
    J = np.triu(np.random.default_rng(9).standard_normal((n, n)), 1); J = J + J.T
    E = ising.energies(J)
    q, c = chains.QuenchSampler(J, E, 1.0), chains.ChebQuenchSampler(J, E, 1.0)
    st = np.array([0, 37, 200])
    for gi in (0, 5):
        B = np.zeros((2 ** n, 3), complex); B[st, np.arange(3)] = 1
        ref = expm_multiply(-1j * q.H[gi], B, start=q.times[0], stop=q.times[-1], num=len(q.times), endpoint=True)
        assert np.max(np.abs(c.evolve(gi, st) - ref)) < 1e-10


def test_chebyshev_one_step_kernel():
    J = J4(); E = ising.energies(J)
    out = chains.ChebQuenchSampler(J, E, beta).step(np.full(C, s0), np.random.default_rng(1))
    P = mcmc.mh_matrix(sum(proposals.quench_by_gamma(E, J, np.zeros(n))) / 6, E, beta)
    check(out, P[s0])


def test_chebyshev_single_time_and_complex64_accuracy():
    n = 8
    J = np.triu(np.random.default_rng(10).standard_normal((n, n)), 1); J = J + J.T
    E = ising.energies(J)
    c128, c64 = chains.ChebQuenchSampler(J, E, 1.0), chains.ChebQuenchSampler(J, E, 1.0, dtype=np.complex64)
    st = np.array([3, 77])
    for gi in (0, 3, 5):
        allt = c128.evolve(gi, st)
        for ti in range(6):
            one = c128.evolve(gi, st, ti=ti)
            assert np.max(np.abs(one - allt[ti])) < 1e-11
            p64 = np.abs(c64.evolve(gi, st, ti=ti)) ** 2
            assert np.max(np.abs(p64 - np.abs(allt[ti]) ** 2)) < 2e-5


def test_numba_kernel_equals_numpy_path():
    pytest.importorskip("numba")
    n = 9
    J = np.triu(np.random.default_rng(12).standard_normal((n, n)), 1); J = J + J.T
    E = ising.energies(J)
    a, b = chains.ChebQuenchSampler(J, E, 1.0, use_numba=True), chains.ChebQuenchSampler(J, E, 1.0, use_numba=False)
    assert a.kernel is not None and b.kernel is None
    st = np.array([0, 5, 300])
    for gi in range(6):
        for ti in range(6):
            assert np.max(np.abs(a.evolve(gi, st, ti=ti) - b.evolve(gi, st, ti=ti))) < 1e-11
