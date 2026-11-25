"""E3c partition-space code path (qsr2/exact_ps.py, cluster_numba.sw_wolff_lumped) against the validated full-space
code path used for G1c and E3 Tier A (qsr2/exact.py, cluster.sw_wolff_transitions)."""
import numpy as np
import pytest

from qsr2 import cluster, exact, exact_ps, ising, mcmc, proposals

RNG = np.random.default_rng(2026)


def rand_J(n):
    J = np.triu(RNG.standard_normal((n, n)), 1)
    return J + J.T


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("beta", [0.4, 1.3])
def test_cluster_lumped_equals_lumped_full(n, beta):
    J = rand_J(n)
    SW, WO = cluster.sw_wolff_transitions(J, beta)
    SWb, WOb = exact_ps.cluster_lumped(J, beta)
    assert np.max(np.abs(SWb - mcmc.lump(SW, n))) < 1e-12
    assert np.max(np.abs(WOb - mcmc.lump(WO, n))) < 1e-12
    assert np.allclose(SWb.sum(1), 1) and np.allclose(WOb.sum(1), 1)


@pytest.mark.parametrize("n,block", [(5, 3), (6, 32), (7, 5)])
def test_qaoa_lumped_blocks(n, block):
    J = rand_J(n)
    E = ising.energies(J)
    a = proposals.layden_alpha(J, np.zeros(n))
    Q = proposals.qaoa_mc_proposal(E, a, *proposals.QAOA_GRID[0])
    N, M = 2 ** n, 2 ** (n - 1)
    r = np.arange(M)
    Qb = Q[np.ix_(r, r)] + Q[np.ix_(r, r ^ (N - 1))]
    assert np.max(np.abs(exact_ps.qaoa_lumped(E, a, *proposals.QAOA_GRID[0], block=block) - Qb)) < 1e-13


def test_local_lumped():
    n = 6
    assert np.max(np.abs(exact_ps.lumped_local(n) - mcmc.lump(proposals.local(n), n))) < 1e-15


@pytest.mark.parametrize("n", [5, 6, 7])
def test_compute_ps_equals_compute_exact(n):
    J = rand_J(n)
    sigma = float(np.std(J[np.triu_indices(n, 1)]))
    ref = exact.compute_exact(J, sigma, betas_tilde=(1.0, 2.0), full=False, asym=True, pt=False, qaoa_angles=True)
    new = exact_ps.compute_ps(J, sigma, betas_tilde=(1.0, 2.0))
    keys = [f"b{bt}_{c}_{s}" for bt in (1.0, 2.0) for c in ("local", "label_swap", "uniform", "quantum", "sw", "wolff")
            for s in ("lumped", "s2mean", "s2max")] + ["b1.0_varmean", "b2.0_varmean", "b1.0_qaoa_a0_lumped"]
    for k in keys:
        assert abs(new[k] - ref[k]) <= 1e-9 * max(abs(ref[k]), 1e-12), k
