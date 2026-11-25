"""Known-answer tests for exact cluster Monte Carlo matrices (G1b). Brute force = explicit bond enumeration."""
import itertools

import numpy as np
import pytest

from qsr2 import cluster, ising, mcmc

RNG = np.random.default_rng(11)


def rand_J(n):
    J = np.triu(RNG.standard_normal((n, n)), 1)
    return J + J.T


def brute(J, beta):
    n = len(J)
    N = 2 ** n
    Z = ising.spins(n)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    SW, WO = np.zeros((N, N)), np.zeros((N, N))
    for s in range(N):
        sat = [(i, j) for i, j in edges if J[i, j] * Z[s, i] * Z[s, j] > 0]
        p = [1 - np.exp(-2 * beta * abs(J[i, j])) for i, j in sat]
        for b in itertools.product([0, 1], repeat=len(sat)):
            pr = np.prod([pk if bk else 1 - pk for pk, bk in zip(p, b)])
            lab = list(range(n))
            def f(x):
                while lab[x] != x:
                    x = lab[x]
                return x
            for (i, j), bk in zip(sat, b):
                if bk:
                    lab[f(i)] = f(j)
            comp = {}
            for i in range(n):
                comp.setdefault(f(i), 0)
                comp[f(i)] |= 1 << i
            cl = list(comp.values())
            for flips in itertools.product([0, 1], repeat=len(cl)):
                A = sum(c for c, fl in zip(cl, flips) if fl)
                SW[s, s ^ A] += pr / 2 ** len(cl)
            for i in range(n):
                WO[s, s ^ comp[f(i)]] += pr / n
    return SW, WO


@pytest.mark.parametrize("n,beta", [(4, 0.7), (5, 1.3)])
def test_cluster_matrices_match_bond_enumeration(n, beta):
    J = rand_J(n)
    SW, WO = brute(J, beta)
    assert np.allclose(cluster.sw_transition(J, beta), SW, atol=1e-12)
    assert np.allclose(cluster.wolff_transition(J, beta), WO, atol=1e-12)


@pytest.mark.parametrize("which", ["sw", "wolff"])
def test_cluster_matrices_reversible_and_ergodic(which):
    n, beta = 8, 1.0
    J = rand_J(n)
    E = ising.energies(J)
    P = (cluster.sw_transition if which == "sw" else cluster.wolff_transition)(J, beta)
    gap, chk = mcmc.spectral_gap(P, ising.gibbs(E, beta), checks=True)
    assert 0 < gap <= 1 and max(chk.values()) < 1e-10


def test_ferromagnet_sw_beats_local_at_low_temperature():
    n, beta = 8, 1.0
    J = np.ones((n, n)) - np.eye(n)
    E = ising.energies(J)
    from qsr2 import proposals
    assert mcmc.spectral_gap(cluster.sw_transition(J, beta), ising.gibbs(E, beta)) > 100 * mcmc.gap_of(proposals.local(n), E, beta)


def test_houdayer_is_replica_exchange_on_dense_couplings():
    n = 9
    J = rand_J(n)
    for _ in range(50):
        z1, z2 = RNG.choice([-1, 1], n), RNG.choice([-1, 1], n)
        diff = np.where(z1 != z2)[0]
        if len(diff) == 0:
            continue
        C = cluster.houdayer_cluster(J, z1, z2, int(diff[0]))
        assert C == set(diff.tolist())
        a, b = z1.copy(), z2.copy()
        a[list(C)] *= -1; b[list(C)] *= -1
        assert np.array_equal(a, z2) and np.array_equal(b, z1)
    Js = J.copy(); Js[0, 1:] = Js[1:, 0] = 0                    # sparse case: cluster can be a strict subset
    z1, z2 = np.ones(n, int), -np.ones(n, int)
    assert cluster.houdayer_cluster(Js, z1, z2, 0) == {0}
