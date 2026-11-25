"""The Qiskit hardware circuit must reproduce the exact QAOA-MC proposal matrix (all start states)."""
import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qsr2 import ising, proposals, qiskit_circuits  # noqa: E402


@pytest.mark.parametrize("flip", [False, True])
def test_qiskit_rows_equal_exact_unitary(flip):
    n = 5
    rng = np.random.default_rng(8)
    J = np.triu(rng.standard_normal((n, n)), 1); J = J + J.T
    E = ising.energies(J)
    a = proposals.layden_alpha(J, np.zeros(n))
    b1, b2, g1 = proposals.QAOA_GRID[7]
    Q = np.abs(proposals.qaoa_mc_unitary(E, a, b1, b2, g1)) ** 2
    for x in range(2 ** n):
        p = qiskit_circuits.statevector_probs(J, x, a, b1, b2, g1, flip=flip)
        assert np.allclose(p, Q[:, x], atol=1e-12)


def test_counts_parsing_is_little_endian():
    p = qiskit_circuits.counts_to_probs({"001": 3, "100": 1}, 3)
    assert p[1] == 0.75 and p[4] == 0.25
