"""Qiskit implementation of the palindromic QAOA-MC proposal (QSR v1's depth-2 circuit), the IBM hardware path.

From basis state |x>:  M(b1) P(g1) M(b2) P(g1) M(b1) |x>,  M(b) = prod_k RX_k(2b) = exp(-i b sum X),
P(g) = exp(-i g alpha E(Z)),  E = -sum_{i<j} J_ij Z_i Z_j  ->  prod_{i<j} RZZ(-2 g alpha J_ij)  (field-free).
Bit convention: our basis index has bit k on qubit k; Qiskit is little-endian (qubit k = bit k of the integer
outcome), so no re-indexing is needed. VERIFY (tests): statevector probabilities equal |U[:, x]|^2 of
proposals.qaoa_mc_unitary for every x.
Flip twirl (optional): with a uniformly random global flip F applied to the input and undone on the output,
the hardware proposal is exactly F-symmetric, so the partition-space reduction stays exact under noise.
"""
import numpy as np


def qaoa_mc_circuit(J, x, alpha, b1, b2, g1, measure=True, flip=False):
    from qiskit import QuantumCircuit
    n = len(J)
    qc = QuantumCircuit(n, n if measure else 0)
    bits = x ^ ((1 << n) - 1) if flip else x
    for k in range(n):
        if (bits >> k) & 1:
            qc.x(k)

    def mixer(b):
        for k in range(n):
            qc.rx(2 * b, k)

    def phase(g):
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0:
                    qc.rzz(-2 * g * alpha * J[i, j], i, j)

    mixer(b1); phase(g1); mixer(b2); phase(g1); mixer(b1)
    if flip:                       # undo the global flip on the output (X before measurement)
        for k in range(n):
            qc.x(k)
    if measure:
        qc.measure(range(n), range(n))
    return qc


def statevector_probs(J, x, alpha, b1, b2, g1, flip=False):
    from qiskit.quantum_info import Statevector
    qc = qaoa_mc_circuit(J, x, alpha, b1, b2, g1, measure=False, flip=flip)
    return np.abs(Statevector(qc).data) ** 2


def counts_to_probs(counts, n):
    p = np.zeros(2 ** n)
    for bitstr, c in counts.items():
        p[int(bitstr, 2)] += c
    return p / p.sum()
