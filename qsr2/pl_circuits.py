"""PennyLane implementation of the palindromic QAOA-MC proposal (the hardware path; reuses QSR v1's PennyLane stack).

Circuit from basis state |x>:  M(b1) P(g1) M(b2) P(g1) M(b1),  M(b) = prod_k RX_k(2b) = exp(-i b sum X),
P(g) = exp(-i g alpha E(Z)) with E = -sum J_ij Z_i Z_j - sum h_i Z_i, built from IsingZZ and RZ gates.
Bit convention: our basis index s has bit k on wire k (k = 0 least significant); PennyLane orders wire 0 as the
most significant bit, so outputs are re-indexed. VERIFY: probabilities equal |U[:, x]|^2 of proposals.qaoa_mc_unitary.
"""
import numpy as np


def _phase_layer(qml, J, h, theta):
    n = len(J)
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                qml.IsingZZ(-2 * theta * J[i, j], wires=[i, j])   # exp(+i theta J Z_i Z_j)
        if h is not None and h[i] != 0:
            qml.RZ(-2 * theta * h[i], wires=i)                    # exp(+i theta h Z_i)


def proposal_probs(J, h, x, alpha, b1, b2, g1, device="default.qubit"):
    import pennylane as qml
    n = len(J)
    dev = qml.device(device, wires=n)
    bits_x = [(x >> k) & 1 for k in range(n)]

    @qml.qnode(dev)
    def circ():
        qml.BasisState(np.array(bits_x), wires=range(n))
        for b, g in ((b1, g1), (b2, g1)):
            for k in range(n):
                qml.RX(2 * b, wires=k)
            _phase_layer(qml, J, h, g * alpha)
        for k in range(n):
            qml.RX(2 * b1, wires=k)
        return qml.probs(wires=range(n))

    p_pl = np.asarray(circ())
    idx = np.arange(2 ** n)
    pl_index = sum((((idx >> k) & 1) << (n - 1 - k)) for k in range(n))
    return p_pl[pl_index]
