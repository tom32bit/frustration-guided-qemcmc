"""Corrected versions of the QSR v1 Phase-2 quantum pieces (audit P2-1 ... P2-4).

1. ising_from_v1_hamiltonian: the exact Ising form of the v1 binary Hamiltonian (Eqs. 9-11 as coded in
   Phase_2._construct_hamiltonian_binary): E(x) = H_stab + lam * H_mod with x = (1 - z)/2.
       H_stab = sum_{i<j} Hs_ij (x_i + x_j - 2 x_i x_j) = sum_{i<j} Hs_ij (1 - z_i z_j)/2
       H_mod  = -sum_{i != j} B_ij [x_i == x_j]       = -sum_{i<j} B_ij (1 + z_i z_j)
   =>  J_ij = Hs_ij/2 + lam B_ij,  h = 0,  const = sum_{i<j} (Hs_ij/2 - lam B_ij)      (v1 used wrong signs + fields)
2. qaoa_sample: exact state-vector QAOA with a UNITARY cost layer exp(-i g E) (v1 used exp(g H): not unitary),
   returning the measured bit-string of LOWEST TRUE ENERGY among the shots (v1 returned a random bit-string).
"""
import numpy as np

from .ising import energies, spins


def ising_from_v1_hamiltonian(W_patch, Hs_patch, d_global, m_global, lam=1.5):
    n = len(W_patch)
    B = W_patch - np.outer(d_global, d_global) / (2 * m_global)
    Hs = Hs_patch.copy()
    np.fill_diagonal(B, 0); np.fill_diagonal(Hs, 0)
    J = Hs / 2 + lam * B
    iu = np.triu_indices(n, 1)
    const = float(np.sum(Hs[iu] / 2 - lam * B[iu]))
    return J, np.zeros(n), const


def qaoa_sample(J, h, gammas, betas, shots=1000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n = len(J)
    E = energies(J, h)
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * E)
        for k in range(n):
            v = psi.reshape(2 ** (n - k - 1), 2, 2 ** k)
            a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :] = np.cos(b) * a0 - 1j * np.sin(b) * a1
            v[:, 1, :] = -1j * np.sin(b) * a0 + np.cos(b) * a1
            psi = v.reshape(-1)
    p = np.abs(psi) ** 2
    assert abs(p.sum() - 1) < 1e-9, "evolution must be unitary"
    samples = rng.choice(len(p), size=shots, p=p / p.sum())
    best = samples[np.argmin(E[samples])]
    return spins(n)[best].astype(int), float(E[best])
