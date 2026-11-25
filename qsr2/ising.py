"""Ising utilities. Convention: z_i in {+1, -1}; basis index s has bit k = (s >> k) & 1 and z_k = 1 - 2*bit.

E(z) = - sum_{i<j} J_ij z_i z_j - sum_i h_i z_i        (J symmetric, zero diagonal)
"""
import numpy as np


def spins(n: int) -> np.ndarray:
    """All 2^n spin configurations as a (2^n, n) array of +-1 (row s <-> basis state |s>)."""
    idx = np.arange(2 ** n, dtype=np.int64)
    return (1 - 2 * ((idx[:, None] >> np.arange(n)[None, :]) & 1)).astype(np.int8)


def energies(J: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
    """Exact energies of all 2^n states. VERIFY: E(z) == E(-z) when h == 0 (Z2 symmetry)."""
    n = len(J)
    Z = spins(n).astype(float)
    Ju = np.triu(J, 1)
    E = -np.einsum("si,ij,sj->s", Z, Ju, Z, optimize=True)
    if h is not None:
        E -= Z @ h
    return E


def gibbs(E: np.ndarray, beta: float) -> np.ndarray:
    """Normalised Boltzmann weights pi(s) ∝ exp(-beta E(s))."""
    w = np.exp(-beta * (E - E.min()))
    return w / w.sum()


def ground_state(J: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
    """Exact ground-state spin vector (first minimiser) by enumeration (n <= ~24)."""
    E = energies(J, h)
    return spins(len(J))[int(np.argmin(E))].astype(int)
