"""Exact proposal matrices Q[x, x'] = probability to propose x' from x (rows sum to 1), n <= ~12.

Classical : local (flip one random spin), label-swap mixture (1/2 local, 1/2 flip all), uniform,
            block heat-bath (independence sampler drawing from the exact conditional; cost 2^n per step)
Quantum   : quench (Layden et al., Nature 619:282, 2023): |<x'| exp(-i t [(1-g) a H_C + g sum X]) |x>|^2
            palindromic QAOA-MC (Nakano et al., PRR 6:033105, 2024): U = M(b1) P(g1) M(b2) P(g1) M(b1)
Both quantum proposals are symmetric (U^T = U; Proposition 4) so plain Metropolis acceptance is exact.
"""
import numpy as np

LAYDEN_GAMMAS = np.linspace(0.25, 0.6, 6)
LAYDEN_TIMES = np.linspace(2.0, 20.0, 6)


def mixer(n: int) -> np.ndarray:
    """H_mix = sum_k X_k as a dense 2^n x 2^n matrix."""
    N = 2 ** n
    M = np.zeros((N, N))
    idx = np.arange(N)
    for k in range(n):
        M[idx, idx ^ (1 << k)] = 1.0
    return M


def local(n):
    return mixer(n) / n


def label_swap(n):
    N = 2 ** n
    F = np.zeros((N, N)); idx = np.arange(N); F[idx, idx ^ (N - 1)] = 1.0
    return 0.5 * local(n) + 0.5 * F


def uniform(n):
    N = 2 ** n
    return np.full((N, N), 1.0 / N)


def heat_bath_transition(pi):
    """TRANSITION matrix (not a proposal) of exact block heat-bath: draw the whole patch from its exact conditional.
    P[x, :] = pi. Gap = 1 per step, but each step costs O(2^n) energy evaluations, so compare on a cost-normalised
    basis (gap / 2^n). Do NOT pass this to mcmc.mh_matrix (the independence proposal is not symmetric)."""
    return np.tile(pi, (len(pi), 1))


def layden_alpha(J, h):
    n = len(J)
    return np.sqrt(n) / np.sqrt(np.sum(np.triu(J, 1) ** 2) + np.sum(np.asarray(h) ** 2))


def quench_by_gamma(E, J, h, gammas=LAYDEN_GAMMAS, times=LAYDEN_TIMES):
    """List of t-averaged quench proposal matrices, one per gamma. Mean over the list = Layden's random (g, t)."""
    n = len(J)
    Hm = mixer(n)
    a = layden_alpha(J, h)
    out = []
    for g in gammas:
        w, V = np.linalg.eigh((1 - g) * a * np.diag(E) + g * Hm)
        Q = np.zeros_like(Hm)
        for t in times:
            Q += np.abs((V * np.exp(-1j * w * t)) @ V.conj().T) ** 2
        out.append(Q / len(times))
    return out


def quench_mean(E, J, h=None, gammas=LAYDEN_GAMMAS, times=LAYDEN_TIMES):
    """Grid-averaged quench proposal (= mean of quench_by_gamma) with low memory: H real symmetric, so
    U_t = V cos(wt) V^T - i V sin(wt) V^T and |U_t|^2 = R^2 + S^2 from two real GEMMs (no complex N x N arrays)."""
    n = len(J)
    a = layden_alpha(J, np.zeros(n) if h is None else h)
    Q = np.zeros((len(E), len(E)))
    for g in gammas:
        A = g * mixer(n)
        A[np.diag_indices_from(A)] += (1 - g) * a * E
        w, V = np.linalg.eigh(A)
        del A
        for t in times:
            R = (V * np.cos(w * t)) @ V.T
            Q += R * R
            R = (V * np.sin(w * t)) @ V.T
            Q += R * R
        del V
    return Q / (len(gammas) * len(times))


def _sector_mixers(n):
    """sum_k X_k restricted to the flip-even and flip-odd sectors, in the basis |r +-> = (|r> +- |NOT r>)/sqrt 2 over
    representatives r < 2^(n-1). X_k (k < n-1) permutes representatives; X_{n-1} maps |r +-> to +-|r xor (2^(n-1)-1) +->."""
    M = 2 ** (n - 1)
    r = np.arange(M)
    P = np.zeros((M, M))
    for k in range(n - 1):
        P[r, r ^ (1 << k)] += 1.0
    T = np.zeros((M, M))
    T[r, r ^ (M - 1)] = 1.0
    return P + T, P - T


def quench_mean_sectors(E, J, gammas=LAYDEN_GAMMAS, times=LAYDEN_TIMES, full=True):
    """Grid-averaged quench proposal for FIELD-FREE E via the flip sectors (two 2^(n-1) eigenproblems instead of one
    2^n): with U_e, U_o the sector blocks, <s|U|s'> = (U_e + U_o)/2 and <F s|U|s'> = (U_e - U_o)/2 on representatives.
    Returns (Qbar, Qfull): Qbar = partition-space proposal (|U_e|^2 + |U_o|^2)/2, Qfull = labelled-state proposal
    (None if full=False). VERIFY (tests): equals quench_mean and its lumping to 1e-12."""
    n = len(J)
    N, M = 2 ** n, 2 ** (n - 1)
    Er = np.asarray(E[:M], float)
    a = layden_alpha(J, np.zeros(n))
    Xe, Xo = _sector_mixers(n)
    A = np.zeros((M, M)); B = np.zeros((M, M))
    for g in gammas:
        eig = []
        for X in (Xe, Xo):
            H = g * X
            H[np.diag_indices_from(H)] += (1 - g) * a * Er
            eig.append(np.linalg.eigh(H))
            del H
        (we, Ve), (wo, Vo) = eig
        for t in times:
            Re = (Ve * np.cos(we * t)) @ Ve.T
            Ro = (Vo * np.cos(wo * t)) @ Vo.T
            A += (Re + Ro) ** 2; B += (Re - Ro) ** 2
            del Re, Ro
            Se = (Ve * np.sin(we * t)) @ Ve.T
            So = (Vo * np.sin(wo * t)) @ Vo.T
            A += (Se + So) ** 2; B += (Se - So) ** 2
            del Se, So
    A /= 4 * len(gammas) * len(times); B /= 4 * len(gammas) * len(times)
    Qbar = A + B
    if not full:
        return Qbar, None
    Q = np.empty((N, N))
    Q[:M, :M] = A; Q[:M, M:] = B[:, ::-1]; Q[M:, M:] = A[::-1, ::-1]; Q[M:, :M] = B[::-1, :]
    return Qbar, Q


def _rx_layer(n, b):
    """exp(-i b sum_k X_k) as a dense matrix (tensor power of a 2x2 rotation)."""
    r = np.array([[np.cos(b), -1j * np.sin(b)], [-1j * np.sin(b), np.cos(b)]])
    U = np.array([[1.0 + 0j]])
    for _ in range(n):
        U = np.kron(r, U)
    return U


def _apply_rx_layer(A, b):
    """Left-multiply A (N x M) by exp(-i b sum_k X_k) using per-qubit tensor contractions: O(n N M), not O(N^2 M)."""
    N = A.shape[0]
    n = int(np.log2(N))
    r = np.array([[np.cos(b), -1j * np.sin(b)], [-1j * np.sin(b), np.cos(b)]])
    T = A.reshape([2] * n + [A.shape[1]])
    for ax in range(n):
        T = np.moveaxis(np.tensordot(r, T, axes=([1], [ax])), 0, ax)
    return T.reshape(N, A.shape[1])


def qaoa_mc_unitary(E, alpha, b1, b2, g1):
    """Palindromic depth-2 proposal U = M(b1) P(g1) M(b2) P(g1) M(b1), P(g) = exp(-i g alpha diag(E))."""
    N = len(E)
    ph = np.exp(-1j * g1 * alpha * E)[:, None]
    A = _apply_rx_layer(np.eye(N, dtype=complex), b1)
    A = _apply_rx_layer(ph * A, b2)
    return _apply_rx_layer(ph * A, b1)


def qaoa_mc_proposal(E, alpha, b1, b2, g1):
    """Full QAOA-MC proposal matrix Q[x, x'] = |<x'|U|x>|^2 for FIELD-FREE E, applying U only to the 2^(n-1)
    representative inputs: U commutes with the global flip F, so Q[F x, F x'] = Q[x, x'] (rows of F r are the
    reversed rows of r). VERIFY (tests): equals |qaoa_mc_unitary|^2 transposed."""
    N = len(E)
    M = N // 2
    ph = np.exp(-1j * g1 * alpha * E)[:, None]
    A = np.zeros((N, M), dtype=complex)
    A[np.arange(M), np.arange(M)] = 1.0
    A = _apply_rx_layer(A, b1)
    A = _apply_rx_layer(ph * A, b2)
    A = _apply_rx_layer(ph * A, b1)
    Q = np.empty((N, N))
    Q[:M] = (np.abs(A) ** 2).T
    Q[M:] = Q[:M][::-1, ::-1]
    return Q


QAOA_GRID = [(b1, b2, g1) for b1 in (np.pi / 8, np.pi / 4) for b2 in (np.pi / 8, np.pi / 4, 3 * np.pi / 8)
             for g1 in (0.5, 1.0, 2.0)]


def qaoa_mc(E, J, h, grid=QAOA_GRID):
    a = layden_alpha(J, h)
    Q = np.zeros((len(E), len(E)))
    for b1, b2, g1 in grid:
        Q += np.abs(qaoa_mc_unitary(E, a, b1, b2, g1)) ** 2
    return Q / len(grid)
