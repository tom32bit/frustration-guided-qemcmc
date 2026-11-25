"""Compiled Chebyshev kernel for the quench proposal (optional; used by chains.ChebQuenchSampler when importable).

psi = sum_k coef_k T_k(H') e_s with H' = diag(d) + gD sum_b X_b, one fused pass per term over all 2^n amplitudes:
T_{k+1}[x] = 2 (d[x] T_k[x] + gD sum_b T_k[x ^ 2^b]) - T_{k-1}[x];  psi[x] += coef_{k+1} T_{k+1}[x].
At n <= 16 the three working vectors (<= 1 MB each) stay cache-resident. VERIFY: equals the NumPy path (tests).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def evolve_basis(s, d, gD, coef, n):
    N = d.shape[0]
    T0 = np.zeros(N, np.complex128)
    T1 = np.zeros(N, np.complex128)
    T2 = np.zeros(N, np.complex128)
    psi = np.zeros(N, np.complex128)
    T0[s] = 1.0
    psi[s] = coef[0]
    K = coef.shape[0]
    if K == 1:
        return psi
    c = coef[1]
    for x in range(N):
        acc = 0.0j
        for b in range(n):
            acc += T0[x ^ (1 << b)]
        v = d[x] * T0[x] + gD * acc
        T1[x] = v
        psi[x] += c * v
    for k in range(2, K):
        c = coef[k]
        for x in range(N):
            acc = 0.0j
            for b in range(n):
                acc += T1[x ^ (1 << b)]
            v = 2.0 * (d[x] * T1[x] + gD * acc) - T0[x]
            T2[x] = v
            psi[x] += c * v
        T0, T1, T2 = T1, T2, T0
    return psi
