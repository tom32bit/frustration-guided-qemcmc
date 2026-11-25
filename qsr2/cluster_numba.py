"""Compiled subset dynamic programmes for the exact Swendsen-Wang / Wolff matrices (optional; used by cluster.py when
Numba is importable). Same recursions and (mask x state) layout as cluster._tables / cluster._h_table.
VERIFY: tests compare with the NumPy versions and with brute-force bond enumeration."""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def tables(l, n):
    N = l.shape[0]
    w = np.zeros((N, N))
    for X in range(1, N):
        k = 0
        while (X >> (k + 1)) > 0:
            k += 1
        Xp = X ^ (1 << k)
        for s in range(N):
            acc = w[Xp, s]
            for j in range(n):
                if (Xp >> j) & 1:
                    acc += l[s, k, j]
            w[X, s] = acc
    g = np.zeros((N, N))
    acc = np.zeros(N)
    for T in range(1, N):
        low = T & -T
        rest = T ^ low
        if rest == 0:
            for s in range(N):
                g[T, s] = 1.0
            continue
        for s in range(N):
            acc[s] = 0.0
        sub = (rest - 1) & rest
        while True:
            U = low | sub
            V = T ^ U
            for s in range(N):
                acc[s] += g[U, s] * np.exp(w[T, s] - w[U, s] - w[V, s])
            if sub == 0:
                break
            sub = (sub - 1) & rest
        for s in range(N):
            v = 1.0 - acc[s]
            g[T, s] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    return w, g


@njit(cache=True)
def h_table(w, g):
    N = w.shape[0]
    h = np.zeros((N, N))
    for s in range(N):
        h[0, s] = 1.0
    acc = np.zeros(N)
    for S in range(1, N):
        low = S & -S
        rest = S ^ low
        for s in range(N):
            acc[s] = 0.0
        sub = rest
        while True:
            U = low | sub
            V = S ^ U
            for s in range(N):
                acc[s] += g[U, s] * np.exp(w[S, s] - w[U, s] - w[V, s]) * h[V, s]
            if sub == 0:
                break
            sub = (sub - 1) & rest
        for s in range(N):
            h[S, s] = 0.5 * acc[s]
    return h


@njit(parallel=True, cache=True)
def sw_wolff_lumped(l, n, popcount):
    """Partition-space (lumped) Swendsen-Wang and Wolff transition matrices, M x M with M = 2^(n-1), computed
    directly and in parallel over representative states s < M. The bond tables of s and NOT s are identical
    (satisfaction J_ij z_i z_j > 0 is flip-invariant), so l (M, n, n) holds log(1 - p) for representatives only.
    Per state s the same recursions as tables / h_table run on vectors over masks (no N x N tables are stored);
    the move s -> s xor A is credited to the class of s xor A. VERIFY (tests): equals mcmc.lump of the full
    matrices from cluster.sw_wolff_transitions."""
    M = l.shape[0]
    N = 2 * M
    full = N - 1
    SW = np.zeros((M, M))
    WO = np.zeros((M, M))
    for s in prange(M):
        w = np.zeros(N)
        for X in range(1, N):
            k = 0
            while (X >> (k + 1)) > 0:
                k += 1
            Xp = X ^ (1 << k)
            acc = w[Xp]
            for j in range(n):
                if (Xp >> j) & 1:
                    acc += l[s, k, j]
            w[X] = acc
        g = np.zeros(N)
        for T in range(1, N):
            low = T & -T
            rest = T ^ low
            if rest == 0:
                g[T] = 1.0
                continue
            acc = 0.0
            sub = (rest - 1) & rest
            while True:
                U = low | sub
                acc += g[U] * np.exp(w[T] - w[U] - w[T ^ U])
                if sub == 0:
                    break
                sub = (sub - 1) & rest
            v = 1.0 - acc
            g[T] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
        h = np.zeros(N)
        h[0] = 1.0
        for S in range(1, N):
            low = S & -S
            rest = S ^ low
            acc = 0.0
            sub = rest
            while True:
                U = low | sub
                V = S ^ U
                acc += g[U] * np.exp(w[S] - w[U] - w[V]) * h[V]
                if sub == 0:
                    break
                sub = (sub - 1) & rest
            h[S] = 0.5 * acc
        for A in range(N):
            cut = np.exp(w[full] - w[A] - w[full ^ A])
            t = s ^ A
            if t >= M:
                t = t ^ full
            SW[s, t] += cut * h[A] * h[full ^ A]
            if A > 0:
                WO[s, t] += popcount[A] / n * g[A] * cut
    return SW, WO
