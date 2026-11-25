"""Simulated Markov chains for E3 Tier B (n > 12, where exact transition matrices are out of reach).

All chains target pi(z) ∝ exp(-beta E(z)), E = -sum_{i<j} J_ij z_i z_j (field-free, P3 ROI networks).
State = integer basis index (bit k <-> spin k, z = 1 - 2 bit). Each function advances a batch of independent chains
by ONE step (one proposal per chain; for PT one proposal on one replica) and returns the new states.

Quantum quench (Layden et al. 2023): gamma, t drawn uniformly from the G1 grid per chain per step;
|psi> = exp(-i t [(1-gamma) alpha H_C + gamma sum X]) |s>, s' ~ |<s'|psi>|^2, Metropolis-Hastings accept.
All six t values of the grid are produced by one Krylov/Taylor call per gamma (expm_multiply on a linspace).
VERIFY (tests): the empirical one-step transition frequencies of every simulated chain match the exact matrices
of proposals.py / cluster.py at n = 4.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from .proposals import LAYDEN_GAMMAS, LAYDEN_TIMES, layden_alpha


def zof(states, n):
    return (1 - 2 * ((np.asarray(states)[:, None] >> np.arange(n)[None, :]) & 1)).astype(float)


def sparse_mixer(n):
    N = 2 ** n
    idx = np.arange(N)
    rows = np.concatenate([idx] * n)
    cols = np.concatenate([idx ^ (1 << k) for k in range(n)])
    return sp.csr_matrix((np.ones(n * N), (rows, cols)), shape=(N, N))


class QuenchSampler:
    def __init__(self, J, E, beta, gammas=LAYDEN_GAMMAS, times=LAYDEN_TIMES):
        self.n, self.E, self.beta = len(J), E, beta
        self.gammas, self.times = np.asarray(gammas), np.asarray(times)
        assert np.allclose(np.diff(self.times), self.times[1] - self.times[0]), "times must be evenly spaced"
        a = layden_alpha(J, np.zeros(len(J)))
        X = sparse_mixer(self.n)
        self.H = [((1 - g) * a * sp.diags(E) + g * X).tocsr() for g in self.gammas]

    def step(self, states, rng):
        states = np.asarray(states).copy()
        C, N = len(states), 2 ** self.n
        gi = rng.integers(0, len(self.gammas), C)
        ti = rng.integers(0, len(self.times), C)
        prop = states.copy()
        for g in np.unique(gi):
            sel = np.where(gi == g)[0]
            B = np.zeros((N, len(sel)), complex)
            B[states[sel], np.arange(len(sel))] = 1.0
            # expm_multiply over a linspace returns exp(-i t_k H) B for every grid time t_k
            out = expm_multiply(-1j * self.H[g], B, start=self.times[0], stop=self.times[-1],
                                num=len(self.times), endpoint=True)
            for c, ch in enumerate(sel):
                p = np.abs(out[ti[ch], :, c]) ** 2
                prop[ch] = rng.choice(N, p=p / p.sum())
        return _mh(states, prop, self.E, self.beta, rng)


class ChebQuenchSampler:
    """Same chain as QuenchSampler, with exp(-i t H) |s> by a Chebyshev expansion: H = c_d diag(E) + g sum X has
    spectrum inside [c_d Emin - g n, c_d Emax + g n]; with H' = (H - c)/D,
    exp(-i t H) = exp(-i c t) sum_k (2 - delta_k0) (-i)^k J_k(D t) T_k(H').
    One recursion per gamma serves all grid times (coefficient sets accumulated side by side); the X-mixer is
    applied in place with axis flips of the (2,)*n-reshaped block. Truncation |J_k(D t)| < tol beyond K.
    VERIFY (tests): equals expm_multiply to 1e-10 and reproduces the exact one-step kernel."""
    def __init__(self, J, E, beta, gammas=LAYDEN_GAMMAS, times=LAYDEN_TIMES, tol=1e-13, dtype=complex, use_numba=True):
        """dtype=np.complex64 halves memory traffic (the step is bandwidth-bound); tests bound its error.
        use_numba: compiled fused kernel (qsr2.cheb_numba) for single-time evolution when Numba is importable."""
        from scipy.special import jv
        self.kernel = None
        if use_numba and dtype == complex:
            try:
                from .cheb_numba import evolve_basis
                self.kernel = evolve_basis
            except ImportError:
                self.kernel = None
        self.n, self.E, self.beta = len(J), np.asarray(E, float), beta
        self.gammas, self.times = np.asarray(gammas), np.asarray(times)
        self.dtype = dtype
        a = layden_alpha(J, np.zeros(len(J)))
        self.par, self.Kt = [], []
        for g in self.gammas:
            cd = (1 - g) * a
            lo, hi = cd * self.E.min() - g * self.n, cd * self.E.max() + g * self.n
            c, D = (lo + hi) / 2, (hi - lo) / 2 * (1 + 1e-9) + 1e-12
            K = int(D * self.times.max() + 10 * (D * self.times.max()) ** (1 / 3) + 30)
            k = np.arange(K + 1)
            coef = np.array([(2 - (k == 0)) * (-1j) ** k * jv(k, D * t) * np.exp(-1j * c * t) for t in self.times])
            keep = np.max(np.abs(coef), axis=0) > tol
            K = int(np.max(np.nonzero(keep)[0])) + 1
            self.par.append((cd, g, c, D, coef[:, :K + 1]))
            self.Kt.append([int(np.max(np.nonzero(np.abs(row) > tol)[0])) + 1 for row in coef])

    def _H(self, V, cd, g, c, D):
        n = self.n
        Vr = V.reshape((2,) * n + (V.shape[1],))
        X = np.zeros_like(Vr)
        for ax in range(n):
            X += np.flip(Vr, axis=ax)
        diag = ((cd * self.E - c) / D).astype(V.real.dtype)
        return diag[:, None] * V + (g / D) * X.reshape(V.shape)

    def evolve(self, gi, states, ti=None):
        """exp(-i t H_g)|s>: for every grid time (array (len(times), 2^n, C)), or, with ti given, only for the single
        time index ti (array (2^n, C)) running the recursion just as far as that time needs."""
        cd, g, c, D, coef = self.par[gi]
        if ti is not None and self.kernel is not None:
            K = self.Kt[gi][ti]
            d = (cd * self.E - c) / D
            cf = np.ascontiguousarray(coef[ti, :K]).astype(np.complex128)
            return np.stack([self.kernel(int(s0), d, g / D, cf, self.n) for s0 in states], axis=1)
        if ti is not None:
            coef = coef[ti:ti + 1]
            nz = np.nonzero(np.abs(coef[0]) > 0)[0]
            coef = coef[:, : int(nz.max()) + 1] if len(nz) else coef[:, :1]
            K = self.Kt[gi][ti]
            coef = coef[:, :K]
        coef = coef.astype(self.dtype)
        N = 2 ** self.n
        T0 = np.zeros((N, len(states)), self.dtype)
        T0[states, np.arange(len(states))] = 1.0
        out = coef[:, 0][:, None, None] * T0[None]
        if coef.shape[1] > 1:
            T1 = self._H(T0, cd, g, c, D)
            out += coef[:, 1][:, None, None] * T1[None]
            for k in range(2, coef.shape[1]):
                T0, T1 = T1, 2 * self._H(T1, cd, g, c, D) - T0
                out += coef[:, k][:, None, None] * T1[None]
        return out if ti is None else out[0]

    def step(self, states, rng):
        """One step for every chain. Chains are grouped by (gamma, t): each group runs only the Chebyshev terms its
        own time needs (|J_k(D t)| > tol)."""
        states = np.asarray(states).copy()
        C, N = len(states), 2 ** self.n
        gi = rng.integers(0, len(self.gammas), C)
        ti = rng.integers(0, len(self.times), C)
        prop = states.copy()
        for g, t in sorted(set(zip(gi.tolist(), ti.tolist()))):
            sel = np.where((gi == g) & (ti == t))[0]
            out = self.evolve(g, states[sel], ti=t)
            for c, ch in enumerate(sel):
                p = np.abs(out[:, c]).astype(float) ** 2
                prop[ch] = rng.choice(N, p=p / p.sum())
        return _mh(states, prop, self.E, self.beta, rng)


def _mh(states, prop, E, beta, rng):
    acc = rng.random(len(states)) < np.exp(-beta * np.maximum(E[prop] - E[states], 0.0))
    return np.where(acc, prop, states)


def local_step(states, n, E, beta, rng):
    prop = states ^ (1 << rng.integers(0, n, len(states)))
    return _mh(states, prop, E, beta, rng)


def label_swap_step(states, n, E, beta, rng):
    flipall = rng.random(len(states)) < 0.5
    prop = np.where(flipall, states ^ (2 ** n - 1), states ^ (1 << rng.integers(0, n, len(states))))
    return _mh(states, prop, E, beta, rng)


def _bonds(z, J, beta, rng):
    """Open bonds on satisfied couplings with p = 1 - exp(-2 beta |J|); returns adjacency (n x n bool)."""
    n = len(z)
    sat = (J * np.outer(z, z)) > 0
    open_ = sat & (rng.random((n, n)) < 1 - np.exp(-2 * beta * np.abs(J)))
    open_ = np.triu(open_, 1)
    return open_ | open_.T


def _components(adj):
    n = len(adj)
    lab = -np.ones(n, int)
    c = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        stack = [s]; lab[s] = c
        while stack:
            i = stack.pop()
            for j in np.where(adj[i] & (lab < 0))[0]:
                lab[j] = c; stack.append(j)
        c += 1
    return lab, c


def sw_step(states, n, J, beta, rng):
    out = states.copy()
    Z = zof(states, n)
    for c, z in enumerate(Z):
        lab, nc = _components(_bonds(z, J, beta, rng))
        flip = rng.random(nc) < 0.5
        mask = flip[lab]
        out[c] = states[c] ^ int(np.sum((1 << np.arange(n))[mask]))
    return out


def wolff_step(states, n, J, beta, rng):
    out = states.copy()
    Z = zof(states, n)
    for c, z in enumerate(Z):
        adj = _bonds(z, J, beta, rng)
        seed = rng.integers(n)
        seen = np.zeros(n, bool); seen[seed] = True; stack = [seed]
        while stack:
            i = stack.pop()
            for j in np.where(adj[i] & ~seen)[0]:
                seen[j] = True; stack.append(j)
        out[c] = states[c] ^ int(np.sum((1 << np.arange(n))[seen]))
    return out


def pt_step(reps, n, E, betas, rng, p_move=0.5):
    """reps: (C, R) states; replica r at betas[r]. Same kernel as mcmc.pt_matrix (one proposal per step)."""
    reps = reps.copy()
    C, R = reps.shape
    move = rng.random(C) < p_move
    for c in np.where(move)[0]:
        r = rng.integers(R)
        s = reps[c, r:r + 1]
        reps[c, r] = label_swap_step(s, n, E, betas[r], rng)[0]
    for c in np.where(~move)[0]:
        r = rng.integers(R - 1)
        a, b = reps[c, r], reps[c, r + 1]
        if rng.random() < np.exp(min(0.0, (betas[r] - betas[r + 1]) * (E[a] - E[b]))):
            reps[c, r], reps[c, r + 1] = b, a
    return reps


def exact_corr(J, E, beta):
    """Exact <z_i z_j> under pi (enumeration; n <= ~20)."""
    n = len(J)
    w = np.exp(-beta * (E - E.min())); w /= w.sum()
    C = np.zeros((n, n))
    idx = np.arange(2 ** n)
    for i in range(n):
        zi = 1 - 2 * ((idx >> i) & 1)
        for j in range(i, n):
            zj = 1 - 2 * ((idx >> j) & 1)
            C[i, j] = C[j, i] = float(w @ (zi * zj))
    return C


class RunningCorr:
    """Running estimates of <z_i z_j> per chain; rmse() = mean over chains of RMSE over pairs i<j."""
    def __init__(self, C, n, exact):
        self.S = np.zeros((C, n, n)); self.T = 0; self.n = n
        self.iu = np.triu_indices(n, 1); self.exact = exact[self.iu]

    def add(self, states):
        Z = zof(states, self.n)
        self.S += Z[:, :, None] * Z[:, None, :]; self.T += 1

    def rmse(self):
        est = (self.S / self.T)[:, self.iu[0], self.iu[1]]
        return float(np.mean(np.sqrt(np.mean((est - self.exact[None]) ** 2, axis=1))))
