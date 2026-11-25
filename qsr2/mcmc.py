"""Metropolis-Hastings transition matrices, exact spectral gaps, and exact parallel-tempering gaps."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

GAP_FLOOR = 1e-14  # below this, double-precision eigenvalues are not trustworthy


def mh_matrix(Q, E, beta):
    """P[x, x'] = Q[x, x'] min(1, exp(-beta (E' - E))) for x' != x; diagonal absorbs rejections.
    Valid (detailed balance w.r.t. Gibbs) when Q is symmetric."""
    N = len(E)
    A = np.minimum(1.0, np.exp(-beta * (E[None, :] - E[:, None])))
    P = Q * A
    np.fill_diagonal(P, 0.0)
    P[np.arange(N), np.arange(N)] = 1.0 - P.sum(1)
    return P


def spectral_gap(P, pi=None, checks=False, logpi=None):
    """Absolute spectral gap 1 - max(|lambda_2|, |lambda_min|) of a pi-reversible P, via S = D^{1/2} P D^{-1/2}.
    S is formed in log space, S_ij = exp(log P_ij + (log pi_i - log pi_j)/2) where P_ij > 0, so that Gibbs weights
    that underflow at low temperature cannot produce 0/0 (for Metropolis moves S_ij = Q_ij exp(-beta|dE|/2) <= Q_ij).
    VERIFY (checks=True): rows sum to 1, pi P = pi, S symmetric."""
    if logpi is None:
        logpi = np.log(pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(P > 0, np.exp(np.log(np.where(P > 0, P, 1.0)) + 0.5 * (logpi[:, None] - logpi[None, :])), 0.0)
    lam = np.sort(np.linalg.eigvalsh((S + S.T) / 2))[::-1]
    gap = float(max(1.0 - max(abs(lam[1]), abs(lam[-1])), 0.0))
    if checks:
        pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
        return gap, {"row_sum": float(np.max(np.abs(P.sum(1) - 1))), "stationarity": float(np.max(np.abs(pi @ P - pi))),
                     "reversibility": float(np.max(np.abs(S - S.T)))}
    return gap


def gap_of(Q, E, beta):
    return spectral_gap(mh_matrix(Q, E, beta), logpi=-beta * (E - E.min()))


def pt_matrix(E, n, betas, p_move=0.5, lumped=False):
    """Sparse transition matrix and log-stationary weights of parallel tempering with R = len(betas) replicas (sparse, R*n <= ~20).
    One step: with prob p_move pick a uniform replica and apply a local+label-swap Metropolis move at its beta;
    otherwise attempt an exchange of a uniform adjacent pair. Each step = one proposal (cost-comparable).
    lumped=True: the exact chain on partition classes {x, NOT x} of every replica (field-free E only), i.e. states are
    R-tuples of representatives with top bit 0; flip-all moves become identities. VERIFY: test_lumping."""
    N = 2 ** n
    M = N // 2 if lumped else N
    R = len(betas)
    S = M ** R
    states = np.arange(S, dtype=np.int64)
    xs = [(states // M ** r) % M for r in range(R)]
    rows, cols, vals = [], [], []
    diag = np.ones(S)
    for r in range(R):
        xr = xs[r]
        moves = [(xr ^ (1 << k), p_move / R * 0.5 / n) for k in range(n)] + [(xr ^ (N - 1), p_move / R * 0.5)]
        for new, w in moves:
            if lumped:
                new = np.where(new >= M, new ^ (N - 1), new)
            acc = np.minimum(1.0, np.exp(-betas[r] * (E[new] - E[xr])))
            tgt = states + (new - xr) * M ** r
            p = w * acc
            rows.append(states); cols.append(tgt); vals.append(p); diag -= p
    for r in range(R - 1):
        a, b = xs[r], xs[r + 1]
        acc = np.minimum(1.0, np.exp((betas[r] - betas[r + 1]) * (E[a] - E[b])))
        tgt = states + (b - a) * M ** r + (a - b) * M ** (r + 1)
        p = (1 - p_move) / (R - 1) * acc
        rows.append(states); cols.append(tgt); vals.append(p); diag -= p
    rows.append(states); cols.append(states); vals.append(diag)
    P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(S, S))
    logpi = np.zeros(S)
    for r in range(R):
        logpi -= betas[r] * (E[xs[r]] - E.min())
    return P, logpi


def pt_gap(E, n, betas, p_move=0.5, maxiter=5000, lumped=False):
    """Exact absolute spectral gap of parallel tempering (see pt_matrix; lumped=True -> partition space)."""
    P, logpi = pt_matrix(E, n, betas, p_move, lumped=lumped)
    C = P.tocoo()
    keep = C.data > 0
    r, c = C.row[keep], C.col[keep]
    data = np.exp(np.log(C.data[keep]) + 0.5 * (logpi[r] - logpi[c]))
    Ssym = sp.csr_matrix((data, (r, c)), shape=P.shape)
    Ssym = (Ssym + Ssym.T) / 2
    top = eigsh(Ssym, k=2, which="LA", return_eigenvectors=False, tol=1e-10, maxiter=maxiter)
    bot = eigsh(Ssym, k=1, which="SA", return_eigenvectors=False, tol=1e-8, maxiter=maxiter)
    lam2 = sorted(top)[0]
    return float(max(1.0 - max(abs(lam2), abs(bot[0])), 0.0))


# ---------------------------------------------------------------------------------------------------------------
# Partition space. For field-free E (E(x) = E(NOT x)) every kernel used here commutes with the global flip F, so it
# is exactly lumpable onto the 2^(n-1) bipartitions {x, F x}; the lumped spectrum is the F-even part of the full one.
# Co-assignment observables z_i z_j are F-even, so their autocorrelations are governed by the LUMPED chain only.

def lump(P, n):
    """P_bar[a, b] = P[a, b] + P[a, F b] on representatives a, b < 2^(n-1) (top bit 0)."""
    N = 2 ** n
    r = np.arange(N // 2)
    return P[np.ix_(r, r)] + P[np.ix_(r, r ^ (N - 1))]


def flip_symmetry_error(P, n):
    N = 2 ** n
    idx = np.arange(N)
    return float(np.max(np.abs(P[np.ix_(idx ^ (N - 1), idx ^ (N - 1))] - P)))


def lumped_gap(P, E, beta, n):
    """Absolute spectral gap of the partition-space chain (requires field-free E)."""
    N = 2 ** n
    return spectral_gap(lump(P, n), logpi=-beta * (E[: N // 2] - E.min()))


def coassignment_observables(n):
    """Matrix (2^(n-1) x n(n-1)/2) of z_i z_j on partition representatives, pairs in np.triu_indices order."""
    idx = np.arange(2 ** (n - 1))
    Z = 1 - 2 * ((idx[:, None] >> np.arange(n)[None, :]) & 1)
    iu = np.triu_indices(n, 1)
    return (Z[:, iu[0]] * Z[:, iu[1]]).astype(float)


def asymptotic_variances(Pbar, logpi_bar, Fobs):
    """Exact asymptotic variances sigma^2(f) = lim T Var(mean_T f) of every column f of Fobs for the chain Pbar
    (any ergodic chain), via the fundamental matrix: sigma^2 = <fc, (2 Z - I) fc>_pi, Z = (I - P + 1 pi^T)^-1.
    Steps needed for RMSE eps of an estimator ~ sigma^2 / eps^2. Returns (sigma2, var_pi)."""
    pi = np.exp(logpi_bar - logpi_bar.max()); pi /= pi.sum()
    m = len(pi)
    fc = Fobs - pi @ Fobs
    A = np.eye(m) - Pbar + np.outer(np.ones(m), pi)
    X = np.linalg.solve(A, fc)
    var = pi @ (fc * fc)
    sig2 = 2 * (pi @ (fc * X)) - var
    return sig2, var


def sector_gaps(P, E, beta, n):
    """(partition-space gap, labelled-state gap) of a flip-symmetric P from two 2^(n-1) eigenproblems: the full
    spectrum is spec(even block P[r,r] + P[r,Fr]) U spec(odd block P[r,r] - P[r,Fr]). VERIFY: tests vs spectral_gap."""
    N, M = 2 ** n, 2 ** (n - 1)
    r = np.arange(M)
    A, B = P[np.ix_(r, r)], P[np.ix_(r, r ^ (N - 1))]
    lp = -beta * (E[:M] - E.min())
    ev = _sym_eigs(A + B, lp)
    od = _sym_eigs(A - B, lp)
    lumped = float(max(1.0 - max(abs(ev[-2]), abs(ev[0])), 0.0))
    full = float(max(1.0 - max(abs(ev[-2]), abs(ev[0]), np.max(np.abs(od))), 0.0))
    return lumped, full


def _sym_eigs(P, logpi):
    """Eigenvalues (ascending) of a pi-reversible (possibly signed) matrix via the log-space similarity transform."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.abs(P)
        S = np.where(mag > 0, np.sign(P) * np.exp(np.log(np.where(mag > 0, mag, 1.0)) + 0.5 * (logpi[:, None] - logpi[None, :])), 0.0)
    return np.linalg.eigvalsh((S + S.T) / 2)
