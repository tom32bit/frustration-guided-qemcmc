"""Inverse participation ratios and long-time quench proposals on partition space (E8).

Follows Orfi & Sels, Phys. Rev. A 110, 052434 (2024) [arXiv:2408.07881], Eqs. 9, 11, 12 and 17, rewritten for
field-free networks on partition space (representatives r < 2^(n-1), M = 2^(n-1) partitions).

For H(g) = (1-g) a diag(E) + g sum_k X_k, split into the flip-even and flip-odd sectors (proposals._sector_mixers), the
long-time (dephased) quench keeps, in each sector, only the terms within one eigenspace:
  Qinf_g(r, r') = (1/2) sum over both sectors and their eigenspaces w of P_w(r, r')^2
  IPR_g(r)      = Qinf_g(r, r) = (1/2) sum P_w(r, r)^2                               in [1/M, 1]
with P_w the projector onto the eigenspace of energy w, in the sector basis |r +-> = (|r> +- |NOT r>)/sqrt 2. When no
level of a sector is degenerate, P_w = v v^T and these reduce to Orfi & Sels' eigenvector forms
(W_e W_e^T + W_o W_o^T)/2 and (sum W_e^2 + sum W_o^2)/2 with W = V * V. Degenerate levels do occur: a sub-network
with two or more regions that have no couplings to the rest has an exact symmetry, and then the eigenvector form is
wrong and depends on the arbitrary basis that eigh returns inside a level. With the eigenvectors v_a of one level,
P o P = sum_a w_a w_a^T + 2 sum_{a<b} u_ab u_ab^T (w_a = v_a^2, u_ab = v_a v_b, elementwise), which is how
sector_long_time computes it. (Amendment 1, results/e8_amendment1.txt: the first E8 run used the eigenvector form.)
Identities (tests/test_e8.py): Qinf_g symmetric, rows sum to 1, diag(Qinf_g) = IPR_g, IPR = 1 and Qinf = I at g = 0
when the partition energies are distinct, and Qinf_g equals the long-time average of the finite-time quench, also on
a network with degenerate levels.

Bound. For B = partitions above an energy level and B^c = the rest, Metropolis moves from B to B^c are always
accepted, so for the chain built from Qinf = mean_g Qinf_g,
  gap <= Lambda(B) = E(B, B^c) / (pi(B) pi(B^c))
      <= mean_g sqrt(<IPR_g>_{pi_B, B} <IPR_g>_{B^c}) / pibar(B^c)
using P_w(x, y)^2 <= P_w(x, x) P_w(y, y) and the Cauchy-Schwarz inequality (as in Orfi & Sels' Eq. 17), with
pibar(B^c) = pi(B^c) / |B^c| the mean Gibbs probability of a partition in B^c and <.>_{B^c} the plain mean.
"""
import numpy as np

from . import mcmc, proposals

DEGENERACY_RTOL = 1e-8       # eigenvalues closer than this (relative) are one level; exact degeneracies give ~1e-15


def _levels(w):
    """Index ranges [s, e) of equal eigenvalues in the ascending spectrum w."""
    out, s = [], 0
    for i in range(1, len(w) + 1):
        if i == len(w) or w[i] - w[i - 1] > DEGENERACY_RTOL * max(1.0, abs(w[i])):
            out.append((s, i))
            s = i
    return out


def sector_long_time(Er, a, g, n):
    """(Qinf_g, IPR_g) for one gamma, exact for any spectrum (module docstring).
    VERIFY: rows of Qinf_g sum to 1, diag(Qinf_g) = IPR_g; equals the long-time average of the quench (tests)."""
    Xe, Xo = proposals._sector_mixers(n)
    M = len(Er)
    Q, I = np.zeros((M, M)), np.zeros(M)
    for X in (Xe, Xo):
        H = g * X
        H[np.diag_indices_from(H)] += (1 - g) * a * Er
        w, V = np.linalg.eigh(H)
        W = V * V
        Q += 0.5 * W @ W.T
        I += 0.5 * (W ** 2).sum(1)
        pairs = [V[:, s + i] * V[:, s + j] for s, e in _levels(w) for i in range(e - s) for j in range(i + 1, e - s)]
        if pairs:                                     # degenerate levels: add the cross terms of P o P
            U = np.array(pairs).T
            Q += U @ U.T
            I += (U ** 2).sum(1)
    return Q, I


def ipr_and_qinf(E, J, gammas=proposals.LAYDEN_GAMMAS):
    """Per-gamma partition-space IPR (array len(gammas) x M) and the gamma-averaged long-time proposal Qinf (M x M).
    E: energies of all 2^n labelled states (ising.energies); only the representatives E[:M] are used."""
    n = len(J)
    M = 2 ** (n - 1)
    Er = np.asarray(E[:M], float)
    a = proposals.layden_alpha(J, np.zeros(n))
    iprs = np.empty((len(gammas), M))
    Q = np.zeros((M, M))
    for i, g in enumerate(gammas):
        Qg, iprs[i] = sector_long_time(Er, a, g, n)
        Q += Qg
    return iprs, Q / len(gammas)


def gibbs_log(Er, beta):
    lp = -beta * (Er - Er.min())
    return lp - np.log(np.exp(lp).sum())


def chain_gap(Q, Er, beta):
    """Absolute spectral gap of the partition-space Metropolis chain with symmetric proposal Q."""
    return mcmc.spectral_gap(mcmc.mh_matrix(Q, Er, beta), logpi=gibbs_log(Er, beta))


def threshold_bounds(Q, Er, beta, iprs):
    """Minimum over energy-threshold cuts of the bottleneck bound Lambda(B) and of the IPR bound (module docstring).
    Returns (Lambda_star, ipr_star, level_Lambda, level_ipr) with the levels given as |B^c| (partitions in the low set).
    VERIFY: chain gap <= Lambda_star <= ipr_star."""
    order = np.argsort(Er, kind="stable")
    e = Er[order]
    logpi = gibbs_log(Er, beta)[order]
    pi = np.exp(logpi)
    P = mcmc.mh_matrix(Q, Er, beta)[np.ix_(order, order)]
    flow = pi[:, None] * P                                      # E(x, y) = pi(x) P(x, y)
    # cut after k partitions: B^c = order[:k] (low energies), B = order[k:]; flow from B to B^c
    col_prefix = np.cumsum(flow, axis=1)                        # sum_{j < k+1} flow[i, j]
    row_suffix = np.cumsum(col_prefix[::-1], axis=0)[::-1]      # sum_{i >= i0} col_prefix[i, k]
    M = len(e)
    ks = [k for k in range(1, M) if e[k] > e[k - 1]]            # cuts between distinct energy levels
    lam, ipb = [], []
    cpi = np.cumsum(pi)                                         # pi(B^c) for B^c = order[:k]: cpi[k-1]
    spi = np.cumsum(pi[::-1])[::-1]                             # pi(B) = spi[k], as a suffix sum (no cancellation)
    I = iprs[:, order]
    sI_pi = np.cumsum((I * pi[None, :])[:, ::-1], axis=1)[:, ::-1]   # per gamma: sum_{x >= k} pi(x) IPR(x)
    cI = np.cumsum(I, axis=1)                                   # per gamma: sum_{x < k} IPR(x)
    for k in ks:
        flowBBc = row_suffix[k, k - 1]                          # rows i >= k, columns j <= k-1
        piBc, piB = cpi[k - 1], spi[k]
        lam.append(flowBBc / (piB * piBc))
        mean_B = sI_pi[:, k] / piB                              # <IPR>_{pi_B, B}
        mean_Bc = cI[:, k - 1] / k                              # plain mean over B^c
        pibar = piBc / k
        ipb.append(float(np.mean(np.sqrt(mean_B * mean_Bc)) / pibar))
    lam, ipb = np.array(lam), np.array(ipb)
    return float(lam.min()), float(ipb.min()), int(ks[int(lam.argmin())]), int(ks[int(ipb.argmin())])
