"""Inverse participation ratios and long-time quench proposals on partition space (E8).

Follows Orfi & Sels, Phys. Rev. A 110, 052434 (2024) [arXiv:2408.07881], Eqs. 9, 11, 12 and 17, rewritten for
field-free networks on partition space (representatives r < 2^(n-1), M = 2^(n-1) partitions).

For H(g) = (1-g) a diag(E) + g sum_k X_k, split into the flip-even and flip-odd sectors (proposals._sector_mixers), with
eigenvectors V_e, V_o and W = V * V (elementwise):
  IPR_g(r) = (sum_n W_e[r,n]^2 + sum_n W_o[r,n]^2) / 2          in [1/M, 1]
  Qinf_g   = (W_e W_e^T + W_o W_o^T) / 2                         dephased (t -> infinity) proposal, lumped
Starting from partition r, eigenstate n of a sector is occupied with weight W[r,n]/2 (the state |r> has amplitude
1/sqrt2 on each sector), and an eigenstate of a sector lands in partition r' with probability W[r',n]; summing over
both sectors gives Qinf_g. Identities (tests/test_e8.py): Qinf_g symmetric, rows sum to 1, diag(Qinf_g) = IPR_g,
IPR = 1 and Qinf = I at g = 0 when the partition energies are distinct.

Bound. For B = partitions above an energy level and B^c = the rest, Metropolis moves from B to B^c are always
accepted, so for the chain built from Qinf = mean_g Qinf_g,
  gap <= Lambda(B) = E(B, B^c) / (pi(B) pi(B^c))
       = mean_g (1/pibar(B^c)) sum_n f_g(n) g_g(n)
      <= mean_g sqrt(<IPR_g>_{pi_B, B} <IPR_g>_{B^c}) / pibar(B^c)          (Cauchy-Schwarz twice, as in Eq. 17)
with pibar(B^c) = pi(B^c) / |B^c| the mean Gibbs probability of a partition in B^c and <.>_{B^c} the plain mean.
"""
import numpy as np

from . import mcmc, proposals


def sector_weights(Er, a, g, n):
    """W_e, W_o (elementwise squared eigenvectors) of H(g) in the flip-even / flip-odd sectors.
    VERIFY: columns of W sum to 1 (normalised eigenvectors), rows of W sum to 1 (completeness)."""
    Xe, Xo = proposals._sector_mixers(n)
    out = []
    for X in (Xe, Xo):
        H = g * X
        H[np.diag_indices_from(H)] += (1 - g) * a * Er
        V = np.linalg.eigh(H)[1]
        out.append(V * V)
    return out


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
        We, Wo = sector_weights(Er, a, g, n)
        iprs[i] = 0.5 * ((We ** 2).sum(1) + (Wo ** 2).sum(1))
        Q += 0.5 * (We @ We.T + Wo @ Wo.T)
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
