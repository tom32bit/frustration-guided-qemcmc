"""Exact patch-conditional Ising models for 2-label decisions, and the frustration index.

For a 2-label decision the modularity Hamiltonian H(c) = -sum_{i<j} 2 B_ij delta(c_i, c_j) becomes, with
delta = (1 + z_i z_j)/2, an Ising model E(z) = -sum_{i<j} J_ij z_i z_j - sum_i h_i z_i + const with:

  P1 border  (nodes U choose community A (z=+1) or B (z=-1), rest fixed):
              J_ij = B_ij (i, j in U),  h_i = sum_{j not in U, c_j = A} B_ij - sum_{j not in U, c_j = B} B_ij
  P2 split   (whole module M splits into two labels, rest outside {M1, M2}):
              J_ij = B_ij restricted to M, h = 0  (cross terms vanish exactly)
  P3 standalone ROI network (the ROI set is the object of study):
              J_ij = own-modularity matrix of the induced sub-network, h = 0
"""
import numpy as np

from .ising import energies, spins
from .networks import modularity_matrix


def border_patch(W, c, U, a, b, gamma=1.0):
    B = modularity_matrix(W, gamma)
    U = np.asarray(U)
    out = np.ones(len(W), bool); out[U] = False
    J = B[np.ix_(U, U)].copy(); np.fill_diagonal(J, 0)
    h = np.array([B[i, out & (c == a)].sum() - B[i, out & (c == b)].sum() for i in U])
    return J, h


def split_patch(W, M, gamma=1.0):
    B = modularity_matrix(W, gamma)
    J = B[np.ix_(M, M)].copy(); np.fill_diagonal(J, 0)
    return J, np.zeros(len(M))


def standalone_patch(W, U):
    Ws = W[np.ix_(U, U)]
    J = modularity_matrix(Ws, 1.0).copy(); np.fill_diagonal(J, 0)
    return J, np.zeros(len(U))


def frustration(J, h=None):
    """F = share of |J| on couplings unsatisfied by the exact ground state. F = 0 for ferromagnets.
    VERIFY: F(ferromagnet) == 0; F(antiferromagnetic triangle) == 1/3."""
    n = len(J)
    z = spins(n)[int(np.argmin(energies(J, h)))].astype(float)
    iu = np.triu_indices(n, 1)
    w = J[iu]
    unsat = w * z[iu[0]] * z[iu[1]] < 0
    tot = np.abs(w).sum()
    return float(np.abs(w)[unsat].sum() / tot) if tot > 0 else 0.0


def field_to_coupling_ratio(J, h):
    iu = np.triu_indices(len(J), 1)
    rj = np.sqrt(np.mean(J[iu] ** 2))
    return float(np.sqrt(np.mean(np.asarray(h) ** 2)) / rj) if rj > 0 else np.inf


def canonical_rois(labels, min_n=4, max_n=10):
    """Anatomically defined ROI networks: Yeo-7 network x hemisphere from Schaefer labels
    ('7Networks_LH_Default_PFC_1' -> ('LH', 'Default'))."""
    groups = {}
    for i, lab in enumerate(labels):
        parts = lab.split("_")
        groups.setdefault((parts[1], parts[2]), []).append(i)
    return {f"{hemi}_{net}": np.array(idx) for (hemi, net), idx in groups.items() if min_n <= len(idx) <= max_n}
