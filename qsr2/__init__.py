"""FG-QeMCMC (package qsr2) - frustration-guided quantum-enhanced sampling of network-partition uncertainty.

Modules
  networks   : connectome loading, weighted Leiden, modularity matrix, co-assignment / entropy (fixed QSR v1 pieces)
  patches    : exact patch-conditional Ising models (P1 border, P2 module split, P3 standalone ROI network), frustration F
  ising      : spin enumeration, energies, Gibbs weights
  proposals  : exact proposal matrices - local, label-swap, uniform, block heat-bath, quantum quench (Layden 2023),
               palindromic QAOA-MC (Nakano 2024)
  mcmc       : Metropolis-Hastings matrices, spectral gaps, parallel tempering (sparse, exact), sanity checks
  legacy     : corrected versions of the QSR v1 Phase-2 pieces (Ising mapping, unitary QAOA sampler)
  pl_circuits: PennyLane palindromic proposal circuits (hardware path), checked against the exact matrices
"""
__version__ = "0.1.0"
