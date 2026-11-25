# E8: frustration and the participation-ratio bound (protocol, 24 Nov 2026)

**Status.** This plan is fixed before any participation ratio is computed. Its text, the code it names and the tests are
fingerprinted in `results/e8_prereg_hashes.txt` before the run. The networks and their mixing advantages are those of
the scaling study (E3 Tier A), which were already known. What is new is the participation ratio and the quantities
built from it.

## 1. Why

Orfi & Sels bound the speed of QeMCMC in the long-time limit of the quench:
- *Phys. Rev. A* 110, 052434 (2024), arXiv:2408.07881, Eqs. 11–17;
- see also *Phys. Rev. A* 110, 052414.

The spectral gap of the chain is bounded by the inverse participation ratio (IPR) of the classical configurations in the
eigenbasis of the quench Hamiltonian. Fully delocalised (ergodic) eigenstates give no advantage; fully localised ones
give no movement.

The manuscript routes networks to the quantum proposal by the frustration index F and the barrier depth βD. This
analysis asks four questions:
1. How does F relate to the IPR?
2. Does the IPR explain the measured advantage beyond what F explains, or the reverse?
3. Are our chains consistent with the bound?
4. Where does our quench sit between the ergodic and localised limits?

## 2. Objects (partition space, field-free networks)

For a network with couplings J, partition energies E_r on the representatives r < 2^(n-1) (M = 2^(n-1) partitions),
α = `proposals.layden_alpha(J, 0)`, and each γ of the manuscript's grid (`LAYDEN_GAMMAS`, six values 0.25–0.60):

- **Quench Hamiltonian.** H(γ) = (1−γ) α diag(E) + γ Σ_k X_k, split into the flip-even and flip-odd sectors
  (`proposals._sector_mixers`). The eigenvectors are V_e and V_o; W = V∘V (elementwise square).
- **Partition-space IPR.** IPR_γ(r) = [Σ_n W_e(r,n)² + Σ_n W_o(r,n)²] / 2. It lies in [1/M, 1]: equal to 1 for a
  localised partition, and to 1/M when the partition is spread evenly over all partitions.
- **Long-time proposal.** Q∞_γ = (W_e W_eᵀ + W_o W_oᵀ)/2. This is Orfi & Sels' Eq. 11 (dephased limit), lumped onto
  partition space.
    - Q∞ = mean over γ of Q∞_γ is the long-time version of our grid-averaged quench.
    - Identities checked by tests: Q∞_γ is symmetric with unit row sums; diag Q∞_γ = IPR_γ; at γ = 0 with distinct
      energies, IPR = 1 and Q∞ = I. On a small network, Q∞_γ equals the lumped long-time proposal computed in the full
      2^n space.
- **IPR averaged over γ.** IPR̄(r) = mean over γ of IPR_γ(r). Its Boltzmann average is
  ℓ_π = log₂ Σ_r π(r) IPR̄(r), at β = β̃/σ with β̃ = 1, the manuscript's temperature.
- **Gaps.**
    - δ∞ is the absolute spectral gap, on partition space, of the Metropolis chain built from Q∞.
    - δ_grid is our stored finite-time chain gap (`b1.0_quantum_lumped` in `results/e3_tierA_rows.jsonl`).
    - The mixing advantage r is computed with `e3c_analyze.r_of` (β̃ = 1). F and βD are taken from
      `results/e3_posthoc_barrier_rows.jsonl`.
- **Bounds over energy-threshold cuts.** B is the set of partitions above an energy level, and Bᶜ is the rest (ties
  kept together). For each level:
    - Λ(B) = E(B,Bᶜ)/(π(B)π(Bᶜ)) for the δ∞ chain (Orfi & Sels' Eq. 9);
    - the IPR bound is mean over γ of √(⟨IPR_γ⟩_{π_B,B} · ⟨IPR_γ⟩_{Bᶜ}) / π̄(Bᶜ), with π̄(Bᶜ) the mean probability of
      a partition in Bᶜ (their Eq. 17, on partition space).
    - Λ* and IPR* are the minima over all levels. Theory requires δ∞ ≤ Λ* ≤ IPR*.
- **Other temperatures.** β̃ = 0.5 and 2 are also computed, for the bound checks only.

## 3. Data

- All 288 ROI sub-networks of the scaling study: FC and SC, n = 7–12, 24 per size and modality. They are regenerated
  by `experiments/instances.e3_instances()` and matched to the stored rows by key.
- The four canonical-network rows are excluded, as in every Tier A analysis.
- n = 13 is excluded (cost).

## 4. Pre-specified analyses

The cells are (modality, n). There are 6 cells per modality, with 24 networks each. All tests are two-sided.

1. **P1 (F and localisation).** The stratified Spearman correlation ρ(F, ℓ_π) is the mean of the within-cell
   Spearman coefficients. Its p-value comes from 10,000 permutations of F within cells (seed 2028). The 95% interval
   comes from 2,000 bootstrap resamples of networks within cells. Computed per modality.
2. **P2 (what explains the advantage).** Stratified partial Spearman correlations, per modality:
    - ρ(r, ℓ_π | F): within each cell, the ranks of r and of ℓ_π are each regressed on the ranks of F, and the
      residuals are correlated. The cell values are averaged, and the p-value comes from permuting the ℓ_π residuals
      within cells (10,000 permutations).
    - ρ(r, F | ℓ_π): the same, with the roles of F and ℓ_π exchanged.
3. **Multiplicity.** The six p-values of P1 and P2 (2 modalities × 3) are Holm-adjusted together. "Related" or "adds
   information" is reported when the Holm-adjusted p is below 0.05, together with the sign and size of ρ.
4. **P3 (bounds; known-answer check).** δ∞ ≤ Λ* ≤ IPR* must hold on every network at every β̃, within a relative
   tolerance of 1e-9. A violation is a code error, not a result. In that case the run stops, and the fix is recorded as
   a dated amendment.
    - Tightness: median log₂(IPR*/δ∞) and log₂(Λ*/δ∞) per modality.
5. **P4 (finite time against long time).** Per modality:
    - the median and interquartile range of log₂(δ_grid/δ∞);
    - the number of networks on which our finite-time chain is faster than the long-time IPR bound (δ_grid > IPR*).
6. **P5 (distance from the ergodic limit).** Per modality, an ordinary least-squares slope of the cell medians of ℓ_π
   against n, with a 2,000-resample within-cell bootstrap interval.
    - The ergodic limit has slope −1 bit per node, since IPR = 1/M.
    - "Our quench stays away from the ergodic limit" is reported if the interval lies entirely above −1.

**Secondary (descriptive, not tested).**
- The stratified Spearman correlations of ℓ_π with βD and with r.
- The same analyses with the IPR of the ground partition, or with the uniform mean IPR, in place of ℓ_π.

## 5. Interpretation, fixed in advance

- **If P1 is significant.** The routing rule's F is, in part, a proxy for the localisation that Orfi & Sels identify
  as the limiting factor. The sign says which way.
- **If ρ(r, ℓ_π | F) is significant but ρ(r, F | ℓ_π) is not.** The IPR subsumes F, and the manuscript should say so.
- **If the reverse holds.** F carries information about the advantage that the long-time IPR does not.
- **If neither is significant.** The two measures are interchangeable at these sizes.
- **Either way.** The result is post hoc with respect to the routing rule, and it does not change any routing decision
  or prior result.

## 6. Code

- `qsr2/ipr.py`: IPR, long-time proposal and bounds.
- `tests/test_e8.py`: the known-answer tests.
- `experiments/e8_ipr_run.py`: computes the per-network quantities and writes `results/e8_ipr_rows.jsonl`.
- `experiments/e8_ipr_analyze.py`: computes P1–P5 and writes `results/e8_ipr_summary.json`.
