# G1c protocol: re-test G1b on partition space (pre-specified 21 Nov 2026)

**Status:** written before any partition-space (lumped) gap was computed. The code is `experiments/g1c_partition_space.py` and `experiments/g1c_analyze.py`.

## The issue

With no external field (P2, P3), the configurations z and −z describe the **same bipartition**, because community labels are arbitrary.

- G1 and G1b measured spectral gaps on the 2ⁿ labelled states. Those gaps include the slowest mode of the "label flip" z → −z.
- That mode is irrelevant for co-assignment p_ij = (1 + ⟨z_i z_j⟩)/2, which is invariant under the flip.
- So a chain can be penalised, or flattered, by a mode that the community-detection question never uses. Local moves are the extreme case: their tiny G1 gaps (about 2⁻²³ at n = 10) are probably this mode.

Every kernel used here commutes with the global flip F. Each one is therefore **exactly lumpable** onto the 2ⁿ⁻¹ bipartitions, and the lumped spectrum is the F-even part of the full spectrum. This holds for local, label-swap, uniform, the quantum quench, QAOA-MC, Swendsen–Wang, Wolff, and parallel tempering (lumped per replica). The unit tests in `tests/test_lumping.py` verify it, including the eigenvalue inclusion.

The lumped ("partition-space") gap is the label-invariant quantity. It is the one that governs the autocorrelation of every co-assignment estimate.

## What is recomputed

The same 858 field-free instances, rebuilt from seed and matched to G1 as in G1b, at β̃ ∈ {0.5, 1, 2}:

1. Partition-space gaps of:
    - local, label-swap and uniform;
    - the quantum quench (grid average, untuned, as in G1);
    - Swendsen–Wang and Wolff;
    - PT-2 oracle-tuned (6 ladders; n ≤ 8; β̃ ∈ {1, 2});
    - PT-3 oracle-tuned (3 ladders; n ≤ 6; β̃ = 1).
2. The full-space quantum gap again, as a determinism check against G1.
3. Exact asymptotic variances of all n(n−1)/2 co-assignment estimators for each single-chain kernel, from the fundamental matrix. Secondary.
4. Partition-space gaps of the palindromic QAOA-MC for each of its 18 grid angles at β̃ = 1. Secondary; used later to choose hardware angles.

On partition space the exact block heat-bath costs 2ⁿ⁻¹, so its per-cost gap is 2^−(n−1) and its exponent is still k = 1.

## Decision rule (inherited unchanged from G1b)

The G1b rule is applied with every gap replaced by its partition-space version, the classical oracle recomputed from them, and heat-bath at 2^−(n−1):

- **(a′)** n = 4–10;
- **(b′)** n = 4–8 with PT-2;
- β̃ = 1, for P3_roi_FC and P3_roi_SC.

**Interpretation:**

- A family that passes G1c carries the G1b conclusion over to the label-invariant metric.
- A family that passed G1b but fails G1c had an advantage that was an artefact of the label-flip mode. That is a retraction, and the reports must say so.
- Also reported, descriptively, as in G1b §3: the ratio metric r = log₂(gap_quantum / gap_oracle) by n, and the asymptotic-variance ratio.

## Consequence for E3

Partition-space gaps become the primary E3 metric. This is amendment 7 of the E3 protocol, dated before any E3 data.
