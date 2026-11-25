# E3c protocol: does the quantum lead keep growing from n = 12 to n = 13? Exact, within-network (pre-specified 22 Nov 2026)

**Status:**

- Written after the E3 decision ("A passes, B fails") and before any n = 13 data.
- A design pilot (n = 10 → 11) was run first to set the power statement. It is reported in section 5 and enters no endpoint.
- The sealed routing predictions for the n = 13 instances were written at 11:28:44 +06, with 0 study rows existing (`results/e3c_routing_predictions.json`, read-only).
- E3c cannot change the E3 outcome. Its result is reported next to it.

**Question.** When a brain sub-network gains one node, does the quantum proposal's lead over the best classical sampler grow? The test is exact (no simulation noise), at the step from n = 12 to n = 13, one size beyond E3 Tier A.

**Why a paired design.** Endpoint B failed because between-network variation (per-instance SD of r 2.3–2.5 bits at n = 12) and simulation error swamped the slope. Growing each network by one node and comparing it with itself removes the between-network part. In the pilot this cut the SD of the per-step change to 1.2 bits on FC.

## 1. Instances

- **Parents:** all 48 Tier A random ROI node sets at n = 12: FC-200, FC-400, SC-200 and SC-400, each with 6 border sets and 6 within-module sets. Their exact n = 12 values are the Tier A rows.
- **Growth:** each set gains one node, drawn uniformly from the pool it was drawn from, with seed 20261124:
    - for a border set, the 24 most ambiguous nodes of its community pair;
    - for a within-module set, its module.
- **Reproducibility:** `experiments/e3c_instances.py` replays the Tier A generator exactly (asserted node for node) and records the parent and the added node. All 48 sets can grow (no pool is exhausted).
- **Optional descriptive extension, run only after the 48 primary instances:** the twelve canonical Yeo-7 × hemisphere ROIs with exactly 13 parcels:
    - Schaefer-100: LH_Default;
    - Schaefer-200: LH_Cont, LH_DorsAttn, RH_DorsAttn;
    - Schaefer-400: LH_Limbic, RH_Limbic;
    - each in FC and SC.

    They have no parent, so r(13) is reported per instance.

## 2. Quantities (exact, partition space, β = β̃/σ_ROI)

- **Gaps and variances.** At β̃ = 1 (primary) and β̃ = 2, for local, label-swap, uniform, the untuned Layden quench (G1 grid average), Swendsen–Wang, Wolff and QAOA-MC angle 0:
    - the partition-space gaps;
    - the exact asymptotic variances of the co-assignment estimators.
- **Code.** `qsr2/exact_ps.py` works on the 2¹² bipartitions directly. `tests/test_exact_ps.py` (19 tests) checks it against the validated Tier A code path:
    - every gap and variance agrees to a relative 10⁻⁹ at n = 5–7;
    - the new parallel Swendsen–Wang/Wolff kernel agrees to 10⁻¹² at n = 3–8.
- **Metric.** r = log₂(gap_quantum / gap_oracle). The oracle is the best of local, label-swap, uniform, Swendsen–Wang, Wolff, and heat-bath at per-cost gap 2^−(n−1). This is exactly the E3 endpoint-A definition, with PT excluded.
- **Per pair:** Δr = r(13) − r(12).

## 3. Primary endpoint C (per modality: FC = FC-200 + FC-400, SC = SC-200 + SC-400; 24 pairs each)

1. **(i)** Mean Δr > 0, with the 95 % bootstrap CI over pairs excluding 0.
2. **(ii)** Mean r(13) > 0, with the CI excluding 0.

- **Bootstrap:** 10,000 resamples, seed 41, percentile intervals.
- **Pass rule:** C passes for a modality iff (i) and (ii) both hold.
- **Decisive modality: FC.** It has adequate power; SC does not (section 5). As in E3 amendment 4, FC decides and SC is confirmatory. A failure on SC alone is reported as underpowered, not as evidence against growth.

## 4. Secondary and descriptive (no gate)

- **Steps to accuracy:** the exact asymptotic-variance ratio, log₂(s²_best classical / s²_quantum), at n = 13, and its paired change.
- **Temperature:** β̃ = 2, r(13) and Δr.
- **Hardware circuit:** QAOA-MC angle 0, r(13) and Δr.
- **Third test of the two-factor routing rule** (F ≥ 0.10 and βD ≥ 1). The predictions are sealed: 27 of 48 are predicted wins, 14 FC and 13 SC.
    - Compare the share with r(13) > 0 for predicted wins against the rest, one-sided Fisher exact test.
    - Report Δr within the predicted-win class.
- **Pooled:** both modalities together, 48 pairs: mean Δr with its CI.
- **Mechanism:** the paired change of log₂ gap per chain, showing which samplers slow down as the network grows.

## 5. Design pilot (n = 10 → 11; run before this protocol, not part of any endpoint)

**Design.** The same growth rule (seed 20261126) applied to the 48 Tier A n = 10 sets, computed with the same code (`results/e3c_pilot_rows.jsonl`, `e3c_pilot_summary.json`).

| | FC | SC |
|---|---|---|
| Mean Δr [95 % CI] | +0.75 [+0.31, +1.25] | +0.74 [−0.08, +1.53] |
| SD of Δr | 1.23 | 2.02 |
| Pairs with Δr > 0 | 79 % | 67 % |
| Change of log₂ gap per added node: quantum | −0.12 | −0.31 |
| Change of log₂ gap per added node: local / uniform / Swendsen–Wang / Wolff | −0.91 / −0.97 / −1.84 / −1.83 | −1.05 / −1.30 / −2.25 / −2.11 |

**Power at n = 12 → 13** (24 pairs; one-sided α = 0.025, the "CI excludes 0" rule):

- The per-network SD of r grew by 1.3× from n = 10 to n = 12 in Tier A, so the SD of Δr is assumed to be 1.23–1.65 (FC) and 2.02–2.6 (SC).
- For an expected Δr equal to the pilot's value (0.75) or the Tier A slope (FC 0.95, SC 0.46 per node):
    - **FC: power ≈ 60–85 %**;
    - **SC: power ≈ 15–45 %**.

## 6. Compute

- Measured on a throwaway n = 13 node set, not a study instance: 15.4 min per instance (one process, 4 BLAS threads), peak memory 2.0 GB.
- The 48 primary instances take about 12.5 h, run one at a time, alternating datasets so that a partial run stays balanced.
- The canonical extension adds about 3 h.

## 7. Outcomes (fixed in advance)

| Result | Statement allowed in the paper |
|---|---|
| C passes on FC (with or without SC) | Exactly and within networks, the quantum lead keeps growing from n = 12 to 13. Growth is shown through n = 13, not beyond. E3 stays "A passes, B fails" |
| (ii) passes on FC, (i) fails | The lead persists at n = 13 (exact), but growth beyond n = 12 is not shown |
| C fails on FC | Growth beyond n = 12 is not shown by either the exact or the simulated method. The claim stays at n = 7–12 |

## 8. Files

- **Code:** `experiments/e3c_instances.py`, `experiments/e3c_run.py`, `experiments/e3c_routing_predict.py`, `experiments/e3c_analyze.py` (written before any study row); `qsr2/exact_ps.py`, `qsr2/cluster_numba.py`.
- **Outputs:** `results/e3c_rows.jsonl`, `e3c_canon_rows.jsonl`, `e3c_study_summary.json`.
- **Hashes:** those of this protocol and of the analysis inputs are recorded at launch in `results/e3c_prereg_hashes.txt`.
