# E3 scaling protocol: does the ROI-network advantage grow beyond n = 10? (pre-specified)

**Status:** FROZEN 21 Nov 2026, together with the G1b report (`G1b_hardening_report_2026-11-21.md`). The draft was written while G1b ran. The amendments below were made after G1b results and **before any E3 data**. Any later change will be dated here.

**Question.** On P3 ROI sub-networks (field-free, physical β = β̃/σ_ROI, β̃ = 1 primary), does the untuned Layden quench need fewer steps than the best classical sampler to estimate the community co-assignment probabilities p_ij = (1 + ⟨z_i z_j⟩)/2? And does the gap between them widen with n?

## Instances

- **Data:** P3 ROI node sets drawn exactly as in G1 (ambiguous border sets and within-module sets, same seeded generator) from FC-200, SC-200 and FC-400.
- **Canonical ROIs:** the canonical Yeo-7 × hemisphere networks of Schaefer-200 with 11–20 parcels. The 400-parcel label file will be downloaded and added if its ROIs fall in range.
- **Count:** 12 random instances per (dataset, n), plus every canonical ROI in range.

## Tier A: exact gaps, n = 11–12

- **Chains:** the same quantum proposal as G1, against local, label-swap, uniform, Swendsen–Wang, Wolff and the per-instance classical oracle. PT is omitted: the 2-replica state space is 2²⁴.
- **Primary endpoint A:** Δk over n = 4–12, pooling the G1 rows (n = 4–10) with Tier A, against the G1b set (a′). Pass iff the 95 % bootstrap CI is above 0 for each P3 modality that passed G1b.

## Tier B: simulated chains, n = 12, 14, 16 (18 and 20 if cluster time is available)

- **Reference:** exact ⟨z_i z_j⟩ by enumeration of all 2ⁿ states (n ≤ 20 fits in memory).
- **Metric T_ε:** the smallest number of steps T at which the RMSE over pairs of the running estimate of ⟨z_i z_j⟩ is ≤ ε = 0.05. The RMSE is averaged over 32 independent chains from uniformly random starts, with no burn-in discarded. A chain that has not reached ε by its budget (10⁵ steps) is recorded as censored at the budget.
- **Quantum chain:** at each step, draw γ and t uniformly from the G1 grid. Evolve the current basis state under H = (1 − γ)αH_C + γΣX by sparse Krylov exponentiation. Sample s′ from the output distribution and accept with Metropolis–Hastings. This is the same chain whose exact gap G1 measured.
- **Classical chains:**
    - local Metropolis–Hastings;
    - label-swap mixture;
    - Swendsen–Wang;
    - Wolff;
    - parallel tempering with R ∈ {2, 4, 8} replicas, geometric ladders from β to rβ with r ∈ {1/3, 1/10, 1/30} and local + label-swap moves, oracle-tuned per instance.
- **Cost accounting (primary):** one proposal on one replica = 1 step, as in G1 and G1b.
- **Cost accounting (sensitivity):** one full sweep of n single-spin proposals, or one PT round over all replicas, = 1 step. This favours classical by a factor of up to n·R.
- **Primary endpoint B:** the slope of log₂ T_ε against n, for quantum versus the classical oracle, with a bootstrap CI over instances. Pass iff the quantum slope is smaller, CI excluding 0, for each P3 modality that passed G1b.
- **Prospective routing test:** among Tier A ∪ Tier B instances, the share where quantum beats the classical oracle must be higher for routed instances (F ≥ 0.05) than for the rest. One-sided Fisher exact test, p < 0.05. This is the first test of the G1 post-hoc criterion on new data.

## Outcomes

| Endpoints | Consequence |
|---|---|
| A and B pass | Scaling section of the main paper; proceed to E4 (uncertainty maps) and E5 (hardware) |
| A passes, B fails | Report the advantage as small-n only; G2 becomes a fallback decision |
| A fails | Fallback characterisation paper |

## Compute

- This machine (8 cores, 7 GB RAM) runs Tier A, and Tier B to n = 16.
- n = 18–20 needs a cluster node: the quantum step costs about 1 s at n = 20.

## Amendments (21 Nov 2026, after G1b, before any E3 data). These override the sections above where they conflict.

1. **Fresh instances.** Every E3 instance is generated with a new seed (20261122), not the G1 seed, and Tier A covers **n = 7–12**. All E3 tests are then out-of-sample, including the routing test. The G1 rows are not pooled into any E3 endpoint.
2. **Comparator set** = the G1b set: local, label-swap, uniform, heat-bath (cost 2ⁿ), Swendsen–Wang, Wolff, and the per-instance classical oracle over all of these. PT is dropped beyond n = 8: oracle-tuned PT-2 was never the per-instance best chain in 858 G1b instances, and its exact state space at n ≥ 9 exceeds this machine's memory.
3. **Primary endpoint A (replaces "Δk over n = 4–12").** For each modality (FC and SC), with r = log₂(gap_quantum / gap_oracle) per instance, pass iff both hold:
    - **(i)** the slope of mean r against n over n = 7–12 is greater than 0, 95 % bootstrap CI excluding 0;
    - **(ii)** mean r at n = 12 is greater than 0, CI excluding 0.

    Reason: G1b showed that a minimum over per-chain exponents rewards a chain whose gap is non-monotone at small n (Wolff on FC: shallow slope, yet 20× worse at n = 10). The per-instance oracle ratio has no such artefact. The G1b min-exponent rule over n = 7–12 is still reported, as a secondary endpoint.
4. **FC is decided here.** FC enters the main claim only if it passes endpoint A. SC is confirmatory. If SC fails endpoint A, the outcome table applies with "A fails".
5. **Prospective routing test** (now genuinely out-of-sample): on the fresh Tier A instances, the quantum win share against the oracle must be higher for F ≥ 0.05 than for F < 0.05 (one-sided Fisher exact test, p < 0.05). Secondary: the same comparison with a threshold of F ≥ 0.10.
6. **Tier B compute (measured).** The simulated quench step with SciPy `expm_multiply` costs 0.57 s per chain-step at n = 12, 2.8 s at n = 14 and 14 s at n = 16 (one core). Swendsen–Wang costs 2.5e-4 s.
    - Tier B above n = 12 needs either a batched Chebyshev propagator with an in-place X-mixer (target: more than 10× faster; it must pass `tests/test_chains.py` first) or a cluster or GPU node.
    - Tier B runs at n = 12 on this machine. n = 14–16 waits for the faster propagator.
7. **Partition-space metric (21 Nov 2026, before any E3 data; see `G1c_partition_space_protocol_2026-11-21.md`).** Configurations z and −z are the same bipartition, so every gap in endpoint A is the **partition-space (lumped) gap**, with heat-bath at per-cost gap 2^−(n−1). The full-space gaps of the frozen text are still computed and reported, as a secondary endpoint. The exact asymptotic variance of the co-assignment estimators, the exact n ≤ 12 version of the Tier B metric, is a further secondary endpoint.
8. **Pools (clarification before data).** Endpoint A pools the random ROI node sets of each modality: FC = FC-200 + FC-400, SC = SC-200. Canonical anatomical ROIs exist only at a few sizes, so they are reported separately, per instance.
9. **No cluster or GPU is available** (confirmed by the user, 21 Nov 2026). Tier B is redesigned around this machine once the faster propagator is benchmarked, and the design is recorded here as amendment 10 before any Tier B data.

11. **SC-400 added (21 Nov 2026, before any E3 data).** The same ENIGMA source also provides the Schaefer-400 structural matrix and the 400-parcel labels; both were downloaded and checked (400 × 400, symmetric; 28 small negative entries, clipped by the loader). Each modality now has two parcellations: FC = FC-200 + FC-400, SC = SC-200 + SC-400. Tier A has 12 instances (6 border sets, 6 within-module sets) per (dataset, n), so 288 random instances, plus the 4 canonical ROIs with 7–12 parcels (Schaefer-200 FC/SC; every Schaefer-400 canonical ROI has at least 13 parcels).
12. **Same comparator at every n (before any E3 data).** The endpoint-A oracle excludes PT at every n. Including PT only at n = 7–8 would strengthen the classical side at small n alone and so tilt the slope in endpoint (i) towards quantum. PT-2 partition-space gaps (6 ladders) are still computed at n = 7–8 and reported, as a check that PT is not the best classical chain there.
10. **Tier B design for this machine (21 Nov 2026, after benchmarking, before any Tier B data).**
    - **Propagator.** Batched Chebyshev with per-(γ, t) truncation and a Numba-compiled fused kernel, tested against `expm_multiply` (1e-10) and against the exact one-step kernel.
    - **Measured cost** per chain-step, with Tier A sharing the machine: 0.053 s (n = 12), 0.185 s (n = 14), 1.5 s (n = 16).
    - **Instances.** n = 12: 16, the first 2 Tier A instances per (dataset, kind), so the simulated metric can be checked against the exact asymptotic variance on the same instances. n = 14: 16 (fresh seed 20261123, 2 per dataset and kind). n = 16: 8 (seed 20261123, 1 per dataset and kind).
    - **Chains and budget.** C = 8 chains (C = 4 at n = 16), T = 1,000 quantum steps per chain, from uniform random starts. Every classical chain uses the same starts and runs n·T steps; checkpoints are recorded at T (per-step accounting) and at n·T (sweep-cost sensitivity).
    - **Metric.** ρ = 2·log₂(RMSE_oracle / RMSE_quantum) at T. RMSE is over pairs, for the pooled co-assignment estimate. The oracle is the minimum RMSE over local, label-swap, uniform, Swendsen–Wang, Wolff and i.i.d. heat-bath (cost 2ⁿ⁻¹ per sample). This approximates log₂ of the ratio of steps needed for equal accuracy.
    - **Primary endpoint B, per modality.** Pass iff both hold:
        - (i) the slope of mean ρ over n ∈ {12, 14, 16} is greater than 0, bootstrap CI over instances excluding 0;
        - (ii) mean ρ at n = 14 (8 instances per modality) is greater than 0, CI excluding 0.
    - **Descriptive.** Mean ρ at n = 16 (4 instances per modality); the sensitivity value of ρ at n·T; Spearman correlation and mean difference between simulated ρ and the exact log₂ asymptotic-variance ratio at n = 12.
    - **This supersedes** the C = 32, T_ε, 10⁵-step design above, which would need about 4 machine-days per instance at n = 16.
13. **Two-factor routing rule, tested prospectively on Tier B (22 Nov 2026, after the Tier A results, before any Tier B data at n = 14–16).**
    - **Origin (post hoc, Tier A).** The analysis is in `experiments/e3_posthoc_barrier.py`. On SC the per-instance r is bimodal, and the energy barrier that single-node moves must cross separates the two groups better than F does. At n = 10–12, instances with F ≥ 0.10 and βD ≥ 1 were quantum wins in 70 of 71 cases; instances meeting neither condition, in 2 of 15.
    - **D** is the critical depth of the single-node-move energy landscape on partition space. It is computed exactly by union-find and checked against a brute-force bottleneck search. β = 1/σ_ROI (β̃ = 1).
    - **Rule.** Predicted quantum win ⇔ F ≥ 0.10 and βD ≥ 1. Both thresholds are round values fixed from Tier A; F = 0.10 is the existing secondary threshold of amendment 5.
    - **Sealed predictions.** Computed from instance data alone (couplings, σ, F) for the 24 fresh Tier B instances (n = 14: 16; n = 16: 8) by `experiments/e3_tierB_routing_predict.py`. They are written to `results/e3_tierB_routing_predictions.json` with a timestamp and the number of n = 14–16 Tier B rows existing at that time, which must be 0.
    - **Test (secondary, not a gate).** The share with ρ > 0 (at T, per-step accounting) among predicted wins against the rest, one-sided Fisher exact test, p < 0.05; mean ρ per class is also reported. The n = 12 Tier B instances are Tier A instances and are excluded.
    - **Power.** With 24 instances the test can detect only a large separation, of the size seen in Tier A. A null result does not refute the rule.
