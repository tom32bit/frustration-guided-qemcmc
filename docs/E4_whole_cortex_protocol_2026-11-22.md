# E4 protocol: QSR 2.0 end to end on whole-cortex tilings (pre-specified 22 Nov 2026)

**Status:**

- Written after E3c and before any E4 data.
- **Sealed routing.** The routing decisions were written at 21:48:52 +06, when no E4 row existed. They are in `results/e4_routing_predictions.json` (read-only, SHA-256 61ecbf10…).
- **Hashes.** Those of this protocol, the code and the sealed routing are recorded at launch in `results/e4_prereg_hashes.txt`.
- **Role in the plan.** E4 is the end-to-end experiment behind gate G2 (scheduled for 19 Mar 2027), run early.
- **QPU.** E4 uses no QPU time.

**Question.** QSR 2.0 builds a brain-wide map of community uncertainty within each functional network. It sends only the networks flagged by the sealed rule to the quantum proposal. Does it reach a target accuracy on the whole map with fewer proposals than the best classical sampler for each network?

**Why E4 was redesigned.** The plan's original E4 sampled partitions of the whole connectome, with quantum block moves on patches. That is the regime where the method is known not to help:

- **Embedded patches.** A patch inside a whole-network partition is pinned by fields from the rest of the network, with field-to-coupling ratios of 10–18. G1 found no quantum advantage there.
- **Module splits** are mostly ferromagnetic, and G1 found no advantage there either.

The advantage was shown on standalone ROI networks: community structure within one functional network, such as the subsystems of the default network (G1c, E3 endpoint A, E3c). E4 therefore asks the application question at that level and assembles a within-network uncertainty atlas from every canonical network of the cortex. Community structure between networks is not sampled with quantum proposals. The paper states this as a limitation, and as the regime boundary found in G1.

## 1. Instances: the tilings

- **Primary: Schaefer-100, FC and SC** (HCP group averages). The canonical Yeo-7 × hemisphere networks with 3–13 parcels: 13 per modality, together covering 98 of the 100 cortical parcels.
    - **Excluded: RH_Limbic (2 parcels).** Its model has a single coupling, so σ_ROI, and hence β, is undefined, and two parcels have no community structure to estimate.
    - This was found by the pipeline test (section 3), before any E4 data existed.
- **Secondary: Schaefer-200, FC and SC.** The 7 networks with at most 13 parcels (73 of 200 parcels). The other 7 have 14–27 parcels, beyond exact computation on this machine.
- **Model.** Each network is analysed as its own field-free model: J is its own modularity matrix, h = 0, and β = β̃/σ_ROI, with β̃ = 1 primary and β̃ = 2 secondary. This is the E3 model.
- **Code:** `experiments/e4_instances.py`, which uses no random numbers.

**Primary tilings and their sealed routing** (F = frustration index; βD at β̃ = 1):

| Network | Parcels | FC: F | FC: βD | FC: routed | SC: F | SC: βD | SC: routed |
|---|---|---|---|---|---|---|---|
| LH_Limbic | 3 | 0.000 | 0.00 | no | 0.194 | 0.00 | no |
| LH_Cont | 4 | 0.069 | 2.02 | no | 0.000 | 0.00 | no |
| RH_SalVentAttn | 5 | 0.000 | 0.00 | no | 0.086 | 0.58 | no |
| LH_SomMot | 6 | 0.000 | 0.00 | no | 0.143 | 0.36 | no |
| LH_SalVentAttn | 7 | 0.002 | 0.00 | no | 0.149 | 0.38 | no |
| RH_DorsAttn | 7 | 0.048 | 0.64 | no | 0.251 | 0.65 | no |
| LH_DorsAttn | 8 | 0.094 | 2.45 | no | 0.174 | 1.09 | **yes** |
| RH_SomMot | 8 | 0.030 | 3.47 | no | 0.138 | 0.54 | no |
| RH_Vis | 8 | 0.124 | 2.24 | **yes** | 0.207 | 1.90 | **yes** |
| LH_Vis | 9 | 0.138 | 2.14 | **yes** | 0.254 | 1.63 | **yes** |
| RH_Cont | 9 | 0.191 | 6.23 | **yes** | 0.109 | 0.78 | no |
| RH_Default | 11 | 0.107 | 3.82 | **yes** | 0.132 | 0.40 | no |
| LH_Default | 13 | 0.142 | 3.17 | **yes** | 0.161 | 0.53 | no |

## 2. Routing (sealed before any E4 row)

- **Rule** (E3 amendment 13, also used in E3c): a network is routed to the quantum proposal iff F ≥ 0.10 and βD ≥ 1.
    - D is the critical depth of the single-node-move energy landscape on partition space.
    - Both F and D come from the couplings alone.
- **Routed at β̃ = 1:**
    - FC100: 5 of 13 networks (50 of 98 parcels);
    - SC100: 3 of 13 (25 parcels);
    - FC200: 6 of 7;
    - SC200: 4 of 7.
- **At β̃ = 2 (secondary only):** FC100 5, SC100 7, FC200 6, SC200 4.

## 3. Quantities (exact, partition space)

**Per network and β̃ ∈ {1, 2}** (`qsr2/roi_exact.py`):

- the exact co-assignment probabilities p_ij, and the node uncertainty u_i = mean over j of h₂(p_ij), in bits. These make up the map;
- for each chain: the spectral gap, and the exact asymptotic variance s²_ij of every co-assignment estimator.

**Chains:**

- **Classical:** local, label-swap, uniform, Swendsen–Wang, Wolff, and heat-bath. Heat-bath draws i.i.d. from the Gibbs distribution, and each draw counts as 2^(n−1) proposals.
- **Quantum:** the untuned Layden quench (G1 grid average; primary), and QAOA-MC at angle 0, the hardware circuit.
- **Noisy quantum:** the quench mixed with uniform proposals at ε = 0.2 and 0.5, a depolarising-noise model. Its Metropolis kernel is exactly the same mixture of the two Metropolis kernels.

One step of any chain counts as one proposal. That includes a whole Swendsen–Wang or Wolff cluster update, which favours the classical side.

**Validation** (`tests/test_e4.py`, 20 tests; the full suite has 95 passing):

- Means and maxima of the variances, and the gaps, equal the validated `exact_ps` path to 10⁻⁹ at n = 5–7.
- p_ij equals brute-force enumeration over all 2ⁿ labelled states (n = 2–6).
- The noise-mixture identity holds to 10⁻¹⁴.
- For every chain type, the sampler reaches the Gibbs distribution and reproduces the exact variances to within 20 %. Its output is identical on one thread or several.
- The whole pipeline (sealed routing, exact rows, runs, analysis) runs end to end on synthetic networks.

## 4. Primary endpoint E4-A (exact; per primary tiling)

- **Cost of chain X on network R:** c_X(R) = the maximum of s²_ij over the network's pairs. X needs c_X(R)/(4ε²) proposals to bring every p_ij of R to RMSE ε. ε cancels in every ratio below.
- **Classical oracle C\*(R):** the classical chain with the smallest c(R), chosen per network. It is the strongest classical comparator and is not available in practice.
- **QSR 2.0 hybrid H:** the untuned quench on routed networks, and C\* on all others.
- **ρ = Σ_R c_H(R) / Σ_R c_C\*(R)**, the hybrid's share of the proposals the classical oracle needs for the whole map.
- **E4-A passes for a tiling iff ρ < 1.** ρ ≤ 0.5 is called a *material saving*.
- Every quantity is exact, so ρ has no interval. Robustness is covered by the secondary analyses.

## 5. Secondary and descriptive (no gate)

- **Temperature:** β̃ = 2, with its own sealed routing.
- **Hardware circuit:** QAOA-MC at angle 0 on the routed networks.
- **Noise:** the depolarising-noise quench at ε = 0.2 and 0.5.
    - ε = 0.2 guarantees at least 0.8 of the ideal gap; E5 measured retention of 0.79–1.20 on hardware.
- **Routing variants:**
    - oracle routing, which takes the cheaper of quantum and C\* in each network, and the routing efficiency (1 − ρ)/(1 − ρ_oracle);
    - quantum on every network, with no routing.
- **Mean over pairs** instead of the maximum.
- **A deployable classical side:** one fixed classical chain X on every network, for each X.
- **Leave one network out:** the range of ρ.
- **ρ over the primary networks never inspected before E4** (section 8).
- **The Schaefer-200 partial tilings.**
- **Routing accuracy:** how many routed networks have c_quantum < c_C\*, and how many non-routed networks have the reverse.
- **A per-network table:** parcels, F, routing, C\*, costs, gaps, and share of the classical cost.

## 6. E4-B: end-to-end runs (primary tilings)

- **Arms.**
    - H (QSR 2.0): the quench on routed networks, C\* elsewhere.
    - C: C\* everywhere.
    - Where both arms use the same chain, the run is shared.
- **Budget.** Each run uses its own chain's exact requirement for RMSE ε = 0.02 on every co-assignment of its network:
    - N = ⌈max s²/(4ε²)⌉ recorded steps, after a burn-in of ⌈10/gap⌉ steps that counts as proposals;
    - heat-bath: K = ⌈max var/(4ε²)⌉ draws, counted as K·2^(n−1) proposals.
- **Replicates.** R = 200 per run. Each starts from the exact ground-state partition, the warm start a modularity optimiser would give. Seeds are `SeedSequence([20261127, tiling, network, chain])`.
- **Compute cap:** 3 × 10¹⁰ sampler steps.
    - Above the cap, R is lowered, to a minimum of 50.
    - If that is not enough, ε is raised to the first of 0.03, 0.04 and 0.05 that fits (0.05 if none fits).
    - The values used are recorded in every row.
- **Checks**, per tiling and arm (pairs with min(p, 1 − p) < 0.01 enter neither V1 nor V2):
    - **V1 exactness:** every pair's standardised bias z = √R (mean estimate − p)/RMSE_pred stays below the two-sided Bonferroni threshold at family-wise 0.001.
    - **V2 calibration:** the median over pairs of κ = realised RMSE / predicted RMSE lies in [0.85, 1.15].
    - **V3 target reached:** on every network, the binding pair (the one with the largest predicted variance) has κ ≤ 1.3.
- **ρ_B:** total proposals of H over those of C, including burn-in.
- **Map:** for every parcel, the exact u_i and the hybrid's estimate (mean ± SD over replicates), in `results/e4_parcel_map.csv`.

## 7. Decision rule and outcomes (fixed in advance)

- **End-to-end pass for a tiling:** E4-A passes, and V1–V3 hold for both arms.
- **G2 end-to-end criterion:** met iff the end-to-end pass holds for at least one primary tiling.
- **A failed check** is investigated before anything is reported.
    - If it is a code error, the fix is recorded as an amendment, and E4-B is rerun in full. E4-A is also rerun if shared code changed.
    - Otherwise the failure stands.

| Result | Statement allowed in the paper |
|---|---|
| End-to-end pass on both tilings, with ρ ≤ 0.5 on both | Across 98 of 100 cortical parcels, in both modalities, QSR 2.0 maps within-network community uncertainty to a target accuracy with at most half the proposals of the best classical sampler for each network. It uses the quantum proposal only where the sealed rule routes it |
| End-to-end pass on one or both tilings (other cases) | The same statement with the measured ρ, for each passing modality. G2's end-to-end criterion is met |
| E4-A fails on both tilings (ρ ≥ 1) | Routing by the sealed rule does not reduce the whole-map cost. G2's end-to-end criterion is not met. Per the plan, the paper reports E1–E3 and E5 and drops the E4 claims |
| E4-A passes, but a check fails without a code error | The exact saving is not confirmed by finite runs. G2's end-to-end criterion is not met |

## 8. What was known before E4 (disclosure)

- **Inspected before:** per-network exact results for:
    - the Schaefer-100 networks of 4–10 parcels (gaps and quantum-win counts in G1, G1b and G1c), i.e. 10 of the 13 primary networks per modality;
    - the Schaefer-200 networks of 6 and 11 parcels (G1; E3 Tier A).
- **Computed but not inspected:** the 13-parcel networks: FC100 and SC100 LH_Default, and FC200 and SC200 LH_Cont, LH_DorsAttn and RH_DorsAttn.
    - They were computed in the E3c canonical stage, which was still running when E4 was frozen.
    - That stage is analysed only after the E4 freeze.
- **Never computed:** LH_Limbic (3 parcels) and RH_Default (11) of Schaefer-100.
- **Never computed for any network:** per-pair maximum variances, whole-tiling costs, routing-weighted sums, or end-to-end runs.
- **Reported next to the primary ρ:** ρ over the three primary networks not inspected before (LH_Limbic, RH_Default, LH_Default), descriptive. Most of the classical cost is expected to sit in the largest networks.

## 9. Compute

- **E4-A:** about 2.5 h in all.
    - The 8 networks with 13 parcels take about 15 min each; E3c measured 12 min with fewer chains.
    - The other networks take about 15 min together.
    - Peak memory is about 2.5 GB, one network at a time.
- **E4-B:** at most 3 × 10¹⁰ sampler steps. The measured rate on 2 threads was 2 million (dense kernel) to 13 million (local) steps per second, so it takes at most about 2–3 h on 4 cores.
- **Launch:** after the E3c canonical stage finishes (about 23:45 +06, 22 Nov 2026), so that the two runs do not compete for memory. E4-A, E4-B and the analysis then run in sequence.

## 10. Files

- **Code:** `qsr2/roi_exact.py`, `qsr2/sampler_numba.py`; `experiments/e4_instances.py`, `e4_routing_predict.py`, `e4_exact.py`, `e4_runs.py`, `e4_analyze.py`; `tests/test_e4.py`.
- **Outputs:** `results/e4_routing_predictions.json` (sealed), `e4_exact_rows.jsonl`, `e4_run_rows.jsonl`, `e4_summary.json`, `e4_roi_table.csv`, `e4_parcel_map.csv`, `e4_run.log`.
- **Hashes:** `results/e4_prereg_hashes.txt`.
