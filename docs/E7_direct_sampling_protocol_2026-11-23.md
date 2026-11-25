# E7 protocol: can the quantum device be sampled directly for partition uncertainty, or is the Metropolis step needed? (pre-specified 23 Nov 2026)

**Status:**

- **Timing.** Written before any E7 data existed.
    - A smoke test on models outside the study (section 10) fixed the run settings.
    - It printed only run times and numerical diagnostics, never a bias floor.
- **Hashes.** Those of this protocol, the code and the tests are recorded at launch in `results/e7_prereg_hashes.txt`.
- **Launch.** 23 Nov 2026, on the laptop. E7 uses no QPU time.

**Question.** QSR 2.0 uses the quantum circuit as a *proposal* inside Metropolis–Hastings (QeMCMC). A referee will ask the obvious question: why not use the device *directly*? Run QAOA or a quantum annealer many times and take the measured partitions as the ensemble.

- **Why it matters.** With unlimited shots, a direct sampler converges to its own output distribution, not to the Gibbs target.
- **What E7 measures.** How far that output distribution is from the target, on the paper's brain networks, compared with QeMCMC on the same hardware, which is exact.
- **What it folds in.** E7 includes the audit's "Pivot B": is the low run-to-run variance reported for quantum-annealer brain partitions genuine stability, or sampling bias?

## 1. Networks

- **Primary: the whole cortex at Schaefer-100.** These are the E4 primary tilings: 13 canonical Yeo-7 × hemisphere ROIs per modality (FC and SC), with 3–13 parcels each.
    - Together they cover 98 of the 100 parcels, giving 26 networks.
- **Secondary: replication, 18 networks.**
    - The E4 Schaefer-200 tilings: the 7 ROIs with at most 13 parcels, per modality.
    - The E3c canonical 13-parcel ROIs of Schaefer-400: LH and RH Limbic, FC and SC. Every other E3c canonical ROI is already in an E4 tiling.
- **Model.** Exactly as in E4 and E3c: each network is its own field-free own-modularity Ising model, with β = β̃/σ_ROI. β̃ = 1 is primary and β̃ = 2 is secondary.
- **Code:** `experiments/e7_instances.py`. It uses no random numbers.

## 2. Target

- **What is computed.** The exact Gibbs co-assignment probabilities p_ij, and the node uncertainty u_i = mean over j of h₂(p_ij) in bits (the E4 definition).
- **How.** By enumerating all 2ⁿ⁻¹ bipartitions.
- **Check.** They must equal the stored E4 rows (part of gate E7-K).

## 3. Direct samplers

Every sampler is evaluated **exactly, with unlimited shots**. The full output distribution comes from state-vector simulation or enumeration, on ideal, noiseless devices (`qsr2/direct.py`).

| Sampler | What it is | Settings |
|---|---|---|
| `optimiser` | Returns the exact ground-state bipartition every time: an ideal annealer or exact solver (cf. the "rather deterministic" D-Wave outputs of Wierzbiński et al. 2023) | — |
| `greedy` | Classical multi-start steepest single-flip descent from uniformly random starts: the classical analogue of repeated optimiser runs | exact over all starts |
| `anneal_T` | Coherent transverse-field annealing, H(s) = −(1 − s)ΣX + s·αH_C with s = t/T, started from \|+⟩ⁿ and measured at s = 1 | T ∈ {1, 3, 10, 30, 100}; Strang splitting, dt = 0.01 |
| `qaoa_energy_p` | QAOA from \|+⟩ⁿ with angles that minimise ⟨αH_C⟩: the standard optimiser use | p = 1–4 |
| `qaoa_kl_p` | QAOA with angles that minimise KL(P_θ ‖ π_β) = β·F_θ + log Z. This is variational Gibbs sampling at the target temperature. It needs the output entropy, which is computable here only because n ≤ 13 | p = 1–4, per β̃ |
| `qaoa_oracle_p` | QAOA with angles that minimise the worst-pair co-assignment error against the exact answer. **Not deployable:** it is the best that any angle choice can do at that depth | p = 1–4, per β̃ |

- **Scale.** α = `proposals.layden_alpha`, the scale used by every quantum proposal in this project.
- **Deployable** means the first five rows, 15 configurations in all. **Oracle** means the last row.
- **Why the comparison favours direct sampling.**
    - The devices are ideal and noiseless.
    - Angles are optimised on exact expectation values, with no shot noise, and the optimisation cost is not counted.
    - Shots are unlimited.
    - The best of the 15 configurations is chosen for each network with hindsight.
    - Depth up to 4 exceeds that of the QeMCMC hardware circuit (QAOA-MC: 3 mixer and 2 phase layers).
- **Angle search** (`direct.optimise_qaoa`):
    - **Depth 1:** a 24 × 12 grid over one period of (γ, b); the 4 best points are refined.
    - **Depth p ≥ 2:** starts are the depth p−1 optimum padded with an idle layer (so the optimum cannot get worse with depth) and its INTERP extension (Zhou et al. 2020).
    - **Cross-seeds:** the energy optimum at the same depth, for KL. At β̃ = 2, the β̃ = 1 KL optimum too. For the oracle, the energy and KL optima at the same depth.
    - **Random starts:** 24, refined by BFGS with numerical gradients.
    - **Second set:** another 24 independent random starts. They serve as a convergence diagnostic and are part of the result, which is the best over both sets.
    - **Oracle:** BFGS on a smooth surrogate (the L₁₆ norm of the errors). Then Nelder–Mead on the exact worst-pair error, started from the 3 best surrogate solutions, the padded depth p−1 result and the cross-seeds. The reported oracle value is exact.
    - **Seeds:** `numpy.random.default_rng([20261130, network index, objective, 10·β̃, stream])`.

## 4. QeMCMC reference (exact)

- **What it is.** The same hardware used as a *proposal* with Metropolis acceptance:
    - QAOA-MC at angle 0, the hardware circuit of E5;
    - the Layden quench, the main quantum proposal.
- **Bias.** The stationary distribution is exactly π_β, whatever the circuit (Propositions 4–5). So the infinite-run bias is zero.
- **Calls to reach ε.** The E4-B design: ⌈10/gap⌉ burn-in steps plus ⌈max_ij s²_ij / (4ε²)⌉ recorded steps, one circuit call each.
    - Gaps and asymptotic variances come from the E4 rows, or the E3c rows for Schaefer-400.
    - E4-B confirmed this finite-run behaviour against replicated runs.
- **Checks on every network** (part of gate E7-K):
    - The QAOA-MC kernel is rebuilt. Its rows must sum to 1 and π_β P = π_β must hold to within 10⁻¹⁰.
    - For n ≤ 11, its gap and maximum s² are recomputed. They must equal the stored values to a relative 10⁻⁸.
    - Both kernels must have a gap > 0, so they are ergodic and π_β is their only stationary distribution.

## 5. Metrics (per network, sampler and β̃)

- **Floor** = max_ij |p_ij(sampler) − p_ij|. This worst-pair error remains however many shots are taken. The target is **ε = 0.02**, the E4 precision.
- **Calls to ε** for a direct sampler: the fewest shots S with bias² + p(1−p)/S ≤ ε² for every pair. It is ∞ if the floor is ≥ ε.
- **Distribution distances:** the RMS error over pairs; TV(P, π); KL(P‖π) in bits; KL(π‖P), which is ∞ if the sampler misses partitions.
- **Fairness.**
    - The **near-degenerate set** G holds the partitions whose Gibbs probability is within a factor of 10 of the most probable one.
    - The **suppression ratio** is max/min over G of P(x)/π(x). It equals 1 for a sampler that is faithful to the Boltzmann weights.
    - This generalises the fair-sampling measure of Mandrà et al. (2017) and Könz et al. (2019) from exactly degenerate ground states to the near-degenerate set weighted by the Gibbs distribution.
- **Node uncertainty under the sampler:** u_i, Δu_i = u_i(sampler) − u_i, and their means over nodes.
- **Precise-but-wrong index:** the worst pair's |bias| divided by its run-to-run SD at 1,000 shots per run. Wierzbiński et al. (2023) reported SD ≈ 0 over 100 runs.

## 6. Endpoints

- **E7-K, gate (known answers).** On every network:
    - every direct output distribution is normalised and flip-symmetric to within 10⁻¹⁰;
    - the annealing Trotter check holds, max_ij |p(dt) − p(2dt)| ≤ 10⁻³;
    - the target p equals the E4 rows to within 10⁻¹⁰;
    - the QeMCMC checks of section 4 pass;
    - and all tests in `tests/test_e7.py` pass at launch.
- **E7-A, primary.** On the 26 primary networks at β̃ = 1, the **best deployable floor exceeds ε on at least 14** (a majority).
    - The one-sided binomial p-value against ½ is reported descriptively. The tiling is a census of the cortex at this parcellation, not a random sample.
- **E7-B, co-primary.** The same criterion for the **oracle QAOA floor**, the minimum over p = 1–4.
- **Inference** is conditional on these networks. The secondary set is a replication.

## 7. Sensitivity re-search (pre-specified)

- **Who gets it.** Every primary network whose best deployable floor lies in (ε, 1.5ε].
- **What is repeated.** Its energy and KL QAOA searches, at every depth, with 64 fresh random starts plus the main-run optima as starts (`experiments/e7_sensitivity.py`).
- **Recount.** E7-A is counted again with the re-searched floors.
- **Rule.** A re-search can only lower floors. If it turns E7-A from pass to fail, the paper uses the failure.

## 8. Secondary and descriptive (no gate)

- **Replication:** E7-A and E7-B counts on the 18 secondary networks (a majority is 10).
- **β̃ = 2:** both endpoints on both sets.
- **Per-sampler table:** floors (median, IQR, share above ε), TV, KL, suppression ratio, P(ground state), Δu and the precise-but-wrong index.
- **Trends:** QAOA floors against depth (p = 1–4) and annealing floors against anneal time.
- **Calls:** QeMCMC calls to ε for QAOA-MC and the quench on every network, and direct calls where they are finite.
- **Noise:** depolarised direct samplers, P → (1 − η)P + η·uniform with η ∈ {0.2, 0.5} (the E4 noise levels; uniform noise moves every p_ij towards ½). Reported as the number of networks whose best deployable floor exceeds ε. QeMCMC stays exact under symmetric noise; E4 measured its cost.
- **Uncertainty map (H-B2, the "stability illusion").**
    - Per parcel: u under each headline sampler against the exact map.
    - Reported: mean u, the share of parcels with u > 0.1 bits, the Spearman correlation with the exact map, and the mean |Δu|.
    - **Direction expected before the data:** optimiser-like samplers (`optimiser`, `greedy`, long anneals) understate uncertainty.
- **Optimiser diagnostics:** how often the second set of starts improved the result, and the floor change between first-set and final optima.
- **Size:** the Spearman correlation between n and the best deployable floor.

## 9. Outcomes (fixed in advance)

In every case the paper states that the direct samplers were simulated as ideal, noiseless devices, which favours them.

| E7 result | Statement allowed in the paper |
|---|---|
| E7-A and E7-B pass | Sampling the quantum device directly cannot deliver co-assignment uncertainty at the paper's precision on most cortical networks. No direct sampler tested gets within ε = 0.02, even with unlimited shots and the best setting chosen with hindsight: QAOA to depth 4 with energy- or free-energy-optimised angles, coherent annealing up to T = 100, an ideal optimiser, or classical multi-start descent. Neither does QAOA with angles tuned to the exact answer. The same hardware inside QeMCMC is exact: the Metropolis step is what makes the device usable for uncertainty |
| E7-A passes, E7-B fails | Deployable direct samplers miss the target on most networks. Shallow QAOA circuits can represent the co-assignments, but only with angles tuned to the exact answer, which is the very thing being estimated. QeMCMC needs no such tuning |
| E7-A fails | On most cortical networks some deployable direct sampler reaches the target with unlimited shots. The case for QeMCMC then rests on exactness without knowing in advance which direct sampler works, and on the call counts |
| E7-K fails | The failure is investigated before anything is reported, as in E4 section 7. If a code error is found, it is fixed and the run repeated, and both runs are reported |

## 10. Smoke test (run before this protocol was fixed)

- **What ran:** `experiments/e7_smoke.py` on three models at n = 13 that are **not** E7 networks: the first 13 parcels of the 22-parcel Schaefer-400 LH_Cont ROI (FC and SC), and a random Gaussian model.
    - It printed only run times and numerical diagnostics (`results/e7_smoke.log`).
    - An earlier, slower implementation was stopped after its timings and is kept as `results/e7_smoke_numpy_partial.log`.
- **Annealing step.**
    - max_ij |p(0.01) − p(0.005)| ≤ 1.1 × 10⁻⁵ for every T ≤ 100, and |p(0.02) − p(0.01)| ≤ 4.3 × 10⁻⁵.
    - Hence dt = 0.01, with dt = 0.02 as the Trotter check.
- **Angle search.**
    - The test used 8 + 8 random starts. The second set improved one search out of 36 materially (by 1.6 nats: KL objective, depth 3, FC model); all others agreed to machine precision.
    - Hence 24 + 24 random starts, the cross-seeding of section 3, and the sensitivity re-search of section 7.
- **Run time** at n = 13 with 8 + 8 starts: about 90–110 s per energy or KL search, and 160–190 s per oracle search. The QAOA-MC kernel check takes 25 s, and its stationarity error was below 10⁻¹⁵.

## 11. Compute

- **Estimate:** about 2–2.5 h with 4 worker processes. The 16 networks with n = 13 dominate, at about 25 min each.
- **Sequence:** `e7_run.py`, then `e7_sensitivity.py`, then `e7_analyze.py`, in one chain.

## 12. Files

- **Code:** `qsr2/direct.py`; `experiments/e7_instances.py`, `e7_run.py`, `e7_sensitivity.py`, `e7_analyze.py`, `e7_smoke.py`; `tests/test_e7.py`.
- **Outputs:** `results/e7_rows.jsonl`, `e7_sensitivity.json`, `e7_summary.json`, `e7_parcel_map.csv`, `e7_run.log`; the smoke logs.
- **Hashes:** `results/e7_prereg_hashes.txt`.
