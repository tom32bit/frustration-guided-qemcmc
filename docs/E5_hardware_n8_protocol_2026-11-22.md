# E5 proper protocol: QAOA-MC proposals on IBM Heron at n = 8 (pre-specified 22 Nov 2026)

**Status:** written before any n = 8 hardware job was submitted.

- The instance selection is fixed by `hardware/e5_select.py` and recorded read-only in `results/e5_selection.json` (SHA-256 `ae8a28ae…7ef21f6`).
- The analysis script `hardware/e5_n8_analyze.py`, including the three thresholds below, was written before any noise-model (dry-run) output was analysed.
- Code: `hardware/e5_n8.py` (compile, dry run, submit) and `hardware/e5_n8_analyze.py`.

**Question.** The ideal palindromic QAOA-MC proposal (the hardware circuit) beats every classical sampler on frustrated FC networks in E3 Tier A. Does the device-run version keep that advantage at n = 8? At n = 8 the circuit has twice the two-qubit gates of the n = 6 pilot.

The ideal advantage on these instances is known exactly. The hardware question is how much of it survives noise, and at what bias in the community co-assignment estimates.

## Instances (fixed by rule, using no hardware or noise-model information)

**Pool** (FC, n = 8):

- the three 8-parcel canonical Yeo-7 × hemisphere ROIs of Schaefer-100: LH_DorsAttn, RH_SomMot, RH_Vis;
- the 24 E3 Tier A random ROI node sets at n = 8 (FC-200, FC-400).

**Eligibility** (β = 1/σ_ROI, β̃ = 1, partition space):

1. The two-factor routing rule of E3 amendment 13 holds: F ≥ 0.10 and βD ≥ 1.
2. There is non-trivial community uncertainty: n_eff = 1/Σ p_c² ≥ 2 over partitions. This applies the pilot's lesson; its FC instance had p_max = 1.
3. The circuit has a real ideal advantage: r_a0 = log₂(gap_QAOA,angle 0 / gap_oracle) ≥ 1, computed exactly.

**Order.** Eligible canonical ROIs first, then eligible random ROIs in a seeded random order (seed 20261125); take three.

**Outcome of the rule.**

- 4 of 27 candidates are eligible.
- No canonical ROI qualifies: LH_DorsAttn and RH_Vis have n_eff ≤ 1.21, and RH_SomMot has F = 0.03.
- The Tier A values are reproduced exactly (relative difference 0).
- On ranks 1 and 2 the best classical chain is Metropolis with uniform random proposals. A fully depolarised device proposal would turn into exactly that chain, so H2 also tests that the device proposal beats random proposals.

| Rank | Instance | F | βD | n_eff | Ideal QAOA gap / best classical | Best classical chain |
|---|---|---|---|---|---|---|
| 1 | FC-200 within-module set, nodes 32, 42, 47, 52, 62, 64, 146, 179 | 0.131 | 3.83 | 2.04 | 6.2× | uniform proposal |
| 2 | FC-400 within-module set, nodes 70, 88, 108, 144, 277, 280, 284, 304 | 0.181 | 6.54 | 3.22 | 3.6× | uniform proposal |
| 3 | FC-200 within-module set, nodes 56, 60, 86, 96, 161, 181, 185, 191 | 0.186 | 1.85 | 3.68 | 4.3× | local |

## Circuits and device

| Item | Choice |
|---|---|
| Proposal | Palindromic QAOA-MC at angle 0, (b₁, b₂, γ₁) = (π/8, π/8, 0.5). This angle ranked 1st of 18 at every n in Tier A, and it is the circuit validated on hardware in the pilot |
| Rows | One circuit per representative start state s < 2⁷: 128 circuits, 2,000 shots each. Outcomes are lumped into partition classes {x, NOT x}. The hardware chain maps its state to the representative before proposing, so these 128 rows define it exactly |
| Backend | The visible device with the lowest median two-qubit error, fixed at first use: `ibm_kingston`, median CZ error 0.20 % (`results/e5_n8_backend.json`) |
| Compilation | optimization_level = 3; the best of transpiler seeds 0–9 by CZ count on the first circuit; that seed and layout are then used for all 128 circuits. At dry-run time: 203 CZ, depth ≈ 362, physical qubits 22, 23, 21, 16, 3, 4, 2, 1. At submission the circuits are recompiled with the calibration then current, and the actual values are recorded |
| Runtime options | SamplerV2, gate and measurement twirling, XY4 dynamical decoupling, as in the pilot. Job tags are generic (`qemcmc`, `n8`) |

## QPU budget

- 516 s are available (the 28-day window is rolling; the pilot's 84 s are counted until about 19 Dec 2026).
- The estimate is about 85 s per instance: the pilot used 42 s for 128,000 shots, and each instance here is 256,000 shots.
- **Cap: 300 s** for E5 proper.
- Instance 1 is submitted first. Each further instance is submitted only if usage so far plus the largest single-job usage so far stays ≤ 300 s.
- `service.usage()` is checked before every submission.

## Metrics (partition space; exact analysis of the measured kernel)

**Hardware chain.** Metropolis with the measured proposal Q̂, treated as symmetric, which is how the device would be used. Its gap is 1 − |λ₂|, which is also defined for the slightly non-reversible measured chain.

**Primary criteria, per instance:**

| | Criterion | Threshold |
|---|---|---|
| H1 | Retention: gap_hardware / gap_ideal | ≥ 0.5 |
| H2 | Advantage: gap_hardware / gap_oracle, where the oracle is the exact best of local, label-swap, uniform, Swendsen–Wang, Wolff and heat-bath per cost | > 1 |
| H3 | Bias: max over node pairs of the co-assignment error, abs(p_hw,ij − p_ij), from the hardware chain's exact stationary law | ≤ 0.05 |

- **E5 success** iff H1–H3 all hold on at least 2 of the 3 instances.
- The thresholds rest on the pilot (n = 6: retention 1.02, bias 0.019) and on a bound: if noise mixes a fraction ε of arbitrary symmetric proposals into the ideal one, the gap stays ≥ (1 − ε) × ideal.

**Descriptive:**

- row TV distance and transpose asymmetry against their finite-shot floors (ideal rows resampled 100 times at 2,000 shots);
- the finite-shot null distribution of retention and bias;
- 95 % intervals from a parametric bootstrap of the hardware rows (200 resamples);
- the Hastings-corrected chain, which uses Q̂'s own Hastings ratio, is exactly Gibbs-reversible and so has zero bias for the measured kernel: its retention and advantage;
- the Layden-quench ideal gap, for context.

## Noise-model prediction (dry run on `AerSimulator.from_backend`, before submission)

| Rank | Retention | Advantage (ideal) | Max / mean co-assignment bias | Row TV (finite-shot floor) | Hastings retention |
|---|---|---|---|---|---|
| 1 | 0.89 | 5.5× (6.2×) | 0.018 / 0.005 | 0.142 (0.088) | 0.89 |
| 2 | 0.88 | 3.1× (3.6×) | 0.024 / 0.010 | 0.142 (0.086) | 0.86 |
| 3 | 1.09 | 4.6× (4.3×) | 0.009 / 0.003 | 0.141 (0.088) | 1.10 |

The pilot's device noise was about 3× the noise model's excess row TV, and its bias was 3× the model's (0.019 against 0.006). If that ratio repeats, H3 is at risk for rank 2. The bias criterion is therefore a real test.

## Interpretation (fixed in advance)

| Outcome | Statement allowed in the paper |
|---|---|
| E5 success | On routed FC brain networks at n = 8 (203 CZ), the device-run proposal keeps at least half its ideal mixing rate, beats every classical sampler tested, and biases co-assignment by at most 0.05 |
| H1 and H2 hold, H3 fails | Mixing advantage survives on hardware, but the bias exceeds 0.05. Report the Hastings-corrected chain (descriptive) as the bias-free route and the bias as the hardware limitation |
| H1 or H2 fails on ≥ 2 instances | Hardware advantage not shown at n = 8. The paper's hardware claim stays at the n = 6 pilot |

## Limits (stated in advance)

- The ideal advantage was used for selection (criterion 3). This is exact, not noisy, so the hardware question (retention) is unaffected, but the instances are not a random sample of FC networks.
- The full 128-row proposal matrix is measured and the chain is analysed exactly. No sequential chain is run on the device.
- There are three instances on one device on one day.
