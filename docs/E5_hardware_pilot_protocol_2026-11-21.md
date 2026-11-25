# E5 hardware pilot protocol: QAOA-MC proposals on IBM Heron (pre-specified 21 Nov 2026)

**Status:** written before any hardware job was submitted. The code is `hardware/e5_pilot.py` (submit) and `hardware/e5_pilot_analyze.py`.

**Purpose:** a feasibility pilot, not an advantage test. It measures how far a real device's proposal distribution is from the ideal one, how much bias that noise puts into the community co-assignment estimates, and how the hardware chain's partition-space gap compares with the ideal one. These numbers set the sizes and depths of E5 proper.

## Fixed choices

| Item | Choice | Why |
|---|---|---|
| Instance | Left-hemisphere somatomotor network (Yeo-7 `LH_SomMot`), Schaefer-100: the only 6-parcel canonical ROI. Run in both modalities: SC (F = 0.143, frustrated) and FC (F = 0.000, unfrustrated) | Anatomically defined, not chosen by outcome. The two modalities sit on opposite sides of the routing rule |
| Temperature | β̃ = 1 (β = 1/σ_ROI) | Primary temperature of G1–G1c |
| Proposal | Palindromic QAOA-MC (QSR v1 circuit), with one grid angle set per modality: the best mean partition-space gap over the G1 `P3_roi_{FC,SC}` instances at n = 6 (`g1c_summary.json`). The pilot ROI is not in those families, so it is held out | Depth 2 with exact symmetry U^T = U. The Layden quench (t up to 20) needs thousands of two-qubit gates |
| Circuits | All 64 start states per modality (128 in total), 2,000 shots each | The full 64 × 64 proposal matrix lets the hardware chain be analysed exactly |
| Backend | The visible Heron device with the lowest median two-qubit error at submission (currently `ibm_kingston`) | — |
| Transpilation | optimization_level = 3, fixed seed; one layout, taken from the first circuit, for all circuits | Every row of the matrix sees the same physical qubits |
| Runtime options | SamplerV2 with gate and measurement twirling and XY4 dynamical decoupling | Measurement twirling symmetrises readout bias |
| Budget | At most 120 s of QPU time (Open plan: 600 s per 28 days). Stop after the SC job if its usage exceeds 80 s | — |
| Before submission | The same pipeline is run on a local noisy simulator built from the backend's calibration data (`AerSimulator.from_backend`), as a dry run | Checks the pipeline end to end and gives a noise-model prediction to compare with |

## Metrics (descriptive; no pass/fail)

1. **Row fidelity.** For each start state, the TV distance between the hardware row and the ideal row. Report the mean over rows, next to the finite-shot floor: the same statistic for ideal rows resampled at 2,000 shots.
2. **Symmetry.**
    - Flip asymmetry: the TV distance between Q̂(s, ·) and the flipped row of F s.
    - Transpose asymmetry: ‖Q̂ − Q̂ᵀ‖ divided by ‖Q̂‖, against its finite-shot floor.
3. **Bias.** The hardware chain is MH with the flip-averaged Q̂, run as if the proposal were symmetric, which is how it would be used. Compute its exact stationary distribution. Report the TV distance to the Gibbs distribution and the maximum co-assignment error, max_ij |p̂_ij − p_ij|.
4. **Mixing.** The partition-space gap of the hardware chain, against:
    - the ideal QAOA-MC chain;
    - the ideal Layden quench chain;
    - the classical per-instance oracle (local, label-swap, uniform, Swendsen–Wang, Wolff, heat-bath).
5. **Calibration.** The noisy-simulator values of 1–4, for comparison with the hardware values.
