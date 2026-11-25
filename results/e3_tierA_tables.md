Instances: 292 (errors recorded: 0)

### Primary endpoint A (β̃ = 1, partition space)

| Modality | Instances | Slope of mean r per added node [95 % CI] | Mean r at n = 12 [95 % CI] | Endpoint A |
|---|---|---|---|---|
| FC | 144 | +0.952 [+0.75, +1.13] | +3.63 [+2.65, +4.52] | **PASS** |
| SC | 144 | +0.457 [+0.29, +0.65] | +1.15 [+0.32, +2.06] | **PASS** |

### Mean r and win share by n (β̃ = 1)

| | n = 7 | n = 8 | n = 9 | n = 10 | n = 11 | n = 12 |
|---|---|---|---|---|---|---|
| FC: mean r | -0.96 | +0.13 | +0.59 | +2.42 | +2.99 | +3.63 |
| FC: share r > 0 | 33 % | 54 % | 62 % | 92 % | 92 % | 88 % |
| FC: instances | 24 | 24 | 24 | 24 | 24 | 24 |
| SC: mean r | -0.79 | +0.16 | +0.65 | +0.99 | +2.15 | +1.15 |
| SC: share r > 0 | 38 % | 46 % | 62 % | 62 % | 88 % | 46 % |
| SC: instances | 24 | 24 | 24 | 24 | 24 | 24 |

### Secondary endpoints (same statistics, same instances)

| Modality | Quantity | Slope per node [95 % CI] | Value at n = 12 [95 % CI] | Both CIs > 0 |
|---|---|---|---|---|
| FC | Labelled-state (full-space) gaps | +0.367 [+0.14, +0.57] | +1.32 [+0.04, +2.45] | **pass** |
| FC | log₂ asymptotic-variance ratio (best classical / quantum) | +0.611 [+0.39, +0.82] | +2.43 [+1.31, +3.46] | **pass** |
| FC | r at β̃ = 0.5 | +0.562 [+0.43, +0.70] | +0.02 [-0.61, +0.71] | fail |
| FC | r at β̃ = 2 | +0.864 [+0.68, +1.05] | +4.78 [+3.94, +5.50] | **pass** |
| FC | r of QAOA-MC, grid average | +0.532 [+0.33, +0.71] | +0.71 [-0.16, +1.55] | fail |
| FC | r of QAOA-MC, angle 0 (out of sample) | +0.687 [+0.48, +0.90] | +2.22 [+1.25, +3.23] | **pass** |
| SC | Labelled-state (full-space) gaps | +0.046 [-0.16, +0.28] | -0.25 [-1.46, +0.94] | fail |
| SC | log₂ asymptotic-variance ratio (best classical / quantum) | +0.385 [+0.20, +0.56] | +1.57 [+0.71, +2.46] | **pass** |
| SC | r at β̃ = 0.5 | +0.476 [+0.35, +0.59] | +0.14 [-0.41, +0.76] | fail |
| SC | r at β̃ = 2 | +0.459 [+0.27, +0.65] | +2.10 [+1.09, +3.24] | **pass** |
| SC | r of QAOA-MC, grid average | +0.026 [-0.13, +0.20] | -1.42 [-2.20, -0.54] | fail |
| SC | r of QAOA-MC, angle 0 (out of sample) | +0.276 [+0.08, +0.47] | +0.30 [-0.71, +1.26] | fail |

### G1b minimum-exponent rule over n = 7–12 (secondary)

| Modality | k quantum | Best classical (k) | Δk [95 % CI] |
|---|---|---|---|
| FC | 0.119 | label_swap (0.836) | +0.716 [+0.41, +0.89] |
| SC | 0.132 | label_swap (0.471) | +0.339 [+0.12, +0.55] |

Per-chain exponents k (log₂ gap lost per added node):

| Modality | heat_bath | label_swap | local | oracle_a | quantum | sw | uniform | wolff |
|---|---|---|---|---|---|---|---|---|
| FC | 1.000 | 0.836 | 0.836 | 1.072 | 0.119 | 1.700 | 0.982 | 1.687 |
| SC | 1.000 | 0.471 | 0.471 | 0.589 | 0.132 | 1.919 | 1.029 | 1.627 |

### Routing test (prospective, fresh instances, all Tier A rows)

| Threshold | Routed: wins / total | Unrouted: wins / total | One-sided Fisher p |
|---|---|---|---|
| F ≥ 0.05 | 187 / 259 (72 %) | 0 / 33 (0 %) | 5.1e-17 |
| F ≥ 0.1 | 159 / 199 (80 %) | 28 / 93 (30 %) | 2.6e-16 |

Win share by frustration bin:

| F bin | Instances | Share r > 0 |
|---|---|---|
| [0,0.02) | 11 | 0 % |
| [0.02,0.05) | 22 | 0 % |
| [0.05,0.1) | 60 | 47 % |
| [0.1,0.2) | 165 | 78 % |
| [0.2,1.01) | 34 | 88 % |

### Which classical chain is the oracle (β̃ = 1)

FC:

| Chain | n = 7 | n = 8 | n = 9 | n = 10 | n = 11 | n = 12 |
|---|---|---|---|---|---|---|
| local | 5 | 3 | 5 | 4 | 7 | 11 |
| uniform | 2 | 6 | 5 | 8 | 10 | 10 |
| wolff | 17 | 15 | 14 | 12 | 7 | 3 |

SC:

| Chain | n = 7 | n = 8 | n = 9 | n = 10 | n = 11 | n = 12 |
|---|---|---|---|---|---|---|
| local | 13 | 15 | 17 | 20 | 16 | 20 |
| sw | 5 | 0 | 0 | 1 | 0 | 0 |
| uniform | 3 | 3 | 5 | 3 | 7 | 4 |
| wolff | 3 | 6 | 2 | 0 | 1 | 0 |

### Mean log₂ partition-space gap by chain (β̃ = 1)

FC:

| Chain | n = 7 | n = 8 | n = 9 | n = 10 | n = 11 | n = 12 |
|---|---|---|---|---|---|---|
| quantum | -3.90 | -4.10 | -4.14 | -4.51 | -4.33 | -4.52 |
| qaoa_a0 | -3.92 | -4.43 | -4.77 | -4.93 | -5.53 | -5.92 |
| qaoa | -4.80 | -5.25 | -5.90 | -6.46 | -6.96 | -7.44 |
| local | -6.18 | -7.38 | -7.63 | -9.24 | -9.37 | -10.52 |
| label_swap | -7.18 | -8.38 | -8.63 | -10.24 | -10.37 | -11.52 |
| uniform | -5.48 | -6.33 | -7.35 | -8.57 | -9.36 | -10.29 |
| sw | -4.26 | -5.67 | -7.18 | -9.90 | -11.22 | -12.28 |
| wolff | -3.34 | -4.70 | -6.22 | -8.92 | -10.24 | -11.29 |
| heat_bath | -6.00 | -7.00 | -8.00 | -9.00 | -10.00 | -11.00 |

SC:

| Chain | n = 7 | n = 8 | n = 9 | n = 10 | n = 11 | n = 12 |
|---|---|---|---|---|---|---|
| quantum | -3.97 | -3.85 | -4.29 | -4.09 | -4.50 | -4.55 |
| qaoa_a0 | -3.80 | -3.89 | -4.54 | -4.63 | -4.85 | -5.40 |
| qaoa | -4.35 | -4.90 | -5.60 | -5.96 | -6.73 | -7.11 |
| local | -4.28 | -4.65 | -5.68 | -5.47 | -7.06 | -6.17 |
| label_swap | -5.28 | -5.65 | -6.68 | -6.47 | -8.06 | -7.17 |
| uniform | -4.75 | -5.76 | -6.97 | -7.59 | -9.19 | -9.77 |
| sw | -5.30 | -7.24 | -8.76 | -10.35 | -13.75 | -14.50 |
| wolff | -6.21 | -7.09 | -8.47 | -10.57 | -13.04 | -13.61 |
| heat_bath | -6.00 | -7.00 | -8.00 | -9.00 | -10.00 | -11.00 |

### By ROI kind (mean r, β̃ = 1)

| Modality | Kind | n = 7 | n = 8 | n = 9 | n = 10 | n = 11 | n = 12 |
|---|---|---|---|---|---|---|---|
| FC | border-set | -1.86 | -0.87 | +0.23 | +1.99 | +2.30 | +4.01 |
| FC | within-set | -0.06 | +1.13 | +0.95 | +2.84 | +3.68 | +3.25 |
| SC | border-set | -0.90 | +0.18 | +0.42 | +0.75 | +1.94 | +1.55 |
| SC | within-set | -0.68 | +0.14 | +0.88 | +1.23 | +2.36 | +0.74 |

### QAOA-MC angles on fresh instances (β̃ = 1)

| Modality | Quantity | n = 7 | n = 8 | n = 9 | n = 10 | n = 11 | n = 12 |
|---|---|---|---|---|---|---|---|
| FC | best angle index | 0 | 0 | 0 | 0 | 0 | 0 |
| FC | rank of angle 0 (of 18) | 1 | 1 | 1 | 1 | 1 | 1 |
| FC | mean r, best angle | -0.98 | -0.20 | -0.04 | +2.00 | +1.79 | +2.22 |
| FC | mean r, angle 0 | -0.98 | -0.20 | -0.04 | +2.00 | +1.79 | +2.22 |
| SC | best angle index | 0 | 0 | 0 | 0 | 0 | 0 |
| SC | rank of angle 0 (of 18) | 1 | 1 | 1 | 1 | 1 | 1 |
| SC | mean r, best angle | -0.62 | +0.12 | +0.41 | +0.45 | +1.80 | +0.30 |
| SC | mean r, angle 0 | -0.62 | +0.12 | +0.41 | +0.45 | +1.80 | +0.30 |

### PT-2 check at n = 7–8 (six ladders, oracle-tuned per instance)

| Modality | Rows | Share where PT-2 beats the endpoint oracle | Mean log₂(PT-2 / oracle) |
|---|---|---|---|
| FC | 48 | 0 % | -3.58 |
| SC | 48 | 0 % | -2.67 |

### Canonical ROIs (β̃ = 1)

| Dataset | ROI | n | F | r |
|---|---|---|---|---|
| FC200 | LH_SalVentAttn | 11 | 0.122 | +4.59 |
| FC200 | RH_SalVentAttn | 11 | 0.124 | +4.96 |
| SC200 | LH_SalVentAttn | 11 | 0.192 | +1.52 |
| SC200 | RH_SalVentAttn | 11 | 0.226 | +3.59 |
