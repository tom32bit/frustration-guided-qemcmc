Instances: 40

### Primary endpoint B (per-step accounting)

| Modality | Slope of mean ρ per added node [95 % CI] | Mean ρ at n = 14 [95 % CI] | Endpoint B |
|---|---|---|---|
| FC | +0.267 [-0.67, +1.16] | +2.52 [-0.48, +5.33] | FAIL |
| SC | -0.343 [-1.14, +0.35] | +2.21 [+0.42, +4.00] | FAIL |

### Mean ρ by n

| Modality | Accounting | n = 12 | n = 14 | n = 16 |
|---|---|---|---|---|
| FC | per step (primary) | +1.37 (8) | +2.52 (8) | +2.44 (4) |
| FC | classical gets n·T steps | -3.25 (8) | -2.63 (8) | -4.38 (4) |
| SC | per step (primary) | +1.73 (8) | +2.21 (8) | +0.35 (4) |
| SC | classical gets n·T steps | -4.48 (8) | -2.92 (8) | -5.59 (4) |

### Best classical chain at T (counts)

| Modality | heat_bath | label_swap | local | sw | uniform | wolff |
|---|---|---|---|---|---|---|
| FC | 1 | 1 | 11 | 1 | 1 | 5 |
| SC | 2 | 1 | 14 | 1 | 2 | 0 |

### Validation at n = 12 against the exact asymptotic variances (same instances)

| Pairs | Spearman | Mean simulated ρ | Mean exact log₂ variance ratio |
|---|---|---|---|
| 16 | +0.77 | +1.55 | +2.24 |

### Amendment 13: sealed two-factor routing predictions (n = 14, 16; fresh instances)

Predictions written 2026-11-22T02:00:47+06:00, with 0 outcome rows in existence.

| Class | ρ > 0 / instances | Mean ρ |
|---|---|---|
| Predicted win (F ≥ 0.10 and βD ≥ 1) | 13 / 15 | +3.60 |
| Rest | 4 / 9 | -0.56 |

One-sided Fisher exact p = 0.042; classification accuracy 75 %.
