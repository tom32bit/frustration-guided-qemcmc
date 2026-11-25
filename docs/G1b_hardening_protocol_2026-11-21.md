# G1b hardening protocol: stronger classical baselines (pre-specified 21 Nov 2026)

**Status:** written and frozen *before* any G1b number was computed. The analysis code (`experiments/g1b_analyze.py`) implements exactly this rule.

## Why

G1 passed for the two P3 ROI families. Its "best classical" set lacked the samplers a referee will ask for on frustrated Ising landscapes:

- cluster moves;
- parallel tempering with a tuned temperature ladder and more than 2 replicas.

G1b adds them. The quantum side stays exactly as in G1: the untuned Layden quench averaged over the γ–t grid. The quantum gaps are reused from `results/g1_rows.jsonl`, not recomputed.

## Instances

The same field-free G1 instances (P2 split, P3 ROI, P3 canonical) are rebuilt by the same seeded code (`g1_regime_map.build_instances`).

**Determinism check:** each rebuilt instance is matched to its G1 row. The local-move gap is recomputed and must agree to 1e-9 relative. Any mismatch aborts the run.

P1 border patches are excluded. They already fail G1, and cluster moves are not exact for them (they carry fields).

## New classical baselines

All new baselines are exact 2ⁿ-state transition matrices. Gaps are computed per step, as in G1.

1. **Swendsen–Wang (SW).** Exact transition matrix, built by a subset dynamic programme over bond configurations:
    - bond probability p = 1 − exp(−2β|J_ij|), on satisfied couplings only;
    - valid for mixed-sign couplings.
2. **Wolff single-cluster.** Same construction, with a uniform seed site.
3. **PT-2, oracle-tuned.** Two replicas at (β, rβ), with r ∈ {1/2, 1/3, 1/5, 1/10, 1/30, 0}. For each instance the best r is chosen after the fact (an oracle choice that favours classical). n = 4–8, β̃ ∈ {1, 2}.
4. **PT-3, oracle-tuned.** Three replicas at (β, (1+r)β/2, rβ), with r ∈ {0.3, 0.1, 0}, best per instance. n = 4–6, β̃ = 1 only, because of the state-space size.
5. **Classical oracle.** For each instance, the maximum gap over *every* classical chain available at that n. This is the strongest single comparator and is included in every comparison set.
6. **Houdayer cluster moves: analytical, not simulated.** For couplings with every off-diagonal J_ij ≠ 0 (true of all modularity matrices here; checked for every instance), the Houdayer cluster is the entire set of sites where the two replicas differ. Flipping it maps (z¹, z²) to (z², z¹), a same-temperature replica exchange. That leaves every estimator unchanged. So Houdayer adds nothing on dense couplings. A unit test verifies this, and the run records the density of every instance.

## Decision rule (β̃ = 1)

Δk = k_best-classical − k_quantum, with a 95 % bootstrap CI from 1,000 resamples of instances within each n. Exponents are fitted exactly as in G1. G1b **passes for a family** iff its CI lies strictly above 0 in both conditions:

- **(a′)** n = 4–10 against {local, label-swap, uniform, heat-bath (k = 1), SW, Wolff, classical oracle};
- **(b′)** n = 4–8 against the (a′) set plus PT-2 oracle-tuned, with the classical oracle recomputed to include it.

**Outcomes:**

| Families passing | Consequence |
|---|---|
| Both P3_roi_FC and P3_roi_SC | Main paper route confirmed as scoped after G1 |
| Exactly one | Main route, with the claim restricted to that modality; the other is reported as a negative |
| Neither | Fall back to the "where does it help?" characterisation paper (plan §fallback) |

## Secondary (reported, not decisive)

- (c′) n = 4–6 with PT-3 added. Only three sizes, so no exponent claim is made.
- β̃ = 0.5 and 2 versions of (a′), and of (b′) at β̃ = 2.
- Instance win shares against the classical oracle, overall and by frustration bin. This is the prospective check of the post-hoc routing rule on these instances.
- P2 families and the 24 canonical ROI networks against the extended set.

## Declared asymmetry

The classical side gets per-instance oracle tuning (ladder, chain type). The quantum side does not. Any pass is therefore conservative.
