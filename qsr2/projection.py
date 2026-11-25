"""Scaling projection of a per-instance log2 advantage (docs/E6_generality_protocol_2026-11-22.md, endpoint E6-C).
Fixed on 22 Nov 2026, before any E6 row existed and before it was applied to the brain data
(experiments/scaling_projection_brain.py).

Model: y = a + b n, fitted by ordinary least squares to instance-level data at the fitting sizes. Because y is a log2
ratio, this is an exponential law for the ratio itself, with b the advantage gained per added node (bits per node).
Prediction for the MEAN of m new instances at a size n* beyond the fit:
  point   a_hat + b_hat n*
  95 % prediction interval from a stratified bootstrap with B resamples:
    1. resample instances with replacement within each (group, size) cell and refit (a_b, b_b);
    2. add new-sample noise: the mean of m residuals drawn with replacement from that refit's residuals at the largest
       fitting size;
    3. take the 2.5 and 97.5 percentiles of a_b + b_b n* + noise.
The interval covers the uncertainty of the fitted line and the spread of new instances. It does NOT cover a change of
the law beyond the fitted range; that is what the prospective test E6-C checks, one and two sizes ahead.
VERIFY (tests/test_projection.py): exact on noise-free lines; near-nominal coverage on simulated linear data.
"""
import numpy as np


def ols(n, y):
    X = np.column_stack([np.ones(len(n)), np.asarray(n, float)])
    coef, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return float(coef[0]), float(coef[1])


def _pct(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def project(n, y, groups, targets, B=10000, seed=0):
    """n, y, groups: instance-level arrays at the fitting sizes. targets: {n_star: m_star}.
    Returns {"fit_sizes", "instances", "intercept", "slope", "slope_ci95", "targets": {str(n*): {"m", "point", "pi95"}}}."""
    n = np.asarray(n, int)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    a, b = ols(n, y)
    cells = {}
    for i, k in enumerate(zip(groups.tolist(), n.tolist())):
        cells.setdefault(k, []).append(i)
    cells = [np.array(v) for _, v in sorted(cells.items())]
    nmax = int(n.max())
    rng = np.random.default_rng(seed)
    slopes = np.empty(B)
    preds = {t: np.empty(B) for t in targets}
    for k in range(B):
        idx = np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])
        ab, bb = ols(n[idx], y[idx])
        slopes[k] = bb
        res = (y[idx] - (ab + bb * n[idx]))[n[idx] == nmax]
        for t, m in targets.items():
            preds[t][k] = ab + bb * t + res[rng.integers(0, len(res), m)].mean()
    return {"fit_sizes": sorted(set(n.tolist())), "instances": int(len(y)), "intercept": a, "slope": b,
            "slope_ci95": _pct(slopes),
            "targets": {str(t): {"m": int(m), "point": a + b * t, "pi95": _pct(preds[t])} for t, m in targets.items()}}
