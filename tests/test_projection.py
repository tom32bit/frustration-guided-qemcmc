"""qsr2/projection.py: the scaling-projection procedure tested prospectively in E6 (endpoint E6-C)."""
import numpy as np

from qsr2 import projection


def _data(rng, a=1.0, b=0.5, sd=1.0, groups=3, sizes=(7, 8, 9, 10), per=8, shift=None):
    n, y, g = [], [], []
    for gi in range(groups):
        off = 0.0 if shift is None else shift[gi]
        for s in sizes:
            n += [s] * per
            y += list(a + off + b * s + sd * rng.standard_normal(per))
            g += [gi] * per
    return np.array(n), np.array(y), np.array(g)


def test_noise_free_line_is_exact():
    n = np.repeat(np.arange(7, 11), 5)
    y = 2.0 - 0.3 * n
    out = projection.project(n, y, np.zeros(len(n)), {12: 10, 13: 10}, B=200, seed=1)
    assert abs(out["slope"] + 0.3) < 1e-12 and abs(out["intercept"] - 2.0) < 1e-12
    for t, v in out["targets"].items():
        assert abs(v["point"] - (2.0 - 0.3 * int(t))) < 1e-12
        assert abs(v["pi95"][0] - v["point"]) < 1e-9 and abs(v["pi95"][1] - v["point"]) < 1e-9


def test_deterministic_given_seed():
    rng = np.random.default_rng(3)
    n, y, g = _data(rng)
    a = projection.project(n, y, g, {12: 24}, B=300, seed=9)
    b = projection.project(n, y, g, {12: 24}, B=300, seed=9)
    assert a == b


def test_coverage_near_nominal():
    """Simulated linear data with group offsets: the 95 % interval for the mean of 24 new instances two sizes beyond
    the fit should cover the realised new mean about 95 % of the time."""
    rng = np.random.default_rng(11)
    hits, sims = 0, 200
    shift = [0.0, 0.7, -0.5]
    for _ in range(sims):
        n, y, g = _data(rng, shift=shift)
        out = projection.project(n, y, g, {12: 24}, B=400, seed=int(rng.integers(1 << 31)))
        new = np.concatenate([1.0 + s + 0.5 * 12 + rng.standard_normal(8) for s in shift]).mean()
        lo, hi = out["targets"]["12"]["pi95"]
        hits += lo <= new <= hi
    assert 0.88 <= hits / sims <= 0.995, hits / sims
