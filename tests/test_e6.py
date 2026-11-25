"""E6 pipeline (sealed routing -> fit stage -> sealed projection -> test stage -> analysis) on SYNTHETIC networks in a
temporary results folder, with small sizes; no network data are touched. Also checks the two guards that make
E6-C prospective: the test stage refuses to run before the projection is sealed, and the projection refuses to be
written once test-size rows exist."""
import json
import os
import sys

import numpy as np
import pytest

from qsr2 import networks, patches


def _fake(sizes=(4, 5, 6, 7)):
    out = []
    for net_i, net in enumerate(("netA", "netB")):
        g = np.random.default_rng(100 + net_i)
        for n in sizes:
            for k in range(4):
                W = g.random((n, n)); W = np.triu(W, 1); W = W + W.T
                J = networks.modularity_matrix(W, 1.0).copy(); np.fill_diagonal(J, 0)
                v = J[np.triu_indices(n, 1)]
                d = dict(family=f"E6_{net}", ds=net, n=n, kind="border-set" if k % 2 else "within-set", nodes=list(range(n)),
                         names=[str(i) for i in range(n)], J=J, h=np.zeros(n), sigma=float(np.std(v)))
                d["F"] = patches.frustration(J, None)
                out.append(d)
    return out


def test_e6_pipeline_dry(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    import e6_analyze
    import e6_project
    import e6_routing_predict
    import e6_run
    fake = _fake()
    for mod in (e6_routing_predict, e6_run, e6_project):
        monkeypatch.setattr(mod, "e6_instances", lambda: fake)
    monkeypatch.setattr(e6_run, "NETS", ("netA", "netB"))
    monkeypatch.setattr(e6_run, "FIT_SIZES", (4, 5))
    monkeypatch.setattr(e6_run, "TEST_SIZES", (6, 7))
    monkeypatch.setattr(e6_project, "B", 300)
    monkeypatch.setattr(e6_analyze, "NB", 300)
    for mod in (e6_routing_predict, e6_run, e6_project, e6_analyze):
        monkeypatch.setattr(mod, "RES", str(tmp_path))
    e6_routing_predict.main()
    monkeypatch.setattr(sys, "argv", ["e6_run.py", "test"])
    with pytest.raises(SystemExit):
        e6_run.main()                                     # test stage before the projection is sealed
    monkeypatch.setattr(sys, "argv", ["e6_run.py", "fit"])
    e6_run.main()
    e6_project.main()
    monkeypatch.setattr(sys, "argv", ["e6_run.py", "test"])
    e6_run.main()
    (tmp_path / "e6_projection.json").rename(tmp_path / "sealed.json")
    with pytest.raises(SystemExit):
        e6_project.main()                                 # test-size rows exist: a new projection is refused
    (tmp_path / "sealed.json").rename(tmp_path / "e6_projection.json")
    e6_analyze.main()
    S = json.loads((tmp_path / "e6_summary.json").read_text())
    assert S["rows"] == len(fake) and S["rows_when_routing_written"] == 0
    assert set(S["projection_rows_by_size_when_written"]) == {"4", "5"}
    assert S["E6C"]["all_r"] is not None and set(S["E6C"]["all_r"]["targets"]) == {"6", "7"}
    for t in S["E6C"]["all_r"]["targets"].values():
        assert t["instances"] == t["expected_instances"] == 8 and t["observed_mean"] is not None
