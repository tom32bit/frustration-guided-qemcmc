"""E6-C: SEALED scaling projection, written after the fit stage (sizes 7-10) and BEFORE any row of the test sizes
(11-12) exists. Applies the frozen procedure qsr2.projection.project (B = 10,000, seed 47; groups = networks) to
instance-level advantages at the fitting sizes and predicts the MEAN over the E6 instances at each test size (their
number is known in advance from the deterministic generator).
Metrics: r = log2(gap_quantum / gap_oracle) at beta_tilde = 1 (E3 endpoint-A definition; PRIMARY for E6-C) and the
steps-to-accuracy ratio log2(s2_best classical / s2_quantum) (secondary). Classes: all instances (PRIMARY) and the
sealed predicted-win class (secondary).
Writes results/e6_projection.json (refuses to overwrite, refuses if any test-size row exists)."""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import projection  # noqa: E402
from instances import RES, key  # noqa: E402
from e3c_analyze import r_of, s2_ratio  # noqa: E402
from e6_instances import e6_instances  # noqa: E402
import e6_run  # noqa: E402

B, SEED = 10000, 47
METRICS = {"r": r_of, "s2": s2_ratio}


def main():
    path = os.path.join(RES, "e6_projection.json")
    if os.path.exists(path):
        sys.exit(f"{path} exists; the projection is sealed and is not recomputed")
    rows = [json.loads(l) for l in open(os.path.join(RES, "e6_rows.jsonl"))]
    if any(r["n"] in e6_run.TEST_SIZES for r in rows):
        sys.exit("rows at a test size already exist; a projection written now would not be prospective")
    P = json.load(open(os.path.join(RES, "e6_routing_predictions.json")))
    pred = {tuple(x["key"]): x["predicted_quantum_win"] for x in P["instances"]}
    inst = [d for d in e6_instances() if d["n"] in e6_run.TEST_SIZES]
    keys = [tuple(json.loads(json.dumps(list(key(d))))) for d in inst]
    out = {"procedure": projection.__doc__, "B": B, "seed": SEED, "fit_sizes": list(e6_run.FIT_SIZES),
           "test_sizes": list(e6_run.TEST_SIZES),
           "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "rows_by_size_when_written": {str(n): sum(r["n"] == n for r in rows) for n in sorted({r["n"] for r in rows})},
           "projections": {}}
    for cls in ("all", "predicted_win"):
        sel = [r for r in rows if r["n"] in e6_run.FIT_SIZES and (cls == "all" or pred[tuple(r["key"])])]
        targets = {n: sum(1 for d, k in zip(inst, keys) if d["n"] == n and (cls == "all" or pred[k])) for n in e6_run.TEST_SIZES}
        for m, f in METRICS.items():
            name = f"{cls}_{m}"
            if len({r["n"] for r in sel}) < 2 or len(sel) < 10 or min(targets.values()) == 0:
                out["projections"][name] = None
                continue
            out["projections"][name] = projection.project([r["n"] for r in sel], [f(r) for r in sel], [r["ds"] for r in sel],
                                                          targets, B=B, seed=SEED)
    json.dump(out, open(path, "x"), indent=1)
    for name, v in out["projections"].items():
        if v:
            print(name, "slope", round(v["slope"], 3), {t: (round(x["point"], 2), [round(y, 2) for y in x["pi95"]]) for t, x in v["targets"].items()})


if __name__ == "__main__":
    main()
