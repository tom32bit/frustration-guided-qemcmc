"""E6: SEALED two-factor routing predictions for every E6 instance, computed from instance data only (couplings,
sigma, F) BEFORE any E6 row exists. Same rule and thresholds as E3 amendment 13, E3c and E4:
  predicted quantum win  <=>  F >= 0.10  and  beta * D >= 1   (beta = 1 / sigma_ROI, beta_tilde = 1)
Writes results/e6_routing_predictions.json (refuses to overwrite)."""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import ising  # noqa: E402
from instances import RES, key  # noqa: E402
from e3_posthoc_barrier import critical_depth  # noqa: E402
from e6_instances import e6_instances  # noqa: E402

F_THR, D_THR = 0.10, 1.0


def main():
    path = os.path.join(RES, "e6_routing_predictions.json")
    if os.path.exists(path):
        sys.exit(f"{path} exists; predictions are sealed and are not recomputed")
    rows_path = os.path.join(RES, "e6_rows.jsonl")
    done = sum(1 for _ in open(rows_path)) if os.path.exists(rows_path) else 0
    out = []
    for d in e6_instances():
        bD = critical_depth(ising.energies(d["J"]), d["n"]) / d["sigma"]
        out.append({"key": list(key(d)), "ds": d["ds"], "n": d["n"], "kind": d["kind"], "F": d["F"], "beta_D": bD,
                    "predicted_quantum_win": bool(d["F"] >= F_THR and bD >= D_THR)})
    S = {"rule": __doc__, "F_threshold": F_THR, "betaD_threshold": D_THR,
         "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "e6_rows_when_written": done, "instances": out, "predicted_wins": sum(x["predicted_quantum_win"] for x in out),
         "predicted_wins_by_network": {ds: sum(x["predicted_quantum_win"] for x in out if x["ds"] == ds)
                                       for ds in sorted({x["ds"] for x in out})}}
    json.dump(S, open(path, "x"), indent=1)
    print(f"{len(out)} instances, {S['predicted_wins']} predicted wins {S['predicted_wins_by_network']}; E6 rows so far: {done}")


if __name__ == "__main__":
    main()
