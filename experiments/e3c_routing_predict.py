"""E3c: SEALED two-factor routing predictions for the n = 13 nested-growth instances, computed from instance data
only (couplings, sigma, F) BEFORE any E3c study row exists. Same rule and thresholds as E3 amendment 13:
  predicted quantum win  <=>  F >= 0.10  and  beta * D >= 1   (beta = 1 / sigma_ROI, beta_tilde = 1)
Writes results/e3c_routing_predictions.json (refuses to overwrite)."""
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
from e3c_instances import nested  # noqa: E402

F_THR, D_THR = 0.10, 1.0


def main():
    path = os.path.join(RES, "e3c_routing_predictions.json")
    if os.path.exists(path):
        sys.exit(f"{path} exists; predictions are sealed and are not recomputed")
    rows_path = os.path.join(RES, "e3c_rows.jsonl")
    done = sum(1 for _ in open(rows_path)) if os.path.exists(rows_path) else 0
    out = []
    for d in nested(12):
        bD = critical_depth(ising.energies(d["J"]), d["n"]) / d["sigma"]
        out.append({"key": list(key(d)), "family": d["family"], "n": d["n"], "F": d["F"], "beta_D": bD,
                    "predicted_quantum_win": bool(d["F"] >= F_THR and bD >= D_THR)})
    S = {"rule": __doc__, "F_threshold": F_THR, "betaD_threshold": D_THR,
         "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "study_rows_when_written": done, "instances": out,
         "predicted_wins": sum(x["predicted_quantum_win"] for x in out)}
    json.dump(S, open(path, "x"), indent=1)
    print(f"{len(out)} instances, {S['predicted_wins']} predicted wins; E3c study rows so far: {done}")


if __name__ == "__main__":
    main()
