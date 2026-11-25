"""E3 amendment 13: SEALED routing predictions for the fresh Tier B instances (n = 14, 16), computed from instance
data only (couplings, sigma, F) BEFORE any Tier B outcome for these instances exists. Reads no Tier B rows.

Two-factor rule (from the post hoc Tier A analysis, experiments/e3_posthoc_barrier.py):
  predicted quantum win  <=>  F >= 0.10  and  beta * D >= 1   (beta = 1 / sigma_ROI, beta_tilde = 1)
where D is the critical depth of the single-node-move energy landscape on partition space.
Writes results/e3_tierB_routing_predictions.json (refuses to overwrite).
"""
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
from e3_tierB import tierB_instances  # noqa: E402

F_THR, D_THR = 0.10, 1.0


def main():
    path = os.path.join(RES, "e3_tierB_routing_predictions.json")
    if os.path.exists(path):
        sys.exit(f"{path} exists; predictions are sealed and are not recomputed")
    rows_path = os.path.join(RES, "e3_tierB_rows.jsonl")
    done = [json.loads(l) for l in open(rows_path)] if os.path.exists(rows_path) else []
    fresh_done = [r for r in done if r["n"] in (14, 16)]
    out = []
    for d in tierB_instances([14, 16]):
        E = ising.energies(d["J"])
        bD = critical_depth(E, d["n"]) / d["sigma"]
        out.append({"key": list(key(d)), "family": d["family"], "n": d["n"], "F": d["F"], "beta_D": bD,
                    "predicted_quantum_win": bool(d["F"] >= F_THR and bD >= D_THR)})
    S = {"rule": __doc__, "F_threshold": F_THR, "betaD_threshold": D_THR,
         "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "tierB_rows_at_n14_16_when_written": len(fresh_done), "instances": out,
         "predicted_wins": sum(x["predicted_quantum_win"] for x in out)}
    json.dump(S, open(path, "x"), indent=1)
    print(f"{len(out)} instances, {S['predicted_wins']} predicted wins; Tier B rows at n=14/16 so far: {len(fresh_done)}")


if __name__ == "__main__":
    main()
