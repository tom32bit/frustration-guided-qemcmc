"""E4: SEALED routing decisions for every tiling ROI, computed from instance data only (couplings, sigma, F) BEFORE
any E4 exact row exists. Same two-factor rule and thresholds as E3 amendment 13 and E3c:
  routed to the quantum proposal  <=>  F >= 0.10  and  beta * D >= 1
with D the critical depth of the single-node-move landscape on partition space. Primary decisions use
beta = 1 / sigma_ROI (beta_tilde = 1); the beta_tilde = 2 secondary uses beta = 2 / sigma_ROI.
Writes results/e4_routing_predictions.json (refuses to overwrite)."""
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
from e4_instances import all_instances  # noqa: E402

F_THR, D_THR = 0.10, 1.0


def main():
    path = os.path.join(RES, "e4_routing_predictions.json")
    if os.path.exists(path):
        sys.exit(f"{path} exists; routing decisions are sealed and are not recomputed")
    rows_path = os.path.join(RES, "e4_exact_rows.jsonl")
    done = sum(1 for _ in open(rows_path)) if os.path.exists(rows_path) else 0
    out = []
    for d in all_instances():
        D = critical_depth(ising.energies(d["J"]), d["n"])
        rec = {"key": list(key(d)), "ds": d["ds"], "roi": d["kind"], "n": d["n"], "F": d["F"], "D": D}
        for bt in (1.0, 2.0):
            bD = bt * D / d["sigma"]
            rec[f"beta_D_b{bt}"] = bD
            rec[f"routed_b{bt}"] = bool(d["F"] >= F_THR and bD >= D_THR)
        out.append(rec)
    S = {"rule": __doc__, "F_threshold": F_THR, "betaD_threshold": D_THR,
         "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "e4_exact_rows_when_written": done, "instances": out,
         "routed_b1.0": {ds: [x["roi"] for x in out if x["ds"] == ds and x["routed_b1.0"]] for ds in sorted({x["ds"] for x in out})}}
    json.dump(S, open(path, "x"), indent=1)
    for ds, v in S["routed_b1.0"].items():
        print(ds, "routed at beta_tilde = 1:", len(v), v)
    print("E4 exact rows when written:", done)


if __name__ == "__main__":
    main()
