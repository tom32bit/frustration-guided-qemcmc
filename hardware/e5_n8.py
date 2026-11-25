"""E5 proper, n = 8 (docs/E5_hardware_n8_protocol_2026-11-22.md): QAOA-MC proposal rows on IBM Heron for the three
instances fixed in results/e5_selection.json.

Usage:  e5_n8.py transpile           -> compile all instances for the chosen backend, print stats (no QPU time)
        e5_n8.py dry RANK            -> run instance RANK (1..3) on AerSimulator.from_backend (no QPU time)
        e5_n8.py submit RANK         -> run instance RANK on the IBM backend (QPU time), save counts + job id + usage
Outputs: results/e5_n8_<RANK>_<dry|hw>.json  (counts per representative start state, transpile stats, usage).
Partition-space design: one circuit per representative s < 2^(n-1) (top bit 0): the hardware chain canonicalises its
state to the representative before proposing, so these 128 rows define it exactly.
Credentials: ~/.qiskit/qiskit-ibm.json. Job tags are generic (no personal or project names).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from qsr2 import networks, patches, proposals, qiskit_circuits  # noqa: E402

RES = os.path.join(ROOT, "results")
SHOTS = 2000
SEEDS = range(10)                     # transpiler seeds tried on the first circuit; fewest two-qubit gates wins
ANGLE_INDEX = 0                       # (pi/8, pi/8, 0.5): ranked 1/18 at every n in E3 Tier A
BACKEND_FILE = os.path.join(RES, "e5_n8_backend.json")


def instance(rank):
    S = json.load(open(os.path.join(RES, "e5_selection.json")))
    c = next(r for r in S["candidates"] if r.get("selected_rank") == rank)
    W, _ = networks.load_connectome(c["ds"])
    J, _ = patches.standalone_patch(W, np.array(c["nodes"]))
    sigma = float(np.std(J[np.triu_indices(len(J), 1)]))
    assert abs(sigma - c["sigma"]) < 1e-12 and abs(patches.frustration(J) - c["F"]) < 1e-12
    return c, J


def circuits(J):
    n = len(J)
    a = proposals.layden_alpha(J, np.zeros(n))
    b1, b2, g1 = proposals.QAOA_GRID[ANGLE_INDEX]
    return a, (b1, b2, g1), [qiskit_circuits.qaoa_mc_circuit(J, s, a, b1, b2, g1) for s in range(2 ** (n - 1))]


def pick_backend(service):
    """Lowest median two-qubit error among visible operational devices; fixed at the first call (file) so that
    every instance and the dry runs use the same device."""
    if os.path.exists(BACKEND_FILE):
        b = json.load(open(BACKEND_FILE))
        return service.backend(b["name"]), b
    best = None
    for be in service.backends(simulator=False, operational=True):
        t = be.target
        op = next(o for o in ("cz", "ecr", "cx") if o in t.operation_names)
        errs = sorted(p.error for p in t[op].values() if p is not None and p.error is not None)
        med = errs[len(errs) // 2]
        if best is None or med < best[1]:
            best = (be, med, op)
    info = {"name": best[0].name, "median_2q_error_at_choice": best[1], "two_qubit_gate": best[2],
            "chosen_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    json.dump(info, open(BACKEND_FILE, "x"), indent=1)
    return best[0], info


def compile_all(circs, backend, op2q):
    from qiskit.transpiler import generate_preset_pass_manager
    trials = []
    for sd in SEEDS:
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=sd)
        c0 = pm.run(circs[0])
        trials.append((c0.count_ops().get(op2q, 0), c0.depth(), sd, c0))
    n2q0, _, seed, c0 = min(trials, key=lambda x: (x[0], x[1], x[2]))
    layout = c0.layout.initial_index_layout(filter_ancillas=True)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed, initial_layout=layout)
    isa = pm.run(circs)
    n2q = [c.count_ops().get(op2q, 0) for c in isa]
    stats = {"seed_transpiler": seed, "seed_trials_2q": [t[0] for t in trials], "layout": list(layout),
             "two_qubit_count_min_max": [int(min(n2q)), int(max(n2q))],
             "depth_max": int(max(c.depth() for c in isa)),
             "active_qubits_max": int(max(len({q for inst in c.data for q in inst.qubits}) for c in isa))}
    return isa, stats


def main():
    mode = sys.argv[1]
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    backend, binfo = pick_backend(service)
    op2q = binfo["two_qubit_gate"]
    ranks = [1, 2, 3] if mode == "transpile" else [int(sys.argv[2])]
    for rank in ranks:
        c, J = instance(rank)
        n = len(J)
        a, (b1, b2, g1), circs = circuits(J)
        isa, stats = compile_all(circs, backend, op2q)
        stats.update({"backend": backend.name, "two_qubit_gate": op2q})
        print(f"rank {rank} ({c['family']} {c['kind']}):", json.dumps(stats), flush=True)
        if mode == "transpile":
            continue
        out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "rank": rank, "key": c["key"], "ds": c["ds"],
               "nodes": c["nodes"], "n": n, "sigma": c["sigma"], "beta_tilde": 1.0, "alpha": a,
               "angle_index": ANGLE_INDEX, "angles": [b1, b2, g1], "shots": SHOTS, "transpile": stats}
        if mode == "dry":
            from qiskit_aer import AerSimulator
            sim = AerSimulator.from_backend(backend, seed_simulator=5)
            res = sim.run(isa, shots=SHOTS).result()
            out["counts"] = [res.get_counts(i) for i in range(len(isa))]
            path = os.path.join(RES, f"e5_n8_{rank}_dry.json")
        elif mode == "submit":
            from qiskit_ibm_runtime import SamplerV2
            path = os.path.join(RES, f"e5_n8_{rank}_hw.json")
            if os.path.exists(path):
                sys.exit(f"{path} exists; not resubmitting")
            sampler = SamplerV2(mode=backend)
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.environment.job_tags = ["qemcmc", f"n{n}"]
            job = sampler.run(isa, shots=SHOTS)
            out["job_id"] = job.job_id()
            print("submitted job", job.job_id(), flush=True)
            json.dump({"job_id": job.job_id(), "rank": rank, "submitted": out["date"]},
                      open(os.path.join(RES, f"e5_n8_{rank}_job.json"), "w"))
            res = job.result()
            out["counts"] = [r.data.c.get_counts() for r in res]
            try:
                out["usage_quantum_seconds"] = job.usage()
            except Exception as e:
                out["usage_quantum_seconds"] = f"unavailable: {str(e)[:80]}"
            try:
                out["metrics"] = job.metrics()
            except Exception:
                pass
        else:
            raise SystemExit("mode must be transpile, dry or submit")
        json.dump(out, open(path, "w"), default=str)
        print("wrote", path, "| usage:", out.get("usage_quantum_seconds"), flush=True)


if __name__ == "__main__":
    main()
