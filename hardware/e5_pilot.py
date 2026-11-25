"""E5 hardware pilot (docs/E5_hardware_pilot_protocol_2026-11-21.md).

Usage:  e5_pilot.py dry SC|FC     -> transpile for the chosen backend, run on AerSimulator.from_backend (no QPU time)
        e5_pilot.py submit SC|FC  -> run on the IBM backend (QPU time), save counts + job id + usage
Outputs: results/e5_pilot_<MOD>_<dry|hw>.json  (counts per start state, transpile stats, backend, angles, usage)
Credentials: ~/.qiskit/qiskit-ibm.json. Job tags are generic (no personal or project names).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from qsr2 import networks, patches, proposals, qiskit_circuits  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
SHOTS = 2000
ROI = "LH_SomMot"


def instance(mod):
    W, labels = networks.load_connectome(f"{mod}100")
    ix = patches.canonical_rois(labels, 6, 6)[ROI]
    J, h = patches.standalone_patch(W, ix)
    sigma = float(np.std(J[np.triu_indices(len(J), 1)]))
    return J, sigma, ix


def angles(mod):
    S = json.load(open(os.path.join(RES, "g1c_summary.json")))
    i = S[f"qaoa_best_angle_index_P3_roi_{mod}_n6"]
    return i, proposals.QAOA_GRID[i]


def pick_backend(service):
    best = None
    for b in service.backends(simulator=False, operational=True):
        t = b.target
        op = next(o for o in ("cz", "ecr", "cx") if o in t.operation_names)
        errs = sorted(p.error for p in t[op].values() if p is not None and p.error is not None)
        med = errs[len(errs) // 2]
        if best is None or med < best[1]:
            best = (b, med, op)
    return best


def main():
    mode, mod = sys.argv[1], sys.argv[2]
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService
    J, sigma, ix = instance(mod)
    n = len(J)
    a = proposals.layden_alpha(J, np.zeros(n))
    ai, (b1, b2, g1) = angles(mod)
    circs = [qiskit_circuits.qaoa_mc_circuit(J, x, a, b1, b2, g1) for x in range(2 ** n)]
    service = QiskitRuntimeService()
    backend, med2q, op2q = pick_backend(service)
    pm0 = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=11)
    first = pm0.run(circs[0])
    layout = first.layout.initial_index_layout(filter_ancillas=True)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=11, initial_layout=layout)
    isa = pm.run(circs)
    n2q = [c.count_ops().get(op2q, 0) for c in isa]
    stats = {"backend": backend.name, "median_2q_error": med2q, "two_qubit_gate": op2q, "layout": list(layout),
             "two_qubit_count_min_max": [int(min(n2q)), int(max(n2q))],
             "depth_max": int(max(c.depth() for c in isa)), "active_qubits_max": int(max(
                 len({q for inst in c.data for q in inst.qubits}) for c in isa))}
    print(json.dumps(stats), flush=True)
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "modality": mod, "roi": ROI, "nodes": ix.tolist(), "n": n,
           "sigma": sigma, "beta_tilde": 1.0, "alpha": a, "angle_index": ai, "angles": [b1, b2, g1],
           "shots": SHOTS, "transpile": stats}
    if mode == "dry":
        from qiskit_aer import AerSimulator
        sim = AerSimulator.from_backend(backend, seed_simulator=5)
        res = sim.run(isa, shots=SHOTS).result()
        out["counts"] = [res.get_counts(i) for i in range(len(isa))]
        path = os.path.join(RES, f"e5_pilot_{mod}_dry.json")
    elif mode == "submit":
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.environment.job_tags = ["qemcmc-pilot", f"n{n}"]
        job = sampler.run(isa, shots=SHOTS)
        out["job_id"] = job.job_id()
        print("submitted job", job.job_id(), flush=True)
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
        path = os.path.join(RES, f"e5_pilot_{mod}_hw.json")
    else:
        raise SystemExit("mode must be dry or submit")
    json.dump(out, open(path, "w"), default=str)
    print("wrote", path, "| usage:", out.get("usage_quantum_seconds"), flush=True)


if __name__ == "__main__":
    main()
