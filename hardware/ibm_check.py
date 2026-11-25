"""Read-only IBM Quantum check (no QPU time): account plan/usage, visible backends, qubits, queue, 2q error.
Credentials come from ~/.qiskit/qiskit-ibm.json (QiskitRuntimeService.save_account); nothing secret is printed."""
import json

from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
try:
    u = service.usage()
    print("usage:", json.dumps({k: v for k, v in u.items() if "instance" not in k.lower()}, default=str))
except Exception as e:
    print("usage: unavailable:", str(e)[:120])
for b in service.backends(simulator=False):
    st = b.status()
    t = b.target
    errs = []
    for op in ("cz", "ecr", "cx"):
        if op in t.operation_names:
            errs = [p.error for p in t[op].values() if p is not None and p.error is not None]
            break
    med = sorted(errs)[len(errs) // 2] if errs else None
    ro = [t["measure"][(q,)].error for q in range(b.num_qubits) if t["measure"].get((q,)) and t["measure"][(q,)].error is not None]
    print(f"{b.name:16s} qubits={b.num_qubits:4d} operational={st.operational} pending_jobs={st.pending_jobs:5d} "
          f"2q-gate={op} median-2q-error={med if med is None else round(med, 5)} "
          f"median-readout-error={round(sorted(ro)[len(ro) // 2], 4) if ro else None}")
