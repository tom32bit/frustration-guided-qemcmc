"""E5 proper budget-guarded driver (docs/E5_hardware_n8_protocol_2026-11-22.md, "QPU budget"): submits ranks 1, 2, 3
in turn via e5_n8.py submit; before each job it checks service.usage() and the 300 s cap rule
(usage so far + largest single-job usage so far <= 300 s; first job estimated at 90 s)."""
import json
import os
import subprocess
import sys

from qiskit_ibm_runtime import QiskitRuntimeService

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
CAP, FIRST_EST, MIN_LEFT = 300.0, 90.0, 100.0


def main():
    service = QiskitRuntimeService()
    used, mx = 0.0, 0.0
    for rank in (1, 2, 3):
        p = os.path.join(RES, f"e5_n8_{rank}_hw.json")
        if not os.path.exists(p):
            u = service.usage()
            left = float(u["usage_remaining_seconds"])
            est = mx if mx else FIRST_EST
            print(f"rank {rank}: E5 usage so far {used:.0f} s, estimate {est:.0f} s, account remaining {left:.0f} s", flush=True)
            if used + est > CAP or left - est < MIN_LEFT:
                print("budget rule: stop before rank", rank, flush=True)
                break
            subprocess.run([sys.executable, os.path.join(HERE, "e5_n8.py"), "submit", str(rank)], check=True)
        d = json.load(open(p))
        q = float(d["usage_quantum_seconds"])
        used += q
        mx = max(mx, q)
        print(f"rank {rank}: job {d.get('job_id')} used {q:.0f} s", flush=True)
    print(f"done: E5 proper QPU usage {used:.0f} s", flush=True)


if __name__ == "__main__":
    main()
