#!/bin/bash
# E7 launch chain (docs/E7_direct_sampling_protocol_2026-11-23.md, section 11).
# Verifies the pre-specification hashes (written before any E7 row existed), then runs:
# runner (4 workers) -> sensitivity re-search -> analysis. No QPU use.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.:${PYDEPS:-../.pydeps/core:../.pydeps/numba}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
H=results/e7_prereg_hashes.txt
test -e "$H" || { echo "missing $H: write the hashes before launching"; exit 1; }
grep -v '^#' "$H" | sha256sum -c --quiet || { echo "hash mismatch: code or protocol changed after hashing"; exit 1; }
echo "hashes verified $(date '+%Y-%m-%d %H:%M:%S %z')"
python3 experiments/e7_run.py 4
python3 experiments/e7_sensitivity.py
python3 experiments/e7_analyze.py > results/e7_analyze.log 2>&1
echo "E7 chain done $(date '+%Y-%m-%d %H:%M:%S %z')"
