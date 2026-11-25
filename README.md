# Frustration-guided quantum-enhanced sampling of community uncertainty in brain networks

Code and data for the paper of the same title, which introduces the Quantum Stability Resolver (QSR). The paper
itself is not part of this repository.

**Authors** (BRAC University, Dhaka, Bangladesh):

- Abdullah Al Shafin, [ORCID 0009-0007-6634-6900](https://orcid.org/0009-0007-6634-6900)
- S. M. Yousuf Iqbal Tomal (corresponding author), [ORCID 0009-0000-3391-3824](https://orcid.org/0009-0000-3391-3824)

QSR samples the uncertainty of community structure in brain networks with quantum-enhanced Markov chain Monte Carlo
(QeMCMC). A quantum circuit proposes moves, a Metropolis step keeps the chain exact, and a frustration-based rule sends
only the sub-networks where quantum proposals help to the quantum device.

This repository contains everything the paper's results rest on:

- the code;
- the input data;
- the stored results;
- the analysis plans, each fixed before its study's data existed, with their SHA-256 fingerprints.

## Contents

```
qsr2/            Python package: models, exact samplers, quantum proposals, direct samplers, projection procedure
experiments/     one script per study step (instances, runs, analyses, post hoc checks)
hardware/        IBM Quantum runs (account check, dry runs, budget-guarded submission, analysis)
tests/           known-answer tests (pytest); tests/fixtures/qsr_v1/ holds two original QSR v1 files used as references
data/            input data (third-party connectomes and parcel centroids; see data/README.md)
results/         raw rows (jsonl), summaries (json, csv), run logs, pre-specification hashes, amendments, manifests
docs/            the analysis plan (protocol) of each study
```

## Studies and their analysis plans

Each protocol names the scripts that run its study and the files they write to `results/`.

| Study | Protocol |
|---|---|
| Survey with stronger classical samplers | [G1b](docs/G1b_hardening_protocol_2026-11-21.md) |
| Survey on partition space | [G1c](docs/G1c_partition_space_protocol_2026-11-21.md) |
| Scaling study (7–12 nodes) and simulation beyond the exact range (12–16 nodes) | [E3](docs/E3_scaling_protocol_2026-11-21.md) |
| Growth by one node (12 → 13 nodes) | [E3c](docs/E3c_exact_n13_protocol_2026-11-22.md) |
| Whole-cortex maps | [E4](docs/E4_whole_cortex_protocol_2026-11-22.md) |
| Hardware pilot (6 nodes) | [E5 pilot](docs/E5_hardware_pilot_protocol_2026-11-21.md) |
| Hardware (8 nodes) | [E5](docs/E5_hardware_n8_protocol_2026-11-22.md) |
| Non-brain networks; projection test | [E6](docs/E6_generality_protocol_2026-11-22.md) |
| Direct sampling | [E7](docs/E7_direct_sampling_protocol_2026-11-23.md) |
| Frustration and the participation-ratio bound | [E8](docs/E8_ipr_protocol_2026-11-24.md) |

The first survey (G1) used `experiments/g1_regime_map.py` and `experiments/g1_analyze.py`; its results are
`results/g1_*`.

## Setup

Python 3.10 was used. Install the pinned dependencies. `requirements-hardware.txt` adds Qiskit and PennyLane, which
only the hardware scripts and their tests need.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # add: -r requirements-hardware.txt for hardware/
pip install -e .                           # makes the qsr2 package importable
```

## Reproduce

```bash
make test        # known-answer tests of every component (about 20 minutes)
make verify      # check every file against the newest SHA-256 manifest in results/
```

- **Running a study.** Run the scripts named in its protocol. The heaviest computations took up to a few hours on a
  4-core laptop.
- **Hardware runs.** These need an IBM Quantum account. Credentials are read from `~/.qiskit` and are never stored in
  this repository.

## Provenance

- **Plans fixed before the data.** Before each study's data existed, its analysis plan, code and any predictions were
  fingerprinted with SHA-256 (`results/*_prereg_hashes.txt`). Check them with `sha256sum -c`. The fingerprints were
  stored locally, not in a public registry.
- **Later changes.** These are recorded as dated amendments (`results/*_amendment*.txt`).
- **Files that no longer match their fingerprints.** Five files changed after they were fingerprinted:
    - `qsr2/direct.py`, `experiments/e7_analyze.py` and `tests/test_e7.py` were fixed after a crash in a secondary
      table. A shot count of 0 now counts as 1, the fewest shots the protocol allows. No endpoint uses that count.
      `results/e7_amendment1.txt` records the cause and their new fingerprints.
    - `qsr2/networks.py` and `experiments/e7_chain.sh` had only paths updated in the restructuring of 23 Nov 2026:
      the data folder (`DATA_DIR`) and the Python environment (`PYTHONPATH`).
- **Restructuring.** The project was restructured into this layout on 23 Nov 2026; `results/restructure_2026-11-23.log`
  records every move.
- **Manifests.** `results/MANIFEST_2026-11-23_manuscript.sha256` records the authors' full working folder before the
  restructuring, so it also lists files that are not published here. The newest manifest covers exactly the files of
  this repository, apart from images and PDFs.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Results, analysis plans and other text:** CC BY 4.0 (see [`LICENSE-CC-BY-4.0.md`](LICENSE-CC-BY-4.0.md)).
- **Third-party data** in `data/external/` keep their original terms. The connectomes come from the ENIGMA Toolbox
  (BSD 3-Clause) under the WU-Minn HCP Open Access Data Use Terms, and the parcel centroids from CBIG (MIT). See
  [`data/README.md`](data/README.md).
