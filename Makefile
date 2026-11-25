# Common tasks. On the original machine the dependencies live outside the repository in ../.pydeps;
# elsewhere, install requirements.txt (and pip install -e .), and the PYDEPS path is simply ignored.
PYTHON ?= python3
PYDEPS ?= ../.pydeps/core:../.pydeps/numba:../.pydeps/ibm
export PYTHONPATH := .:$(PYDEPS)

.PHONY: test verify

# known-answer tests of every component (about 20 minutes)
test:
	$(PYTHON) -m pytest -q

# check every file against the newest SHA-256 manifest in results/, chosen by the date in its name
# (git does not keep file times, so the newest by modification time is unreliable in a clone)
verify:
	@m=$$(ls results/MANIFEST_*.sha256 | sort | tail -1); sha256sum -c --quiet $$m && echo "all files match $$m"

# the authors' local targets for the manuscript, which is not part of this repository
-include local.mk
