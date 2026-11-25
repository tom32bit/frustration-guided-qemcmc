"""E7 networks (docs/E7_direct_sampling_protocol_2026-11-23.md). No random numbers; the models are those of E4/E3c.

PRIMARY    the E4 primary tilings: Schaefer-100 FC and SC, 13 canonical ROIs each (3-13 parcels), which tile 98 of
           the 100 cortical parcels (26 networks).
SECONDARY  the E4 secondary tilings (Schaefer-200 FC and SC, the 7 ROIs with <= 13 parcels each) and the E3c
           canonical 13-parcel ROIs of Schaefer-400 (LH_Limbic and RH_Limbic, FC and SC); every other E3c canonical
           ROI is already in an E4 tiling (18 networks).
Each network is its own field-free model (J = own modularity matrix, h = 0) at beta = beta_tilde / sigma_ROI.
Order: primary then secondary, each in its source order. The index in this list seeds the QAOA restarts.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from e3c_instances import canonical13  # noqa: E402
from e4_instances import PRIMARY, SECONDARY, tiling  # noqa: E402


def primary():
    out = [d for ds in PRIMARY for d in tiling(ds)]
    for d in out:
        d["e7_set"] = "primary"
    return out


def secondary():
    out = [d for ds in SECONDARY for d in tiling(ds)] + [d for d in canonical13() if d["ds"] in ("FC400", "SC400")]
    for d in out:
        d["e7_set"] = "secondary"
    return out


def all_networks():
    return primary() + secondary()


if __name__ == "__main__":
    nets = all_networks()
    for i, d in enumerate(nets):
        print(i, d["e7_set"], d["family"], d["ds"], d["kind"], d["n"])
    print(len(nets), "networks:", sum(d["e7_set"] == "primary" for d in nets), "primary,",
          sum(d["e7_set"] == "secondary" for d in nets), "secondary")
