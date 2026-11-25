"""E4 tilings (docs/E4_whole_cortex_protocol_2026-11-22.md).

Every canonical Yeo-7 x hemisphere ROI with 3 to 13 parcels, each analysed as its own field-free network
(patches.standalone_patch: J = the ROI's own modularity matrix, h = 0), at beta = beta_tilde / sigma_ROI.
  PRIMARY   Schaefer-100 FC and SC: 13 ROIs (3-13 parcels) tile 98 of the 100 cortical parcels. The 2-parcel ROI
            (RH_Limbic) is excluded: its model has a single coupling, so sigma_ROI and beta are undefined, and a
            bipartition of two parcels has no community structure to estimate.
  SECONDARY Schaefer-200 FC and SC: the 7 ROIs with <= 13 parcels (73 of 200 parcels; the other 7 ROIs have 14-27
            parcels and are beyond exact computation on this machine).
No random numbers are used. Order: primary first, then secondary; within a tiling by size, then name.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from qsr2 import networks, patches  # noqa: E402
from instances import sigma_of  # noqa: E402

PRIMARY = ("FC100", "SC100")
SECONDARY = ("FC200", "SC200")
MIN_N, MAX_N = 3, 13


def tiling(ds):
    W, labels = networks.load_connectome(ds)
    rois = patches.canonical_rois(labels, MIN_N, MAX_N)
    out = []
    for name, ix in sorted(rois.items(), key=lambda kv: (len(kv[1]), kv[0])):
        J, h = patches.standalone_patch(W, ix)
        d = dict(family="E4_tiling", ds=ds, n=len(ix), kind=name, nodes=ix.tolist(), labels=[labels[i] for i in ix],
                 J=J, h=h, sigma=sigma_of(J), n_parcels_total=len(W))
        d["F"] = patches.frustration(J, h)
        out.append(d)
    return out


def all_instances(datasets=PRIMARY + SECONDARY):
    return [d for ds in datasets for d in tiling(ds)]


if __name__ == "__main__":
    for ds in PRIMARY + SECONDARY:
        t = tiling(ds)
        cov = sorted(i for d in t for i in d["nodes"])
        assert len(cov) == len(set(cov))
        print(ds, len(t), "ROIs,", len(cov), "of", t[0]["n_parcels_total"], "parcels:", " ".join(f"{d['kind']}:{d['n']}" for d in t))
