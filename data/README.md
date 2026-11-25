# Data

All inputs are third-party files, stored unchanged in `external/`. Nothing in `results/` depends on any other input.

## Terms of the third-party files

These files are redistributed here under their sources' terms (checked 25 Nov 2026), not under this repository's
licenses.

- **`external/enigma_hcp/`.** The ENIGMA Toolbox is distributed under the BSD 3-Clause License; its notice is in
  `external/enigma_hcp/LICENSE-ENIGMA-Toolbox.txt`. The matrices are derived from Human Connectome Project data, which
  may be redistributed only under the [WU-Minn HCP Open Access Data Use
  Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)
  (term 4), so these files are shared under those terms. Anyone who uses them must also accept those terms, including
  the acknowledgment wording of term 5.1 in any publication. The manuscript carries that acknowledgment.
- **`external/schaefer_labels/`.** The CBIG repository is distributed under the MIT License; its notice is in
  `external/schaefer_labels/LICENSE-CBIG.txt`.

## `external/enigma_hcp/`: connectomes

- **Content.** Group-average functional (`funcMatrix_*`) and structural (`strucMatrix_*`) cortical connectivity
  matrices of the Human Connectome Project, in the Schaefer 100-, 200- and 400-region parcellations, with region labels.
- **Source.** As distributed with the ENIGMA Toolbox (Larivière et al., *Nat. Methods* 18, 698, 2021;
  https://github.com/MICA-MNI/ENIGMA), from its `hcp_connectivity` matrices.
- **Loading.** `qsr2/networks.py` reads these files. Matrices are symmetrised, negative weights are set to zero, and the
  diagonal is set to zero.
- **Labels.** The structural matrices use the functional label files; `funcLabels_ctx_schaefer_100.csv` and
  `strucLabels_ctx_schaefer_100.csv` are identical.

## `external/schaefer_labels/`: parcel centroids

- **Content.** Centroid coordinates (MNI, RAS) of the Schaefer 2018 parcels in the 7-network order: columns `ROI Label`,
  `ROI Name`, `R`, `A`, `S`.
- **Source.** The CBIG repository (Schaefer et al., *Cereb. Cortex* 28, 3095, 2018;
  https://github.com/ThomasYeoLab/CBIG, `stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/`).
- **Use.** Only `S100_7net.csv` is used, for the cortex maps of Fig. 5.

## SHA-256 checksums

```
27f695aec7c095e95c1b0215ebf6ab232d7e97bd2d26e0e47821b59b2ecfaaf3  external/enigma_hcp/funcLabels_ctx_schaefer_100.csv
dc3766a114ab914ff1d8e4981eb063157eeff45671c524023d9d634e767f88e5  external/enigma_hcp/funcLabels_ctx_schaefer_200.csv
e110788738d34831958abf33ded83d06c2c46de18d5419d85f3bc0e9a3475c07  external/enigma_hcp/funcLabels_ctx_schaefer_400.csv
2aa7422195916c4a450f57442fadcf5c26ca82571cd63c25cd98f7e48b7cdd76  external/enigma_hcp/funcMatrix_ctx_schaefer_100.csv
e1595f75722456151b803720119e6a49ed633e62fd0f1ba03d1c8e108d05fe42  external/enigma_hcp/funcMatrix_ctx_schaefer_200.csv
005390e68dbaccfc74dd4543fd39f6bdd92b5a5f5a7f403aa18e3f5b82be40e9  external/enigma_hcp/funcMatrix_ctx_schaefer_400.csv
27f695aec7c095e95c1b0215ebf6ab232d7e97bd2d26e0e47821b59b2ecfaaf3  external/enigma_hcp/strucLabels_ctx_schaefer_100.csv
dea65da4029e99880e649414ff258f2236e5e9cfae80d7c448dbd4d183ac014f  external/enigma_hcp/strucMatrix_ctx_schaefer_100.csv
71f1111f25a3fd9b14a8647214019e332962924e856dc79edbeffb9ef635c638  external/enigma_hcp/strucMatrix_ctx_schaefer_200.csv
f500dd0f9d613a905b342ce5d7ff404d5f46dbdc5b9f39ce366318011a656571  external/enigma_hcp/strucMatrix_ctx_schaefer_400.csv
a6e6929c337abbf63511536fa13c1f0cba0063d18192d14a8cc8670c55d6245f  external/schaefer_labels/S1000_7net.csv
b59d28adece3f956103b77319e76afd2afb95f2df89d7e7964f75bb53325adab  external/schaefer_labels/S100_7net.csv
60f08eb38abe1313dc87ddd4ffcfb7c60428769a0304bd66cc46d359dd4e0de9  external/schaefer_labels/S200_7net.csv
4207656a0c8855a824f8fb92681204a8368e8a9dd964b066501136c60ecd1d21  external/schaefer_labels/S400_7net.csv
```

Check them with `cd data && sha256sum -c` on the block above.

The non-brain networks (Zachary's karate club, *Les Misérables*) ship with networkx 3.4.2. The LFR benchmarks are
generated from fixed seeds by `experiments/e6_instances.py`.
