# References

Sources behind the figures and claims in this documentation. Vendor and
self-reported benchmark numbers are marked; they are not independent
measurements. Retrieved September 2026.

## Formats and storage

- Zarr Vectors specification and zarr-vectors-py (format 0.9.4, `gpu-backend`
  branch): <https://github.com/AllenInstitute/zarr_vectors>,
  <https://zarr-vectors-py.readthedocs.io>. Including `ZARR_VECTORS_GPU_RESPONSE.md`
  and `docs/how_to/{gpu,hpc_pipelines}.md` on that branch.
- OME-NGFF RFC-8, *Collections* (under review):
  <https://ngff.openmicroscopy.org/rfc/8/index.html>; PR
  <https://github.com/ome/ngff/pull/343>.
- OME-Zarr 0.6 release (RFC-5 coordinate systems):
  <https://github.com/ome/ngff-spec/releases/tag/0.6>.
- BRIDGE, `COLLECTION_LAYOUT_PLAN.md`, `GPU_VECTORISATION_PLAN.md`,
  `SCALING_REVIEW.md` (internal).
- Icechunk documentation: parallel writes
  <https://icechunk.io/en/latest/understanding/parallel/>, storage
  <https://icechunk.io/en/latest/guides/storage/>; issue #1867 (flat chunk
  directory limits).
- zarr-python 3 configuration and sharding: <https://zarr.readthedocs.io>;
  zarrs benchmarks <https://github.com/zarrs/zarr_benchmarks>.
- NVIDIA GPUDirect Storage release notes
  <https://docs.nvidia.com/gpudirect-storage/release-notes/index.html>;
  nvCOMP benchmarks <https://docs.nvidia.com/cuda/nvcomp/benchmarks.html>.

## Methods

- T. Krüger et al., *The Lattice Boltzmann Method: Principles and Practice*,
  Springer, 2017.
- Z. Guo, C. Zheng, B. Shi, *Discrete lattice effects on the forcing term in
  the lattice Boltzmann method*, Phys. Rev. E 65, 046308 (2002).
- Q. Zou, X. He, *On pressure and velocity boundary conditions for the
  lattice Boltzmann BGK model*, Phys. Fluids 9, 1591 (1997).
- M. Bouzidi, M. Firdaouss, P. Lallemand, *Momentum transfer of a
  Boltzmann-lattice fluid with boundaries*, Phys. Fluids 13, 3452 (2001).
- M. Lehmann et al., *Accuracy and performance of the lattice Boltzmann
  method with 64-bit, 32-bit, and customized 16-bit number formats*, Phys.
  Rev. E 106, 015308 (2022).
- P. Vaněk, J. Mandel, M. Brezina, *Algebraic multigrid by smoothed
  aggregation*, Computing 56 (1996); pyamg <https://github.com/pyamg/pyamg>.

## Ansys

- Fluent theory guide: algebraic multigrid, FAS multigrid and FMG
  initialisation; user guide: partitioning and load balancing, mesh
  adaption; *Fluent GPU solver hardware buying guide* (Ansys Innovation
  Space).
- CFX theory guide §23.7.3, *Algebraic Multigrid*; CFX-Solver Manager,
  chapter 17, *Using the Solver in Parallel*. The notes on out-of-core
  slow-down and non-parallel I/O come from a CFX user.
- Ansys Discovery forum threads on the GPU solver and fidelity setting.

## Benchmarks cited

| Source | Figure | Type |
|---|---|---|
| FluidX3D, <https://github.com/ProjectPhysX/FluidX3D> | H100 SXM D3Q19: 17,602 MLUPS FP32, 29,561 FP16S | self-reported |
| Holzer et al., waLBerla on 1,024 A100s, arXiv 2408.06880 | 2,500 MLUPS/GPU FP64; sparse coronary 1,374 kernel-only | peer-reviewed / preprint |
| Ataei, Salehipour, XLB, arXiv 2311.16080 | 1.43 GLUPS per A100 (JAX) | peer-reviewed |
| Palabos GPU, arXiv 2506.09242 | 75–85 % of roofline on A100; Berea 400³ < 10 min | preprint |
| OHC-1 OpenFOAM benchmark, arXiv 2603.27565 | ~0.1 M cell-iterations/s per core | preprint |
| SPUMA (OpenFOAM GPU), arXiv 2512.22215 | 1 A100 ≈ 200–300 cores | preprint |
| GeoChemFoam on ARCHER2, arXiv 2512.08438 | 8 × 10⁹ voxels in 1,230 s on 81,920 cores | preprint |
| Ansys / NVIDIA / Supermicro / CADFEM blogs and briefs (2022–2024) | Fluent GPU: 1 A100 ≈ 272–400 cores; 8 × H200 34× vs 512 cores | vendor |
| Exxact / ATA, STAR-CCM+ GPU | 8 × H100 ≈ 3,000 cores (25× vs 96 cores) | vendor |
| HemeLB GPU, arXiv 2202.11770 | 90 % efficiency at 6,144 V100s | peer-reviewed |
| Kempner Institute H100 benchmarks | STREAM 3.12 TB/s on H100 SXM | self-reported |

## Data

- HiP-CT coronary lumen mesh: Simpleware ScanIP+FE X-2025.06 Fluent export,
  14,790,642 cells, 1 velocity inlet, 77 pressure outlets; provided by a
  collaborator (not redistributed).
