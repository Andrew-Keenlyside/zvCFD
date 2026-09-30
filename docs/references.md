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
  Springer, 2017, doi:10.1007/978-3-319-44649-3.
- P. L. Bhatnagar, E. P. Gross, M. Krook, *A model for collision processes
  in gases*, Phys. Rev. 94, 511 (1954) (BGK collision).
- Y. H. Qian, D. d'Humières, P. Lallemand, *Lattice BGK models for
  Navier–Stokes equation*, Europhys. Lett. 17, 479 (1992) (D3Q19).
- I. Ginzburg, F. Verhaeghe, D. d'Humières, *Two-relaxation-time lattice
  Boltzmann scheme: about parametrization, velocity, pressure and mixed
  boundary conditions*, Commun. Comput. Phys. 3, 427 (2008) (TRT, Λ = 3/16).
- Z. Guo, C. Zheng, B. Shi, *Non-equilibrium extrapolation method for
  velocity and pressure boundary conditions in the lattice Boltzmann
  method*, Chinese Physics 11, 366 (2002) (inlet and outlet patches).
- R. Mei, L.-S. Luo, P. Lallemand, D. d'Humières, *Consistent initial
  conditions for lattice Boltzmann simulations*, Comput. Fluids 35, 855
  (2006).
- Y. I. Cho, K. R. Kensey, *Effects of the non-Newtonian viscosity of blood
  on flows in a diseased arterial vessel. Part 1: Steady flows*,
  Biorheology 28, 241 (1991) (Carreau–Yasuda blood parameters).
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

## Validation references

- Plane and pipe Poiseuille flow, square-duct series solution: F. M. White,
  *Viscous Fluid Flow*, 3rd ed., McGraw-Hill, 2006, §3-3 and eq. 3-48.
- J. R. Womersley, *Method for the calculation of velocity, rate of flow
  and viscous drag in arteries when the pressure gradient is known*, J.
  Physiol. 127, 553 (1955).
- G. I. Taylor, A. E. Green, *Mechanism of the production of small eddies
  from large ones*, Proc. R. Soc. A 158, 499 (1937).
- R. Mei, L.-S. Luo, P. Lallemand, D. d'Humières, *Consistent initial
  conditions for lattice Boltzmann simulations*, Comput. Fluids 35, 855
  (2006).
- R. B. Bird, R. C. Armstrong, O. Hassager, *Dynamics of Polymeric Liquids*,
  vol. 1, 2nd ed., Wiley, 1987 (generalised Newtonian channel flow).
- A. S. Sangani, A. Acrivos, *Slow flow through a periodic array of
  spheres*, Int. J. Multiphase Flow 8, 343 (1982).
- S. Bogner, S. Mohanty, U. Rüde, *Drag correlation for dilute and
  moderately dense fluid-particle systems using the lattice Boltzmann
  method*, Int. J. Multiphase Flow 68, 71 (2015), arXiv:1401.2025 (Table 1:
  Sangani–Acrivos drag and a TRT code's errors at 32³ and 64³).
- M. Schäfer, S. Turek, *Benchmark computations of laminar flow around a
  cylinder*, Notes Numer. Fluid Mech. 52, 547 (1996); high-accuracy 2D-1
  values from the FeatFlow benchmark page
  <https://featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html>
  and V. John, G. Matthies, Int. J. Numer. Meth. Fluids 37, 885 (2001).
- I. Ginzburg, D. d'Humières, *Multireflection boundary conditions for
  lattice Boltzmann models*, Phys. Rev. E 68, 066614 (2003) (TRT, the magic
  parameter and exact walls).

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

## SimVascular

- SimVascular documentation, <https://simvascular.github.io/> (modelling,
  meshing, Python interface, simulation guides).
- svMultiPhysics, <https://github.com/SimVascular/svMultiPhysics> (BSD-3);
  GPU linear solvers via Trilinos/Kokkos: Codoni et al., arXiv 2607.19631.
- svZeroDSolver, <https://github.com/SimVascular/svZeroDSolver> (BSD-3),
  JOSS 10(109):7595 (2025).
- svOneDSolver, <https://github.com/SimVascular/svOneDSolver>.
- Vascular Model Repository, <https://github.com/SimVascular/vascularmodel>
  (316 projects; 87 coronary).
- Menon et al., coronary 3-D vs 0-D cost, arXiv 2409.02247.
- Feiger et al., LBM boundary conditions for image-derived vessels, Int. J.
  Numer. Meth. Biomed. Eng., doi:10.1002/cnm.3198 (2019).

## Software

zvCFD depends on, or is compared against, these packages. Please cite them
too where your work relies on them.

- zarr-vectors-py, A. Keenlyside, F. Collman, Allen Institute (BSD-3),
  <https://github.com/AllenInstitute/zarr-vectors-py>.
- zarr-python, Zarr Developers, <https://zarr.dev> (Zenodo
  doi:10.5281/zenodo.3773449).
- CuPy: R. Okuta et al., *CuPy: A NumPy-compatible library for NVIDIA GPU
  calculations*, LearningSys workshop, NeurIPS 2017.
- NumPy: C. R. Harris et al., Nature 585, 357 (2020),
  doi:10.1038/s41586-020-2649-2.
- SciPy: P. Virtanen et al., Nat. Methods 17, 261 (2020),
  doi:10.1038/s41592-019-0686-2.
- PyAMG: N. Bell, L. N. Olson, J. Schroder, J. Open Source Softw. 7(72),
  4142 (2022), doi:10.21105/joss.04142.
- OpenFOAM: H. G. Weller, G. Tabor, H. Jasak, C. Fureby, Comput. Phys. 12,
  620 (1998), doi:10.1063/1.168744; ESI OpenFOAM v2506 by OpenCFD Ltd.
- SimVascular: A. Updegrove et al., Ann. Biomed. Eng. 45, 525 (2017),
  doi:10.1007/s10439-016-1762-8.
- PyVista: C. B. Sullivan, A. Kaszynski, J. Open Source Softw. 4(37), 1450
  (2019), doi:10.21105/joss.01450; Matplotlib: J. D. Hunter, Comput. Sci.
  Eng. 9, 90 (2007).
- Documentation: Sphinx, Furo and MyST-Parser.

## Data

- Hierarchical phase-contrast tomography: C. L. Walsh, P. Tafforeau et al.,
  *Imaging intact human organs with local resolution of cellular
  structures using hierarchical phase-contrast tomography*, Nat. Methods
  18, 1532 (2021), doi:10.1038/s41592-021-01317-x.
- HiP-CT coronary lumen mesh: Simpleware ScanIP+FE X-2025.06 Fluent export,
  14,790,642 cells, 1 velocity inlet, 77 pressure outlets; provided by a
  collaborator (not redistributed).
