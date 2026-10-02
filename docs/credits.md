# Credits and citing

zvCFD is built on other people's work. This page says what, and how to
cite it. [References](references.md) lists every source behind the
documentation.

## Citing zvCFD

Cite the software with the repository's
[`CITATION.cff`](https://github.com/Andrew-Keenlyside/zvCFD/blob/main/CITATION.cff)
(GitHub's *Cite this repository* button reads it):

```text
Keenlyside, A. zvCFD: GPU-based computational fluid dynamics backed by
Zarr Vectors (version 0.0.1.dev0). https://github.com/Andrew-Keenlyside/zvCFD
```

Then cite what your results depend on, from the tables below.

## Foundations

| Work | What zvCFD uses | Cite |
|---|---|---|
| [Zarr Vectors / zarr-vectors-py](https://github.com/AllenInstitute/zarr-vectors-py) (Allen Institute; A. Keenlyside, F. Collman; BSD-3) | every brick store, written through `zarr_vectors.building`; GPU decode | the repository |
| [OME-NGFF](https://ngff.openmicroscopy.org) RFC-8 *Collections*, RFC-5 coordinate systems | the run collection layout | the RFCs |
| [zarr-python](https://zarr.dev), [CuPy](https://cupy.dev), [NumPy](https://numpy.org), [SciPy](https://scipy.org), [PyAMG](https://github.com/pyamg/pyamg) | arrays, GPU kernels, CPU numerics, the lubrication solve | [Software](references.md#software) |
| [NVIDIA AmgX](https://github.com/NVIDIA/AMGX) (BSD-3), built from source and called through a thin ctypes binding | the finite-volume solver's algebraic multigrid (`zvcfd.fv.linear`) | M. Naumov et al., *AmgX: a library for GPU accelerated algebraic multigrid and preconditioned iterative methods*, SIAM J. Sci. Comput. 37, S602 (2015) |

## Methods

The solver is a clean-room implementation from the literature
([policy](how_to/cleanroom.md)); each kernel names the equations it
implements.

| Method | Source |
|---|---|
| Lattice-Boltzmann method, pull streaming | Krüger et al., *The Lattice Boltzmann Method*, Springer (2017) |
| BGK collision; D3Q19 lattice | Bhatnagar, Gross & Krook (1954); Qian, d'Humières & Lallemand (1992) |
| TRT collision, magic parameter Λ = 3/16 | Ginzburg & d'Humières (2003); Ginzburg, Verhaeghe & d'Humières (2008) |
| Body forcing | Guo, Zheng & Shi, Phys. Rev. E 65, 046308 (2002) |
| Inlet and outlet patches (non-equilibrium extrapolation) | Guo, Zheng & Shi, Chinese Physics 11, 366 (2002) |
| Pressure outlets on caps (anti-bounce-back on the crossing links) | Ginzburg, Verhaeghe & d'Humières, Commun. Comput. Phys. 3, 427 (2008); Krüger et al. (2017), ch. 5 |
| Interpolated (sub-voxel) walls, on the roadmap | Bouzidi, Firdaouss & Lallemand (2001) |
| Consistent initial conditions | Mei, Luo, Lallemand & d'Humières (2006) |
| Carreau–Yasuda blood rheology, default parameters | Cho & Kensey (1991) |
| 16-bit population storage | Lehmann et al. (2022) |
| Smoothed-aggregation multigrid | Vaněk, Mandel & Brezina (1996) |
| Element-based finite volumes (finite-volume solver) | Schneider & Raw, Numer. Heat Transfer 11, 363 (1987) |
| Rhie–Chow pressure–velocity coupling; its time-step-independent form | Rhie & Chow, AIAA J. 21, 1525 (1983); Choi, Numer. Heat Transfer B 36, 545 (1999) |
| High Resolution limiter | Barth & Jespersen, AIAA paper 89-0366 (1989) |
| Additive-correction multigrid | Hutchinson & Raithby (1986); Raw, AIAA paper 96-0297 (1996); Notay (2010) |
| Flexible GMRES | Saad, SIAM J. Sci. Comput. 14, 461 (1993) |
| SIMPLE-type block preconditioner | Patankar (1980); Elman, Silvester & Wathen (2014) |
| Backflow stabilisation | Esmaily Moghadam et al., Comput. Mech. 48, 277 (2011) |
| Open-loop coronary and RCR outlets | Kim et al., Ann. Biomed. Eng. 38, 3195 (2010); Westerhof et al. (2009) |
| TAWSS, OSI, RRT | He & Ku (1996); Himburg et al. (2004) |

## Validation

| Case | Source |
|---|---|
| Plane and pipe Poiseuille, square duct | White, *Viscous Fluid Flow* (2006) |
| Pulsatile pipe flow | Womersley (1955) |
| Decaying vortices | Taylor & Green (1937) |
| Periodic sphere arrays | Sangani & Acrivos (1982); Bogner, Mohanty & Rüde (2015) |
| Flow past a cylinder (DFG 2D-1, 2D-2) | Schäfer & Turek (1996); John & Matthies (2001) |
| Kovasznay flow; Ethier–Steinman flow | Kovasznay (1948); Ethier & Steinman (1994) |
| Lid-driven cavity | Ghia, Ghia & Shin (1982) |
| Solution verification (GCI) | Roache (1994); Celik et al. (2008) |

## Comparisons and data

- [OpenFOAM](https://www.openfoam.com) (ESI OpenFOAM v2506, OpenCFD Ltd.,
  GPL-3.0) is run as a separate program for the
  [OpenFOAM benchmarks](benchmarks/openfoam.md); zvCFD contains none of its
  code. OpenFOAM is a trademark of OpenCFD Ltd.; zvCFD is not approved or
  endorsed by OpenCFD.
- [SimVascular](https://simvascular.github.io) (Updegrove et al. 2017)
  results for model 0066_H_CORO_H come from the
  [Vascular Model Repository](https://www.vascularmodel.com): Wilson, Ortiz
  & Johnson, J. Med. Devices 7(4):040923 (2013), doi:10.1115/1.4025983; data
  doi:10.25740/dh173bw0673. Its coronary outlet conditions follow Kim et al.,
  Ann. Biomed. Eng. 38, 3195 (2010). *The data used herein was provided in whole or
  in part with Federal funds from the National Library of Medicine under
  Grant No. R01LM013120, and the National Heart, Lung, and Blood
  Institute, National Institutes of Health, Department of Health and Human
  Services, under Contract No. HHSN268201100035C.*
- The HiP-CT coronary geometry was imaged by hierarchical phase-contrast
  tomography at the ESRF (Walsh et al., Nat. Methods 18, 1532 (2021)), and
  segmented and meshed in Simpleware ScanIP by a collaborator. It is not
  redistributed.
- Published performance figures (FluidX3D, waLBerla, XLB, Palabos, HemeLB,
  Ansys, STAR-CCM+ and others) are cited where they are used and
  [listed with their provenance](references.md#benchmarks-cited).

## Trademarks

Ansys, Fluent and CFX are trademarks of ANSYS, Inc.; STAR-CCM+ of Siemens;
NVIDIA, CUDA and H100 of NVIDIA Corporation. They are named for comparison
only.
