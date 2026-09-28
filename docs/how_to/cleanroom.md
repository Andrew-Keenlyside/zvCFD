# Clean-room policy

zvCFD is written from published equations and papers, not from the source
code of other solvers. This keeps its licence choice free: it can be
BSD-licensed like zarr-vectors-py. It also keeps any code that a restrictive
licence would bind out of the package.

---

## May be consulted

- Textbooks and papers: Krüger et al., *The Lattice Boltzmann Method*
  (2017); Succi; the TRT, MRT, regularised and cumulant collision papers;
  Bouzidi et al. (2001) on interpolated bounce-back; Zou and He (1997)
  on pressure and velocity boundaries; Guo et al. (2002) on forcing;
  Lehmann et al. (2022) on 16-bit storage formats; multigrid texts (Briggs,
  Trottenberg; Vaněk et al. on smoothed aggregation).
- Vendor documentation that describes behaviour, such as the Ansys Fluent
  and CFX theory guides on AMG, FMG initialisation and partitioning. These
  describe methods, and were used for the context in this documentation.
- Published benchmark numbers, cited as such.
- Permissively licensed code (BSD, MIT, Apache-2.0) *with attribution*:
  zarr-vectors-py, zarr-python, cupy, scipy, pyamg, XLB (Apache-2.0), NVIDIA
  Warp (Apache-2.0).

## May not be consulted for implementation

| Code | Licence | Why |
|---|---|---|
| OpenFOAM | GPL-3.0 | copyleft; a derived work would have to be GPL |
| waLBerla | GPL-3.0 | copyleft |
| Palabos | AGPL-3.0 | network copyleft |
| FluidX3D | custom, non-commercial, no military use | its terms would bind derived code |
| HemeLB | LGPL-3.0 | reading is fine, but keep implementation independent |
| Ansys, STAR-CCM+, M-Star and other commercial solvers | proprietary | no source access; use documentation only |

Reading a paper *about* one of these codes is fine. Porting its kernels
is not.

## Practice

- Each kernel's docstring names the equations and papers it implements
  (see `zvcfd/lbm/_cuda.py`).
- Verification is against analytic solutions (Poiseuille, Womersley) and
  published benchmark cases, not against another code's output files.
- When a technique is known only from a restrictive codebase, write its
  specification from the paper first, then implement from that
  specification.
