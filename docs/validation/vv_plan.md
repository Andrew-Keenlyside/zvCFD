# Verification and validation of the finite-volume solver

**Question.** How far can the finite-volume solver's answers be trusted,
and for what? This page organises the evidence along the lines of ASME
V&V 20 (verification and validation in CFD) and ASME V&V 40 with the FDA's
credibility guidance (models used for medical decisions). It says what is
established, what is prepared but not yet run, and what is missing.
Details and numbers are on [Finite-volume solver](fv_solver.md).

**Status (30 September 2026).** Code verification is done for the CPU
reference solver and for the GPU kernels that exist. Validation against
benchmarks with published reference data is done for laminar steady flow.
Solution verification tools are built, and so are the cross-code and
experimental validation cases, but the runs that matter (the coronary tree,
the FDA nozzle) wait for the GPU solver.

---

## Context of use

Hemodynamics in image-derived vessels: laminar, incompressible,
isothermal blood flow (Newtonian or shear-thinning), steady or pulsatile,
with flow splits, pressure drops and wall shear stress as the quantities
of interest. The first application is the HiP-CT coronary tree, compared
with the collaborator's Ansys CFX run on the same mesh.

## 1. Code verification: does the code solve its equations, at its order?

| Evidence | Result | Where |
|---|---|---|
| Geometry identities (volumes, closed control volumes) | exact to round-off on every mesh, the coronary mesh included | [Numerics](../spec/fv_numerics.md) |
| Solutions the scheme must reproduce exactly (hydrostatics, Couette) | round-off on every affine element | [Finite-volume solver](fv_solver.md#exact-solutions-the-scheme-must-reproduce) |
| Manufactured solutions: slip walls, no-slip walls, flow crossing a pressure outlet with prescribed traction, Carreau–Yasuda with the full stress, smoothly warped and unstructured (Delaunay) meshes | velocity second order everywhere; pressure 1.5–2.1 (1.5–1.7 with no-slip walls, the equal-order limit) | [Manufactured solutions](fv_solver.md#manufactured-solutions) |
| Exact Navier–Stokes solutions: Kovasznay (steady), Womersley and Ethier–Steinman (unsteady) | second order in space; BDF1 first and BDF2 second order in time | [Exact flows](fv_solver.md#exact-navierstokes-flows) |
| Invariances: rotation, reflection, translation, renumbering, dynamic similarity, mirror symmetry | round-off (the High Resolution limiter is component-wise: 8 × 10⁻⁴ under rotation) | [Invariances](fv_solver.md#invariances-and-balances) |
| Conservation: mass and momentum balances from consistent reactions | round-off on every run; wall force = Δp × area exactly in Stokes pipe flow | [Invariances](fv_solver.md#invariances-and-balances) |
| Time-step independence of steady states (false time step, marching to steady state) | 10⁻¹³ | [Invariances](fv_solver.md#invariances-and-balances) |
| GPU kernels against the CPU reference (gradients, momentum diagonal, residual), fuzzed on all element types | 3 × 10⁻¹⁶ relative; deterministic | [GPU](fv_solver.md#gpu-kernels-against-the-reference) |
| Lagged Rhie–Chow gradient (the GPU and CFX strategy) against the implicit form | same solution to 10⁻¹², in 58–61 outer iterations | [Invariances](fv_solver.md#invariances-and-balances) |

**Open in code verification.** Structured tetrahedral boxes with
prescribed *sliding* velocity on every face carry large boundary-pressure
errors, which pollute the velocity (5× when the exact boundary pressure is
imposed). No-slip walls and flow-through boundaries do not show it. The
GPU linear solve and outer loop do not exist yet, so they are unverified.

## 2. Solution verification: how much numerical error is in a given answer?

| Tool | Status |
|---|---|
| Iterative convergence: RMS change, mass and momentum balances per iteration | built |
| Grid convergence index (Roache; Celik et al. 2008), observed order, asymptotic check (`zvcfd.verification`) | built, tested |
| Uniform conforming refinement of any mesh (`zvcfd.mesh.refine`): the route to a grid study of a mesh that cannot be remeshed | built; demonstrated on a wall-wedge / tet-core pipe |
| Time-step study (halve and double, CFX MG §16.4.2) | built (transient solver); run on Womersley and Ethier–Steinman |
| The coronary tree: residual targets 10⁻⁴/10⁻⁵/10⁻⁶, a three-level GCI on a cut-out subtree (5.7 M → 45 M nodes by refinement), and a two-level GCI on the whole tree | **pending the GPU solver** |

## 3. Validation: are these the right equations and inputs?

| Case | Reference | Result | Status |
|---|---|---|---|
| Kovasznay flow, Re = 40 | exact | second order, 5 × 10⁻⁴ at 32 × 48 | done |
| DFG 2D-1 cylinder, Re = 20 | Schäfer & Turek (1996); FeatFlow | c_D +0.23 %, Δp +0.35 %, c_L +1.6 % at 21,680 nodes | done |
| Lid-driven cavity, Re = 100, 400 | Ghia, Ghia & Shin (1982) | centreline RMS deviation 0.003–0.007 at 64² | done |
| FDA benchmark nozzle, Re_throat = 500 | multi-laboratory PIV (Hariharan et al. 2011) | mesh generator built (`fda_nozzle`), 46 k nodes | **pending**: the GPU solver, and the PIV data (the FDA's hub, `nciphub.org`, no longer resolves) |
| Stenosis with separation | Ahmed & Giddens (1983) | not started | planned |

## 4. Cross-code comparison on identical meshes

| Comparison | Result | Status |
|---|---|---|
| OpenFOAM v2506 on the same mesh (pressure-driven pipe, wall wedges and tet core) | OpenFOAM reads zvCFD's Fluent files (`checkMesh`: Mesh OK). At Re = 50, against the exact flow rate (developed Poiseuille on the polygon): zvCFD −0.70 % and −0.53 %, OpenFOAM −6.8 % and −5.8 % at nc = 6 and 12. The earlier 10 % gap, zvCFD above the exact rate, was zvCFD's momentum entering through the pressure inlet, fixed on 2 October | done |
| Developed pipe through pressure boundaries (exact at every Re) | converges on extruded meshes (+0.56, +0.20, +0.12 % against Stokes at nc = 6, 12, 18) and Delaunay tetrahedra (−0.59, −0.46, −0.23 % at 12, 18, 24); structured tetrahedra leave a 0.5 % floor for every second-order treatment; the opt-in full reconstruction is unstable on coarse meshes (central differencing along the normal) | done; the structured-tetrahedron floor is a property of that mesh |
| OpenFOAM on the coronary mesh | the OpenFOAM run exists ([OpenFOAM](../benchmarks/openfoam.md)) | **pending the GPU solver** |
| Ansys CFX (collaborator), same mesh and boundary table | — | **pending**: the GPU solver, the CFX `.out` file |
| svMultiPhysics on VMR 0066, same `.vtu` | mesh reader built (`zvcfd.mesh.vtk`) | pending: svMultiPhysics is not installed |
| zvCFD LBM against zvCFD FV under refinement | both verified against exact pipe flow separately | pending: joint refinement on one geometry |

## 5. What would still be needed for a regulatory-grade credibility case

Beyond the pending runs: uncertainty quantification of the *inputs*
(segmentation, inlet flow, outlet resistances, blood rheology), which
V&V 40 weighs against the model's influence on the decision. And a
validation comparator closer to the context of use (an in-vitro coronary
phantom or in-vivo flow data). None of that is started.

## Reproducing

```bash
PYTHONPATH=. python benchmarks/validation/fv_cases.py            # every case (hours on one core each)
PYTHONPATH=. python benchmarks/validation/fv_cases.py kovasznay  # one case
FOAM_SIF=esi2506.sif PYTHONPATH=. python benchmarks/fv/openfoam_same_mesh.py
pytest tests/test_fv_validation.py tests/test_fv_properties.py tests/test_fv_gpu.py
```

## References

ASME V&V 20-2009, *Standard for Verification and Validation in
Computational Fluid Dynamics and Heat Transfer*; ASME V&V 40-2018,
*Assessing Credibility of Computational Modeling through Verification and
Validation: Application to Medical Devices*; US FDA, *Assessing the
Credibility of Computational Modeling and Simulation in Medical Device
Submissions* (guidance, 2023); P. J. Roache, J. Fluids Eng. 116, 405 (1994);
I. B. Celik et al., J. Fluids Eng. 130, 078001 (2008); P. Hariharan et al.,
J. Biomech. Eng. 133, 041002 (2011); S. F. C. Stewart et al., Cardiovasc.
Eng. Technol. 3, 139 (2012); U. Ghia, K. N. Ghia, C. T. Shin, J. Comput.
Phys. 48, 387 (1982); S. A. Ahmed, D. P. Giddens, J. Biomech. 16, 505 (1983).
