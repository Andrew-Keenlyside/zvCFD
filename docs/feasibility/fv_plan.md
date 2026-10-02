# Plan: a CFX-style finite-volume solver

**Status: in progress (30 September 2026).** Phases 1–5 are built and
tested on the workstation GPU (RTX A2000 12 GB): the GPU solver with AmgX
and zvCFD's own ACM, hemodynamic boundary conditions with implicitly
coupled lumped outlets, transient runs, `zvcfd run` on meshes, and a
partitioned solver run as partitions of one GPU. Gates A and B are passed.
What remains needs inputs or hardware: the CFX setup and Theory Guide
(Phase 0), the coronary case on one H100 (Phases 3–4, which needs the H100
job's go-ahead), and multi-GPU scaling (Phase 5). See [Progress](#progress). This page plans a
second solver for zvCFD. It uses the same method family as Ansys CFX: an
element-based finite-volume method with unknowns at the mesh nodes, a
coupled solve for pressure and velocity, and algebraic multigrid. It runs
on the GPU, reads the Ansys mesh itself, and uses Zarr Vectors for input
and output. It would sit beside the lattice-Boltzmann solver, not replace
it.

The plan has seven phases, three to five months to a validated steady
solver on one GPU, and six to nine months to the full solver including
multi-GPU runs. Every figure marked *estimate* is a paper figure until the
phase that measures it.

This reverses one line of the [Roadmap](roadmap.md), which lists
unstructured finite volumes under *Not planned*.

---

## Progress

| Built | Module | Evidence |
|---|---|---|
| Lumped outlets (RCR, open-loop coronary), shared by both solvers; svSolver `cort.dat`/`rcrt.dat` readers | `zvcfd.lumped` | order 2.00 (trapezoidal) against the exact impedance; [Boundary conditions](../spec/boundary_conditions.md#lumped-outlet-models) |
| Full Fluent reader (ASCII, binary), element reconstruction, writer; SimVascular `.vtu` reader | `zvcfd.mesh` | coronary mesh in 53 s; exact round trips; [Mesh stores](../spec/mesh_store.md#reading-fluent-meshes) |
| Validation-mesh generators (boxes; butterfly tubes in hex, wedge, tet, and wall-wedge + tet-core) | `zvcfd.mesh.generate` | conforming, exact volumes |
| Shape functions and median-dual geometry, all four element types | `zvcfd.fv.geometry` | volumes and closure exact to round-off; [Finite-volume numerics](../spec/fv_numerics.md) |
| Coupling pattern, memory model, `zvcfd mesh-info --volume` | `zvcfd.fv.pattern` | gate A below |
| `zvcfd:fv-mesh` volume store, boundary mesh store | `zvcfd.io.mesh_store` | exact round trips; [Mesh stores](../spec/mesh_store.md) |
| CPU reference solver: coupled, Rhie–Chow (with the transient term), upwind / blend / High Resolution, walls, velocity inlets and moving walls, pressure outlets (prescribed traction, consistent backflow momentum), symmetry planes; steady, false-time-step and transient (BDF1, BDF2 with coefficient loops); consistent reaction forces; mass and momentum balances | `zvcfd.fv.reference` | [Finite-volume validation](../validation/fv_solver.md) |
| Generalised-Newtonian models in SI (Carreau–Yasuda, Cross, power law, Casson; shear-rate clips) | `zvcfd.rheology` | outlet-crossing Carreau–Yasuda manufactured solution, second order |
| GPU kernels: nodal gradients, momentum diagonal, residual; element colouring, no atomics | `zvcfd.fv.gpu`, `zvcfd.fv._cuda` | 3 × 10⁻¹⁶ against the CPU reference on every element type |
| Uniform conforming refinement; observed order, Richardson, GCI | `zvcfd.mesh.refine`, `zvcfd.verification` | [Solution verification](../validation/fv_solver.md#solution-verification-on-a-fixed-mesh) |
| Validation generators: Delaunay tetrahedra, warped boxes, DFG cylinder channel, FDA nozzle | `zvcfd.mesh.generate` | [Verification and validation](../validation/vv_plan.md) |
| GPU solver: block assembly (element colouring, first-fit), lagged Rhie–Chow, CFX-style outer loop and residuals, false time step, BDF1/BDF2 with variable steps, dimensionless rows, boundary pressure level, consistent-reaction forces and wall shear | `zvcfd.fv.solver`, `zvcfd.fv.gpu` | the CPU reference's solutions to 10⁻¹¹ (host direct, ACM, AmgX); [Finite-volume numerics](../spec/fv_numerics.md#the-outer-loop-on-the-gpu-zvcfdfvsolver) |
| Linear solvers behind one interface: AmgX (ctypes, double or mixed precision), own ACM, FGMRES with rank-one terms | `zvcfd.fv.linear` | gate B below |
| Flow-rate inlets with the developed profile of the cap; backflow stabilisation; lumped outlets coupled implicitly (rank-one); average static pressure; openings; TAWSS/OSI/RRT | `zvcfd.fv.solver`, `zvcfd.fv.profiles`, `zvcfd.fv.wss` | exact checks in `tests/test_fv_boundaries.py`; [Finite-volume validation](../validation/fv_solver.md) |
| `zvcfd run` with `solver.method: fv`: Fluent, `.vtu` or SimVascular mesh-complete input; mesh and boundary stores, node-field snapshots, TAWSS on the boundary store, monitors, patch table | `zvcfd.fv.run`, `zvcfd.io.node_fields` | `tests/test_fv_run.py` |
| Partitions (whole chunks along a Morton curve, one-element halo) and the partitioned solver (halo exchange, global FGMRES, block-Jacobi AMG) | `zvcfd.fv.partition`, `zvcfd.fv.multi` | one partition = one GPU to round-off; 2–4 partitions to the solver tolerance |
| Time-step study (observed temporal order, GCI) | `zvcfd.verification.time_step_study` | BDF2 order 2.0 |
| Momentum entering through pressure boundaries: the faces' corrections carried by the inflow node's boundary flux, plus their cross-stream offset (the full second-order reconstruction opt-in) | `zvcfd.fv.solver`, `zvcfd.fv.boundary_advection` | converging on extruded and Delaunay meshes; [developed pipe](../validation/fv_solver.md#developed-pipe-flow-through-pressure-boundaries) |
| Speed review: fused kernels, GPU set-up, fused boundary pass, AmgX `gs` with an `auto` probe | `zvcfd.fv.fast`, `zvcfd.fv._cuda_fast`, `zvcfd.fv._cuda_bc`, `zvcfd.fv.linear` | [FV solver speed](../benchmarks/fv_speed.md) |

**Gate A: passed.** Measured on the coronary mesh (`benchmarks/fv/mesh_report.py`):
18.7 blocks per row (median 18, p99 27, max 46),
106.0 M blocks, 227 M flux points (two per ip
face). By the itemisation below: **38.8 GB** in double precision with ILU(0),
24.8 GB with DILU, 23.2 GB mixed, 16.0 GB mixed with DILU.
One H100 holds it in double precision.

**Gate B: passed; AmgX is the default, ACM the check.** Measured on the
prism-layered tube (`benchmarks/fv/gate_b.py`, steady Re = 100, High
Resolution, `benchmarks/results/fv/gate_b.json`), each outer iteration
solving to a tenfold residual reduction:

| Nodes | Back end | Outer iterations | Linear its / outer | Per outer | of which linear (setup + solve) | CuPy pool |
|---:|---|---:|---:|---:|---:|---:|
| 169,377 | ACM | 25 | 4.7 | 1.61 s | 0.92 s | 3.5 GB |
| 169,377 | AmgX | 24 | 2.0 | 0.81 s | 0.11 s | 2.4 GB |
| 328,601 | ACM | 24 | 6.1 | 3.21 s | 1.79 s | 5.4 GB |
| 328,601 | AmgX | 23 | 2.3 | 1.66 s | 0.22 s | 2.5 GB |
| 328,601 | AmgX mixed | 24 | 2.1 | 2.17 s | 0.30 s | 2.5 GB |
| 565,297 | AmgX mixed | 24 | 2.8 | 3.87 s | 0.57 s | 4.3 GB |

*Re-measured 1 October*, after the speed review and a fix to the tube
generator. Its wedge layers had been coarsest at the wall; now they are
finest there. With the current code on an idle GPU, AmgX (`gs`) takes
24–25 outer iterations and **0.084, 0.142 and 0.254 s per outer
iteration** at 22 k, 72 k and 169 k nodes; mixed precision is about 10 %
slower. ACM still converges at 22 k and 72 k nodes, but not at 169 k,
where the wall layers are now thin
([FV solver speed](../benchmarks/fv_speed.md#where-the-time-goes)).

A single solve to 10⁻⁶ reduces the residual by 0.51–0.62 per iteration
with AmgX and by 0.68–0.82 with ACM, whose rate degrades with size.
AmgX's linear work is 8× cheaper, and it needs less memory. (These runs
used AmgX's usual DILU smoother. The default is now symmetric
multicolour Gauss–Seidel with pressure-weighted aggregation. It solves
the DFG slab, where DILU stalls, in 2 iterations to the outer loop's 0.1:
see
[numerics](../spec/fv_numerics.md#linear-solvers-on-thin-cells-zvcfdfvlinear).) Its
hierarchy lives outside the CuPy pool, which the mixed mode (single
precision hierarchy) halves; that is what fits 565 k nodes on this card,
which the desktop shares. Both back ends reach the same solutions. The
1 M-node tube of the plan does not fit a shared 12 GB card; on it, only
the mixed mode was tried up to 565 k nodes. Three defects found on the
way are fixed: full hierarchy reuse froze the first matrix's coarse
operators and stalled FGMRES; outer iterations counted a stalled linear
solve as converged; and rows in SI units let one equation dominate the
Krylov norm ([numerics](../spec/fv_numerics.md#the-outer-loop-on-the-gpu-zvcfdfvsolver)).

**Deviation from the plan.** Each ip face carries **two flux points**, one
per planar half, rather than one. With one point per non-planar face,
fluxes of linear fields are exact only at interior nodes; with two they
are exact on every control volume of an affine element, so walls and
inlets get the same consistency as the interior. The matrix pattern is
unchanged; assembly does twice the flux-point work.

**Open issue, characterised.** With prescribed *sliding* velocity on every
face, structured tetrahedral and wedge meshes carry large boundary-pressure
errors. With no-slip walls, flow-through boundaries or pressure outlets,
every element type converges alike: velocity second order, and pressure at the
equal-order limit (1.5–2). The CFX Theory Guide's Rhie–Chow form may still
change the boundary treatment (Phase 0).

---

## Terms

**Node**
: A vertex of the mesh. Every unknown (u, v, w, p) lives at a node.

**Element**
: A tetrahedron, prism (wedge), pyramid or hexahedron of the mesh.

**Control volume**
: The volume that belongs to one node. It is built from one piece of each
  element around the node (the median dual). The flux balance is written
  over it.

**Integration point (ip)**
: A point on a face between two control volumes, inside one element,
  where fluxes are evaluated. A tetrahedron has 6 integration points, a
  prism 9 and a hexahedron 12.

**Coupled solve**
: u, v, w and p are solved together as one linear system of 4 × 4 blocks,
  one block row per node, rather than one equation after another.

**Additive-correction multigrid (ACM)**
: Algebraic multigrid in which coarse equations are formed by adding up
  the equations of groups of fine nodes. It is the multigrid that CFX
  describes.

**False time step**
: A pseudo time step that relaxes a steady solve (CFX's *timescale*).

**Coefficient loop**
: One re-linearisation and linear solve inside a physical time step.
  CFX suggests 3–5 per step.

---

## What done means

**In scope.** The solver supports:

- incompressible, isothermal, laminar flow;
- tetrahedral, prism, pyramid and hexahedral elements, as in the Fluent
  meshes that Simpleware and SimVascular export;
- steady solves (false time step) and transient solves (second-order
  backward Euler with coefficient loops);
- Upwind, Specified Blend and High Resolution advection;
- Newtonian fluids and CFX's generalised-Newtonian models (Carreau–Yasuda
  first; then Casson, Cross and power law), with shear-rate clips;
- walls, velocity and mass-flow inlets with profiles and waveforms, static
  and average-static pressure outlets, openings, RCR and open-loop
  coronary outlets, and backflow stabilisation;
- wall shear stress on the mesh surface;
- one GPU first, then several;
- outputs in the same RFC-8 run collection as the LBM solver.

**Out of scope:** turbulence models, heat transfer, multiphase flow,
moving meshes and fluid–structure interaction, compressible flow, and a
CFX Expression Language interpreter (waveform tables cover the
hemodynamic uses).

**The test of done.** On the HiP-CT coronary mesh, with the same inputs
as the collaborator's CFX run, zvCFD's finite-volume solver reproduces
CFX's outlet flow splits and pressure drop. The targets are in
[Decision gates](#decision-gates).

---

## What the coronary mesh implies

Measured from `mesh 1.msh` (the Simpleware export, 1.9 GB, ASCII) on 29
September 2026:

| Quantity | Value | Consequence |
|---|---|---|
| Nodes | 5,673,949 | 5.67 M block rows, 22.7 M unknowns. The system is sized by nodes, not by the 14.8 M cells |
| Cells | 14,790,642 | The zone header says element type 7 (polyhedral), but every face has 3 or 4 nodes, so the reader has to work out the element type from the faces |
| Interior faces | 20,506,289 triangles, 12,361,072 quadrilaterals | By counting faces: about 6.55 M tetrahedra and 8.24 M prisms. Prisms are the inflation layers, and they need their own shape functions from the start |
| Wall faces | 1,648,880 triangles | The wall-shear-stress surface |
| Boundary zones | 1 inlet, 77 outlets, 1 wall | Same patch names as the LBM runs, so the same patch rules apply |
| Face records | node list plus the two neighbouring cells | Fluent stores which cells each face separates, not which nodes each cell has. The reader rebuilds each element's nodes from its faces |

The current reader (`zvcfd.io.fluent_msh`) skips interior faces and parses
boundary faces in Python. It needs a vectorised parser for the 32.9 M
interior faces, and support for binary sections (`2010`, `2012`, `2013`).

---

## Design

### Modules

| Module | Contents | Reuses |
|---|---|---|
| `zvcfd/mesh/` (new) | `UnstructuredMesh` (nodes, elements by type, boundary faces by zone); full Fluent reader; `.vtu` reader through pyvista/VTK (BSD) for SimVascular meshes; mesh generators for validation; quality checks | `zvcfd.io.fluent_msh` zone table, `FaceZone` |
| `zvcfd/fv/geometry.py` | shape functions and derivatives for tetrahedra, prisms, pyramids and hexahedra; median-dual control volumes; integration-point area vectors; boundary sub-faces | — |
| `zvcfd/fv/reference.py` | a CPU assembler and solver in numpy and scipy, float64. It is the test oracle for the GPU kernels on small meshes | — |
| `zvcfd/fv/_cuda.py`, `assemble.py` | GPU assembly kernels (CuPy `RawKernel`), one family per element type | the `_cuda.py` templating pattern of `zvcfd.lbm` |
| `zvcfd/fv/linear.py` | the block linear solver interface; a thin ctypes binding to the AmgX C API; zvCFD's own additive-correction multigrid | `benchmarks/amg_check.py` as a harness |
| `zvcfd/fv/solver.py` | `CoupledFV`: outer loop, steady and transient, CFX-style residuals and imbalances | — |
| `zvcfd/fv/boundary.py` | `FaceBoundarySet`: patch faces, profiles, boundary kernels | `zvcfd.boundary.Patch` unchanged (SI, geometry-agnostic) |
| `zvcfd/lumped.py` (new, shared) | 0-D outlet models: RCR, open-loop coronary (Kim et al. 2010), implicit coupling coefficients | moves `Patch.rcr_pressure` here; used by both solvers |
| `zvcfd/rheology.py` (new, shared) | viscosity models in SI | `CarreauYasuda.from_si` keeps its interface |
| `zvcfd/io/mesh_store.py` | the `zvcfd:fv-mesh` store and node-field snapshots | `zvcfd.io.fields` three-phase writes |
| `zvcfd/run.py`, `config.py`, `cli.py` | `solver.method: fv`, an `fv:` config section, `zvcfd mesh-info` element and memory report, `zvcfd plan` for FV | the existing run collection and patch rules |

### Mesh ingest (Phase 1)

1. Read nodes, faces with their two neighbouring cells, and zones. Parse
   hexadecimal in bulk with numpy: split the tokens, then decode them
   through a lookup table. Handle ASCII and binary sections. Target: under
   2 minutes for the 1.9 GB coronary file (the boundary-only reader takes
   15 s).
2. Rebuild elements from faces. Group faces by cell. From the triangle
   and quadrilateral counts, classify each cell as a tetrahedron (4 tri),
   prism (2 tri + 3 quad), pyramid (4 tri + 1 quad) or hexahedron
   (6 quad), and order its nodes to the reference element. Check positive
   volume.
3. Check quality: minimum and maximum sub-volume, aspect ratio, and
   orthogonality angle. Report them the way CFX's mesh diagnostics do, so
   the collaborator can compare.
4. Renumber nodes by spatial chunk, then reverse Cuthill–McKee within each
   chunk. That keeps partitions aligned to store chunks, as
   `BrickDomain.partition` does, and keeps the matrix bandwidth low.

Mesh generators for tests are written in-house: a box of hexahedra split
into tetrahedra (Kuhn) or prisms, an O-grid tube with prism layers, and
randomly perturbed nodes. A small Fluent writer makes round-trip tests
possible. gmsh (GPL) may be run only as a separate program, as the
[Clean-room policy](../how_to/cleanroom.md) allows for TetGen.

### Control-volume geometry

Each element is split into one sub-volume per node, bounded by surfaces
through edge midpoints, face centroids and the element centroid. Each
integration point carries an area vector and the local coordinates where
shape functions are evaluated. Two checks run on every mesh:

- the node volumes add up to the mesh volume, to round-off;
- each interior control volume is closed: its outward area vectors sum to
  zero, to round-off (the geometric conservation law).

**Recompute, don't store.** Storing area vectors and shape-function
derivatives at every integration point would take about 17 GB in double precision for the
coronary mesh (113 M integration points). The assembly kernels recompute
them from node coordinates instead. On a GPU that arithmetic is cheaper
than the memory traffic. Only connectivity (≈ 0.3 GB), node coordinates
and the lagged integration-point mass flows (≈ 0.9 GB) persist.

### Discretisation

Each choice follows the published description of CFX's method. The
sources are papers and vendor documentation, as the clean-room policy
requires.

| Term | Choice | Source |
|---|---|---|
| Control volumes, ip fluxes | element-based finite volume, vertex-centred, linear shape functions | Schneider & Raw (1987); CFX Theory Guide, discretisation chapter |
| Pressure–velocity coupling | co-located. The ip velocity carries a Rhie–Chow-type pressure-redistribution term with a coefficient from the momentum equation, so the pressure block is not singular and no checkerboard appears | Rhie & Chow (1983); CFX Theory Guide |
| Advection | upwind, specified blend β, or High Resolution: β chosen per node by a Barth–Jespersen-type limiter. It is applied as deferred correction: upwind implicit, the high-order part explicit | Barth & Jespersen (1989); CFX Modeling Guide §16.5 |
| Diffusion, pressure gradient | shape-function gradients inside each element, so no non-orthogonal correction is needed | Schneider & Raw (1987) |
| Non-linearity | Picard: lagged ip mass flows, re-linearised every outer iteration or coefficient loop | CFX Theory Guide |
| Viscosity | μ(γ̇) at each ip from the shape-function strain rate, lagged one iteration, with CFX's minimum and maximum shear-rate clips | CFX Modeling Guide §1.2.13 |
| Steady | false time step. The timescale is auto, or physical at ¼–⅓ of L/U | CFX Modeling Guide §16.4.1 |
| Transient | second-order backward Euler with variable step; first step first-order; 3–5 coefficient loops per step | CFX Modeling Guide §16.4.2 |
| Residuals | normalised RMS and MAX residuals per equation and global imbalances in %, defined as the Theory Guide defines them, so a zvCFD log reads like a CFX `.out` file | CFX Modeling Guide §16.10 |

The exact form of the Rhie–Chow term, the High Resolution limiter and the
residual normalisation have to come from the CFX Theory Guide, and it is
not in hand yet (see [Inputs needed](#inputs-needed-from-others)).
Differences in these details are the most likely cause of residual
disagreement with CFX at the 0.1–1 pp level.

### Assembly on the GPU

One kernel per element type builds the element's block contributions:
tetrahedron 4 × 4 node blocks, prism 6 × 6, each block 4 × 4. It adds
them into a block-CSR matrix whose sparsity pattern is built once, on the
host, in Phase 1.

zvCFD reruns are byte-identical today, and the finite-volume solver should
keep that. Floating-point atomics would break it, because their order
varies from run to run. Two deterministic schemes are compared in Phase 2:

- **element colouring** (Cecka, Lew & Darve 2011): no two elements of one
  colour share a node, so each colour adds without races. Expect about
  20–30 colours;
- **node gather**: each row walks its incident elements in a fixed order
  and recomputes the terms it needs. This does more arithmetic but has no
  colours.

### Linear solve

One block system per outer iteration, reduced by about 10× rather than
solved tightly, as CFX does (MG §18.3.2). The same interface has two
back ends:

- **AmgX (BSD-3)** through a thin ctypes binding to its C API rather than
  pyamgx. The binding uploads matrices straight from device pointers,
  replaces coefficients without rebuilding the hierarchy
  (`AMGX_matrix_replace_coefficients`, structure reuse), runs mixed
  precision (double vectors, single-precision hierarchy), and later runs
  in distributed mode. Aggregation AMG with block size 4 and
  multicolour-DILU or ILU smoothing, inside FGMRES.
- **zvCFD's own ACM**, the CFX method: pairwise or strongest-coupling
  aggregation, coarse operators by summation, block ILU(0) or multicolour
  Gauss–Seidel smoothing, V or F cycles (Hutchinson & Raithby 1986;
  Raw 1996). It is written in CuPy and has no MPI dependency.

Decision gate B picks one back end. The other stays behind the same
interface as a check.

### Outer loop

```text
initialise (zero, or the LBM result interpolated to the nodes)
repeat (steady: until RMS <= target and imbalances <= 1 %;
        transient: per time step, 1..5 coefficient loops)
    update viscosity at ips             (non-Newtonian)
    update lumped outlets               (RCR, coronary: implicit coefficients)
    assemble block system               (GPU, deterministic)
    solve, residual reduction ~ 0.1     (AMG, block 4 x 4)
    update u, v, w, p; recompute ip mass flows
    log RMS / MAX residuals, imbalances, patch flows
```

### Boundary conditions

Patches keep the existing table and rules. `boundaries.patches` matches
zone names, and `zvcfd.boundary.Patch` carries kind, SI values, profile
and waveform. What is new is where the patches act: on mesh boundary
faces, and on their integration points, instead of on voxels.

| Condition | Treatment |
|---|---|
| Wall | no slip; velocity rows replaced by Dirichlet rows |
| Velocity inlet | Dirichlet velocity from plug or parabolic profile, and waveform |
| Mass-flow inlet | velocity scaled each loop to hit the target flow exactly. The flow control that the LBM needs becomes a direct constraint |
| Static pressure outlet | pressure specified at the ip; velocity from the interior |
| Average static pressure | patch-average pressure specified, with a local profile allowed |
| Opening | pressure specified; inflowing faces take a direction and total-pressure treatment (MG §2.5–2.6) |
| RCR and coronary | outlet pressure from `zvcfd.lumped`. It is linearised in the patch flow and entered implicitly, which is stable with 77 outlets where lagged explicit coupling is not |
| Backflow stabilisation | the inflow-only stabilisation term of Esmaily Moghadam et al. (2011) on outlet ips |

Wall shear stress comes from shape-function gradients at wall
integration points. It is written as an attribute of the wall surface
store. With linear elements the wall gradient is first order in mesh
size, but it is smooth and taken on the true wall. The staircase LBM wall
gives neither.

### Storage and output

The run collection is the one from [Run collection](../spec/run_collection.md).
It gets two new node kinds, and snapshots stay `zvcfd:fields`:

- **`mesh.zarrvectors`**, custom geometry type `zvcfd:fv-mesh`, following
  the precedent of `zvcfd:bricks`. Mesh nodes are ZV vertices, chunked
  spatially. Node attributes are control-volume volume, boundary flag and
  global id. Elements are `links` of width 4 (tetrahedron) or 6 (prism);
  whether that is one links array per element type or one store per type
  is settled in the spec page. zarr-vectors already writes links of any
  width ≥ 3 and records links that cross chunks, as its `mesh` type does
  for faces. So this needs a zvCFD convention, written up as a spec page,
  but no addition to zarr-vectors.
- **`wall.zarrvectors`**, a standard ZV `mesh` store of the boundary
  triangles, with zone as a face attribute and WSS and TAWSS as vertex
  attributes.
- **Snapshots**: node-field stores (`u`, `v`, `w`, `p`, `mu`) in node
  order, written chunk by chunk with the existing three-phase contract.
  Each GPU writes the chunks it owns.

For comparisons, a `sample(points)` function evaluates either solver's
fields at given points: trilinear on bricks, shape functions on
elements. The SimVascular comparison already samples probes and computes
TAWSS this way (`benchmarks/simvascular/compare.py`), and the same code
serves the CFX and OpenFOAM comparisons.

### Configuration

```yaml
name: coronary-fv
source: {kind: mesh, path: "mesh 1.msh", unit: mm}     # no voxel_size for fv
physics:
  rho: 1060.0
  rheology: {model: carreau-yasuda, mu_0: 0.056, mu_inf: 0.00345, lam: 3.313, a: 2, n: 0.3568}
boundaries:
  patches:
    - {match: inlet, kind: velocity, flow_rate: 1.9e-7, profile: parabolic}
    - {match: outlet, kind: pressure, pressure: 0.0}
solver: {method: fv}
fv:
  regime: steady                 # steady | transient
  advection: high-resolution     # upwind | high-resolution | {blend: 0.75}
  timescale: auto                # steady: auto | seconds
  dt: null                       # transient: seconds
  coefficient_loops: [1, 5]      # transient: min, max per step
  residual: {kind: rms, target: 1.0e-5}
  imbalance_target: 0.01
  max_iterations: 300
  linear: {backend: amgx, reduction: 0.1, precision: mixed}
output: {path: runs, every: 0, fields: [u, v, w, p, mu, wss]}
```

Config stays strict (unknown keys are errors). `source.voxel_size` is
required only when `solver.method` is `lbm`.

### Reproducibility

Reruns with the same mesh, config, GPU model and partition count are
byte-identical: fixed node order, deterministic assembly, deterministic
reductions, and AmgX's determinism flag. Runs with different partition
counts agree to the solver tolerance, not bit for bit, because multigrid
aggregation depends on the partition. The docs will say so, as
[Validation](../validation/index.md) does for the LBM.

---

## Phases

Effort is in working weeks for one developer who knows the codebase, as
in the [Roadmap](roadmap.md). Each phase ships its tests, a validation page
and its spec page. Documentation is not left to the end.

### Phase 0 — inputs and decisions (1 week)

- [ ] Obtain the CFX Theory Guide's discretisation and solver chapters.
- [ ] Get the collaborator's CFX setup. The `.out` file is best: it lists
  every setting and the residual history. Get their outlet flows and
  pressures as CSV.
- [ ] Add the licence file (BSD-3, as proposed). AmgX is BSD-3 and needs
  its attribution.
- [ ] Build AmgX in the BRIDGE H100 image and on the workstation, and
  record the version.
- [ ] Run roadmap milestone 0 (the H100 job script), so that H100 access
  is proven before any phase depends on it.

**Exit:** each input is in hand or has a named substitute.

### Phase 1 — mesh, control volumes, stores (3–5 weeks)

- [x] Full Fluent reader (ASCII and binary) with element reconstruction
- [x] `.vtu` reader for SimVascular meshes (`zvcfd.mesh.vtk`; VMR 0066: 3.83 M tetrahedra, 27 zones, in 8 s)
- [x] Mesh generators (Kuhn tetrahedra, prisms, O-grid tube with layers,
  perturbed) and a Fluent writer for round trips
- [x] Shape functions and median-dual geometry for all four element types
- [x] The block sparsity pattern
- [x] Node renumbering (chunk order, then reverse Cuthill–McKee)
- [x] `zvcfd:fv-mesh` store, wall store, spec page `spec/mesh_store.md`
- [x] `zvcfd mesh-info` reports element counts, couplings per node and
  predicted GPU memory

**Tests:** volume sums and closed control volumes on every generator mesh
and on the coronary mesh; reader round trip; store round trip.

**Exit (gate A):** the coronary mesh reads in under 2 minutes, passes
both geometry checks, and its measured couplings per node replace the
memory *estimate* below.

### Phase 2 — steady coupled solver, one GPU (7–10 weeks)

- [x] CPU reference assembler and solver (scipy direct solve) for small
  meshes
- [x] GPU residual, gradient and diagonal kernels with element colouring, verified against the CPU reference
- [x] GPU block-matrix assembly with element colouring (first-fit, near
  the greedy colour count; node gather not needed so far)
- [x] Rhie–Chow coupling; upwind, then High Resolution by deferred
  correction
- [x] AmgX binding and own ACM behind one interface (gate B: AmgX)
- [x] False-timestep outer loop, CFX-style residuals and imbalances
- [x] Walls, velocity inlets, static pressure outlets (reference solver)
- [x] Symmetry (for
  quasi-2-D cases)
- [x] `zvcfd run` with `solver.method: fv`

**Tests:** GPU kernels match the CPU reference to 1 × 10⁻¹² relative;
fast manufactured-solution and Poiseuille cases in the test suite.

**Validation:** V1–V7 in the [validation matrix](#validation).

**Exit (gate B, halfway):** AmgX or own ACM chosen from measurements on a
1 M-node tube with prism layers: residual reduction per cycle, time per
solve, and memory. **Exit (phase):** second order on manufactured
solutions and the pipe, and Schäfer–Turek 2D-1 inside the published
intervals.

### Phase 3 — blood and hemodynamic outlets; steady coronary case (3–5 weeks)

- [x] `zvcfd.rheology`: Carreau–Yasuda, Casson, Cross, power law, clips
- [x] Mass-flow inlets, average static pressure, openings
- [x] `zvcfd.lumped`: RCR and open-loop coronary (Kim et al. 2010), with
  implicit coupling. The LBM `PatchController` uses it too
- [x] Backflow stabilisation
- [x] Wall shear stress on the wall store
- [ ] The steady coronary case on one H100

**Validation:** V8–V9, then X1–X3.

**Exit (gate C):** see [Decision gates](#decision-gates).

### Phase 4 — transient flow (3–5 weeks)

- [x] BDF1 and second-order backward Euler (BDF2) with coefficient loops
  and their own convergence test, constant step (reference solver)
- [x] Variable time step
- [x] Time-dependent inlets and pressures (`f(x, t)`)
- [x] Lumped outlets coupled implicitly in time in the FV solver
- [x] Time-step check: rerun at double and half the step (MG §16.4.2):
  `zvcfd.verification.time_step_study`; the coronary run itself waits for
  the H100
- [x] Transient snapshots and TAWSS

**Validation:** V10–V13, then X2 (transient) and X4.

**Exit:** second order in space and time on Womersley and Ethier–Steinman;
one cardiac cycle of the coronary case on one H100, timed.

### Phase 5 — multi-GPU (6–10 weeks; optional, see gate D)

- [x] Partition nodes by whole store chunks along a Morton curve,
  balanced by nodes. The overlap is reported; it falls under 10 % only
  at a few hundred thousand nodes per partition
- [x] Halo exchange by peer copies (CuPy), exercised as partitions of one
  GPU here
- [x] Distributed assembly; distributed linear solve: one FGMRES over the
  partitions with a block-Jacobi preconditioner (ACM or AmgX V-cycle per
  partition). AmgX's distributed mode needs an MPI build, and a gathered
  coarse level is left for when scaling is measured
- [x] Global reductions for residuals and patch flows
- [ ] Parallel snapshot writes, one writer per GPU (snapshots are gathered
  to one writer today)
- [ ] Scaling on 2–8 H100s

**Exit:** agreement with the one-GPU run to the solver tolerance; weak
scaling ≥ 70 % from 1 to 8 H100s at ≥ 2 M nodes per GPU. The coronary
mesh has 0.7 M nodes per GPU on eight, which is too small to scale well
on its own.

### Phase 6 — performance and write-up (2–3 weeks)

- [x] Profile; tune the kernels; measure how often the hierarchy has to
  be rebuilt; mixed precision. Done in the speed review of 30 September –
  1 October ([FV solver speed](../benchmarks/fv_speed.md)): fused CUDA
  kernels in colour-ordered slots, set-up on the GPU, one fused boundary
  and residual pass, one host read-back per outer iteration, AmgX
  symmetric Gauss–Seidel by default with an `auto` probe of `robust`.
  0.62 → 0.32 s per outer iteration on the 16-cell tube; 0.084, 0.142 and
  0.254 s at 22 k, 72 k and 169 k nodes (gate B); 4.85 s per BDF2 step on
  the 398 k-node SimVascular crop. Approaches and open questions:
  [Speed](#speed-approaches-and-questions-for-phase-6). The rest needs a
  GPU with full FP64 rate
- [x] Benchmarks against OpenFOAM on the same mesh and against the LBM
  ([FV solver speed](../benchmarks/fv_speed.md#against-other-methods-one-pipe)):
  at 72 k nodes the flow rate to 0.1 % 36× sooner than OpenFOAM on 16
  cores, and 0.5–0.7 % from the exact rate against OpenFOAM's 5.8–6.8 %
  after the fix to momentum entering through pressure boundaries
  ([developed pipe](../validation/fv_solver.md#developed-pipe-flow-through-pressure-boundaries))
- [ ] Tutorial: solving on the Ansys mesh; API pages; update the
  [Roadmap](roadmap.md) and [Architecture](architecture.md)

### Totals

| Scope | Phases | Weeks | Months (≈ 4.3 weeks) |
|---|---|---|---|
| Steady, one GPU, validated, coronary case | 0–3 | 14–21 | 3.3–5 |
| + transient with outlet models | 4 | +3–5 | 4–6 |
| + write-up | 6 | +2–3 | 4.5–7 |
| + multi-GPU | 5 | +6–10 | 6–9 |

---

## Validation

The same rules apply as for the LBM solver. Reference solutions live in
`benchmarks/validation/exact.py`, independent of the solver, and the new
ones join them there. Fast versions run in the test suite, and results go
to `benchmarks/results/validation/fv/`. Each case reports the error and
the observed order under grid refinement, on tetrahedral, prism and mixed
meshes.

| # | Case | Reference | Exercises | Expected | Phase |
|---|---|---|---|---|---|
| V1 | Manufactured solution, steady, trigonometric u and p | Roache (2002); Salari & Knupp (2000) | every term, every element type, perturbed nodes | order 2 (u), ≈ 2 (p) | 2 |
| V2 | Plane Poiseuille and Couette | exact | walls, pressure outlets | Couette exact; Poiseuille order 2 | 2 |
| V3 | Hagen–Poiseuille pipe, body-fitted with prism layers | exact | curved walls | order 2 (LBM staircase: 1.2) | 2 |
| V4 | Square duct | White series | 3-D walls | order 2 | 2 |
| V5 | Kovasznay flow, Re = 40 | exact Navier–Stokes (Kovasznay 1948) | advection, High Resolution | order 2 | 2 |
| V6 | Lid-driven cavity, Re = 100, 400, 1000 | Ghia, Ghia & Shin (1982) | recirculation, limiter | centreline profiles within the tabulated scatter | 2 |
| V7 | Cylinder in a channel, Re = 20 (DFG 2D-1) | Schäfer & Turek (1996); FeatFlow values | drag, lift, Δp on a curved body | inside the published intervals | 2 |
| V8 | Carreau–Yasuda channel and pipe | stress-balance quadrature | rheology, clips | order 2 | 3 |
| V9 | Pipe with an RCR outlet, steady | 0-D analytic: P = Q (Rp + Rd) + Pd plus the Poiseuille drop | lumped coupling | to solver tolerance | 3 |
| V10 | Womersley pipe, α = 4 and 12 | exact (Bessel) | time accuracy, curved walls | order 2 in space and time | 4 |
| V11 | Ethier–Steinman flow | exact 3-D unsteady Navier–Stokes (Ethier & Steinman 1994) | all nonlinear terms in 3-D, transient | order 2 | 4 |
| V12 | Cylinder, Re = 100 (DFG 2D-2) | Schäfer & Turek (1996) | vortex shedding | Strouhal, drag and lift maxima inside intervals | 4 |
| V13 | Coronary, dt doubled and halved | self-consistency (MG §16.4.2) | time-step independence | flow splits change < 0.1 pp | 4 |

Cross-code comparisons validate the physics setup, not the code. They
are reported separately, as [Against OpenFOAM](../validation/cross_code.md)
is today:

| # | Comparison | Same | Compared |
|---|---|---|---|
| X1 | OpenFOAM v2506 (`fluent3DMeshToFoam`, official image) | mesh, BCs, Newtonian | outlet splits, inlet pressure, velocity at probes |
| X2 | CFX (collaborator) | mesh, BCs, rheology, residual 10⁻⁵ | splits, pressure drop, WSS; then transient waveforms |
| X3 | zvCFD LBM at 50, 35, 25 µm | geometry (voxelised), BCs | splits, and convergence of the LBM towards the mesh result |
| X4 | SimVascular svMultiPhysics, VMR 0066 | SimVascular's mesh (`.vtu`), BCs | the quantities of the current SimVascular comparison |

### Comparison protocol for X1–X2

What decides agreement is the inputs, so fix them first. That means one
boundary table: the same inlet flow and profile, outlet conditions, blood
model with clips, and reference pressure. The same mesh file, checked by
node count and volume. The same convergence: RMS 10⁻⁵ and imbalances
below 1 % in both codes. Then compare each outlet's split in pp, the
inlet-to-outlet pressure drop, and WSS at matched wall points. Every
difference is explained as input, discretisation or convergence before it
is reported.

---

## Estimates

**Memory, coronary mesh (estimated before gate A; measured since:
38.8 GB double with ILU(0), 23.2 GB mixed, from
106.0 M blocks).** The estimate was: each node
couples to 14–20 others (about 14 in tetrahedral regions and about 20 in
prism layers), so 15–21 blocks per row and 85–119 M blocks:

| Item | Double precision | Mixed (single-precision matrix and hierarchy) |
|---|---|---|
| Fine matrix, 16 values per block | 10.9–15.2 GB | 5.4–7.6 GB |
| ILU(0) factors on the fine level | 10.9–15.2 GB | 5.4–7.6 GB |
| Coarse levels (operator complexity 1.2–1.4) | 2.2–6.1 GB | 1.1–3.0 GB |
| Mass flows, fields, time levels, Krylov vectors, connectivity | 3–6 GB | 3–6 GB (vectors stay double) |
| **Total** | **27–43 GB** | **15–24 GB** |

With DILU smoothing instead of ILU(0), the factor row drops to almost
nothing. Either way the case fits one H100 (80 GB), but not the RTX A2000
(12 GB), even in mixed precision. The earlier figure of 10–20 GB left out
the smoother and the coarse levels. Development on the A2000 is limited to
meshes of about 1–2 M nodes, which covers every validation case.

**Time on one H100 (estimate, from memory bandwidth; Phase 2 measures
it).** One block matrix-vector product reads about 13 GB, which takes
≈ 4 ms at 3.35 TB/s. A V-cycle costs about 5–10 of those. Two to four
cycles reduce the residual 10×, and assembly adds about as much again.
Allowing for kernels below peak bandwidth, that gives **0.1–0.5 s per
outer iteration**:

| Run | Outer iterations | Time |
|---|---|---|
| Steady | 50–150 (MG §16.4.1) | 0.1–1.3 min |
| One 0.8 s cardiac cycle at dt = 1 ms | 800 steps × 3–5 loops | 4–33 min |

For scale: Fluent's GPU solver is quoted at 6–25 min per cycle on one H100
([Comparison](../benchmarks/comparison.md)). The LBM gets all 77 splits
within 0.1 pp in 0.6 min at 50 µm, on an A2000.

---

## Speed: approaches and questions for Phase 6

**Status: written 30 September 2026, before the GPU solver runs the
coronary case.** Everything below is a plan for Phase 6. The measured
numbers come from gate B, the only whole-solver timing so far. The rest
are approaches to try and questions to settle by measurement, in the
order they are likely to pay off.

**Update, 30 September (evening): the speed review.** Approaches 2, 3
and 5 are done and measured on [FV solver speed](../benchmarks/fv_speed.md):

- **Approach 2** (the kernels, and set-up on the GPU).
- **Approach 3**, in part: the smoother. Symmetric Gauss–Seidel takes
  2–4 iterations on hard synthetic systems where DILU took up to 300. On
  the coronary mesh it is 5–11× faster than the old DILU default and
  1.4–1.8× faster than ILU(0). A full 1 s cardiac cycle on the
  398 k-node coronary model ran in 81 min on the A2000 (4.85 s per step).
  Hierarchy reuse through `AMGX_solver_resetup`, with a rebuild signal,
  is built and cuts set-up 3–20× where it works. But AmgX 2.5 fails on
  the first re-set-up of larger systems, so it is opt-in.
- **Approach 5**: residuals, zone flows and statistics stay on the device,
  with one host read-back per outer iteration.

The answers are unchanged: bit-identical, or equal to round-off where
the system is solved in an equivalent form. One finding changes the
priorities. On this workstation the GPU is often shared with another
user's job, and then every host synchronisation costs milliseconds, not
microseconds. At that point AmgX's hierarchy set-up, which is hundreds
of small synchronised steps, costs more than everything else in an
outer iteration together. Fewer synchronisations therefore come before
bandwidth.

**The target** is the bandwidth-bound estimate [above](#estimates): 0.1–0.5 s
per outer iteration for the coronary mesh on one H100, a steady run in
about a minute, and a 1 s cardiac cycle in 15 min or less. For scale,
Fluent's GPU solver is quoted at 6–25 min per cycle
([Comparison](../benchmarks/comparison.md)). Two rules hold throughout:
the validation suite (V1–V13) passes unchanged after every optimisation,
and reruns stay byte-identical ([Reproducibility](#reproducibility)).

### Where the time goes now

Gate B, the prism-layered tube, steady Re = 100, on the RTX A2000
(`benchmarks/results/fv/gate_b.json`). Time per outer iteration, with
each linear system reduced 10×:

| Nodes | Linear solver | Per outer | Assembly | Linear setup | Linear solve | Other | Linear iterations per outer |
|---:|---|---:|---:|---:|---:|---:|---:|
| 21,905 | AmgX | 0.25 s | 0.18 s | 0.02 s | 0.02 s | 0.03 s | 2.3 |
| 72,265 | AmgX | 0.49 s | 0.36 s | 0.03 s | 0.04 s | 0.07 s | 2.3 |
| 169,377 | AmgX | 0.81 s | 0.58 s | 0.05 s | 0.06 s | 0.11 s | 2.0 |
| 169,377 | own ACM | 1.61 s | 0.59 s | 0.57 s | 0.35 s | 0.10 s | 4.7 |
| 328,601 | AmgX | 1.66 s | 1.24 s | 0.09 s | 0.13 s | 0.20 s | 2.3 |
| 328,601 | own ACM | 3.21 s | 1.23 s | 0.91 s | 0.88 s | 0.19 s | 6.1 |
| 565,297 | AmgX, mixed | 3.87 s | 3.05 s | 0.14 s | 0.43 s | 0.25 s | 2.8 |

What this says:

- **Assembly is the bottleneck, not the linear solve.** It takes 72 % of an
  outer iteration with AmgX. It costs 0.8 µs per element, but the blocks
  it writes (2.9 M × 128 B) would take about 1.5 ms at the A2000's
  measured 253 GB/s. So assembly runs about 100× slower than that limit,
  and it is the first place to look. On the A2000, though, bandwidth is
  not the binding limit: its double-precision rate is 1/32 of single
  (about 0.25 TFLOPS), and the kernels are double precision and recompute
  element geometry at every flux point, so they are FP64-compute bound
  there. The bandwidth figures are the H100's yardstick (67 TFLOPS
  FP64), not the A2000's. The passes are not fused: gradient of `u`,
  gradient of `p`, limiter, diagonal, assembly and mass flows are
  separate kernels, each launched once per colour (136 tetrahedral and 72
  wedge colours in these runs; the first-fit colouring since gives 38 and
  23).
- **AmgX solves cheaply.** It needs two iterations per outer iteration at
  0.03 s each, and rebuilds its hierarchy every outer iteration
  (`structure_reuse_levels: 0`) in 0.02–0.14 s. Full reuse (`−1`) froze
  the first matrix's coarse operators and stalled FGMRES, so reuse needs
  a rebuild signal, not a fixed setting. zvCFD's own multigrid
  re-aggregates every 10 outer iterations, but redoes the Galerkin sums,
  the block-diagonal inverses and a dense coarsest-level inverse on the
  host every time (0.57 s).
- **The whole outer iteration is about 20× the bandwidth limit.** One
  block matrix–vector product on this mesh moves about 0.38 GB, 1.5 ms
  on the A2000. Two preconditioned iterations are 15–25 such products,
  and assembly should cost a few more, so the limit is 30–45 ms against
  0.81 s measured (with the FP64 caveat above).
- **The larger meshes first failed for memory on the A2000,** which the
  desktop shares (about 2.4 GB in use). AmgX's error code 5 at 20 cells
  across was out of memory: a second hierarchy (the single-solve test's)
  was still resident, and every outer iteration copied the whole block
  matrix for the reactions. With both fixed, 20 cells across runs with
  either back end, and 24 (565 k nodes) with AmgX in mixed precision.

### Approaches, in the order to try them

**1. Measure before changing anything.**
- Profile one outer iteration with Nsight Systems: kernel launches, host
  synchronisations, host–device copies and idle gaps.
- Measure each kernel with Nsight Compute, as bytes moved against the
  roofline, as [Kernels](../benchmarks/kernels.md) does for the LBM.
  The headline number is the fraction of peak bandwidth per kernel.
- Measure on a GPU that no other job is using.

**2. Assembly.** *Done (30 September): stored geometry, block positions,
local accumulation, fused gradients, no per-launch synchronisation: one
assembly at 72 k nodes 458 → 150 ms (stage 1 180 → 50 ms, assembly kernel
273 → 99 ms, mass flows 71 → 19 ms); on zvcfd-be's 8 k-node all-tet
segment 1.84 → 0.86 s per time step. Still open below: fusing the rest,
CUDA graphs, single-precision blocks.*
- Fuse the per-element passes (shape-function gradients, limiter, ip
  mass flows, momentum and continuity blocks) into one kernel per element
  type and colour, so each element's nodes and geometry are read once.
- Weigh stored against recomputed geometry: ip areas and shape-function
  gradients cost hundreds of bytes per element to read, and recomputing
  them from the node coordinates may be cheaper on an H100.
- Count the colours, and the launches per outer iteration. With 20–30
  colours, two element types and several passes, launch overhead alone
  can dominate on small meshes. Compare larger colour classes, and the
  node-gather scheme ([Assembly on the GPU](#assembly-on-the-gpu)).
- Write the block values straight into the buffer the linear solver
  reads (and in single precision for a mixed-precision hierarchy), with
  no format conversion or copy between assembly and solve.
- Capture the whole outer iteration as a CUDA graph, once the kernel
  sequence is fixed, to remove per-launch host cost.

**3. Linear solve.**
- Keep AmgX's hierarchy across outer iterations, and across time steps,
  replacing only coefficients. Measure how many outer iterations one
  hierarchy lasts before convergence per outer iteration degrades, and
  rebuild on that signal rather than every time.
- Mixed precision (AmgX `dDFI`, already in `zvcfd.fv.linear`): the
  hierarchy in single precision inside double-precision FGMRES. It
  halves hierarchy memory and traffic; measure the time saved and check
  V1–V13.
- Tune the solver configuration on the real mesh: smoother (DILU,
  ILU(0), multicolour Gauss–Seidel), cycle (V or F), aggregation for
  block size 4, and the coarsest-level solve.
- Tune the linear tolerance per outer iteration. It is 0.1, as CFX uses;
  0.2–0.3 with a few more outer iterations may be cheaper in total.

**4. Fewer outer iterations.**
- Steady: pick the false time step (CFX's auto timescale, or ¼–⅓ of L/U)
  that minimises total time, not iterations.
- Transient: start each step from a second-order extrapolation of the
  last two steps, and stop the coefficient loops when the residual target
  is met, not after a fixed 3–5 loops. If two loops suffice for most
  steps of the cardiac cycle, a cycle takes half the time.
- Transient: measure the largest time step that keeps V13 (splits change
  < 0.1 pp when dt is halved). The step may be larger in diastole than in
  systole: CFX supports adaptive steps.

**5. Keep the data on the device.**
- Compute residual norms, imbalances and zone flows on the device, and
  copy them to the host only when they are logged.
- Evaluate waveforms and profiles for the boundary values on the device,
  rather than on the host and uploading every step.

**6. Several GPUs, only if gate D says so.** `zvcfd.fv.partition` and
`zvcfd.fv.multi` exist. Measure strong scaling of one outer iteration
over 2–8 H100s, and overlap the halo exchange with interior assembly
before tuning anything else there.

### Questions to settle

| # | Question | How to measure | Decides |
|---|---|---|---|
| S1 | What fraction of peak bandwidth does each kernel reach (assembly per element type, matrix–vector product, smoother)? | Nsight Compute, bytes against roofline | where to spend effort; target 50 % or more for streaming kernels |
| S2 | How much of an outer iteration is launch and host overhead? | Nsight Systems; the same iteration as a CUDA graph | whether fusion and graphs come first |
| S3 | Why does assembly cost 0.8 µs per element, and how does that split between tetrahedra and prisms and between colours? | per-kernel timing, per colour and element type | the assembly redesign |
| S4 | How long does an AmgX hierarchy last before it must be rebuilt, in steady and transient runs? | convergence per outer iteration against outer iterations since setup | the rebuild policy |
| S5 | What do mixed precision and DILU save, in time and memory, and does any validation case move beyond its tolerance? | V1–V13 and gate B in both modes | the default precision. So far: the same outer and linear iterations as double on gate B; AmgX's hierarchy memory halved (it fits 565 k nodes); AmgX's dense LU coarse solver fails in dDFI mode, so mixed mode smooths the coarsest level instead |
| S6 | Which linear tolerance per outer iteration minimises total time to a converged steady run? | sweep 0.05–0.3 on the tube and the coronary mesh | the default tolerance |
| S7 | How many coefficient loops does a time step need with an extrapolated start, over a whole cardiac cycle? | loops per step against the residual target | the transient default; cycle time |
| S8 | Why did AmgX fail (code 5) at 20 cells across? | the failing matrix, solver settings, conditioning | answered: out of device memory (see above) |
| S9 | Does the time per outer iteration scale with bandwidth from the A2000 to the H100 (about 13×)? | gate B and the coronary mesh on both | whether A2000 profiling predicts the H100 |
| S10 | How does zvCFD compare with CFX and Fluent GPU on the same mesh? | the collaborator's CFX wall-clock per iteration, iterations and cores; a Fluent GPU run if a licence allows | the headline comparison |
| S11 | Do several GPUs pay, and how does one outer iteration scale over 2–8 H100s? | strong scaling on the coronary mesh | gate D |

S10 needs inputs from others: the collaborator's CFX run times on the
coronary mesh (the `.out` file already requested
[below](#inputs-needed-from-others) records them), and access to Fluent's
GPU solver if a comparison with it is wanted.

---

## Decision gates

| Gate | When | Question | If no |
|---|---|---|---|
| A | end of Phase 1 | Does the coronary mesh fit one H100 in double precision, measured? | mixed precision becomes the default |
| B | middle of Phase 2 | AmgX or own ACM: which reduces the residual 10× fastest and most reliably on the prism-layered tube? | keep both; choose again per case |
| C | end of Phase 3 | Steady coronary case: each outlet split within **0.2 pp of OpenFOAM** and **0.5 pp of CFX**, pressure drop within 2 %? (targets, not predictions) | trace the difference to inputs, then to the Rhie–Chow and limiter details, before any new phase |
| D | after Phase 4 | Is a mesh above one H100's limit (≈ 10–17 M nodes in double precision, ≈ 19–30 M mixed), or a faster turnaround, actually needed? | stop before Phase 5; one H100 holds the coronary case |

---

## Risks

1. **CFX's exact variants are not in the papers.** The Rhie–Chow form,
   the limiter and the residual scaling can move splits by tenths of a pp.
   *Retire:* the Theory Guide (Phase 0), and the collaborator's `.out`
   file to compare residual histories.
2. **AMG on the coupled saddle-point system.** Block AMG is less robust
   than scalar AMG when the pressure diagonal is small. *Retire:* FGMRES
   around the AMG; test on the prism-layered tube at gate B, before any
   large case. **Happened** (30 September 2026), not on gate B's tube but
   on the DFG slab and on strongly graded layers: point-block smoothers
   (DILU, block Jacobi) diverge there. *Now:* ILU(0) smoothing with
   pressure-weighted pairwise aggregation as the AmgX default, a SIMPLE
   block preconditioner, and the `auto` ladder that switches between them
   ([numerics](../spec/fv_numerics.md#linear-solvers-on-thin-cells-zvcfdfvlinear)).
   An extreme slab (aspect 10 in every cell) is still solved only by
   ILU(1), which AmgX cannot run on tetrahedral rows.
3. **Prism layers of high aspect ratio.** Thin sub-volumes can make the
   system stiff and anisotropic, which is why CFX coarsens
   anisotropically. *Retire:* aggregate along the strongest couplings;
   run the quality report of Phase 1 on the coronary mesh early.
   *Status:* see 2. The coronary mesh's layer aspect ratios, and a first
   linear solve on it, are the next check; the run history records every
   solve's iterations and convergence.
4. **No H100 run yet.** The coronary case cannot run on the A2000.
   *Retire:* roadmap milestone 0 in Phase 0.
5. **The AmgX build.** It needs CUDA and a compiler that match the
   container. *Retire:* build it in Phase 0; own ACM is the fallback.
6. **fp64 speed on the workstation.** The A2000 runs double precision at
   1/32 of single-precision speed. Phase 2 development is slower there,
   but correct. *Retire:* develop the kernels in double on small meshes,
   and time them on the H100.
7. **It competes with shorter work.** Coronary outlets and sub-voxel walls
   for the LBM take a few weeks and serve both the CFX and SimVascular
   comparisons. *Mitigation:* build `zvcfd.lumped` (Phase 3) first, as
   shared work, before Phase 1.

---

## Inputs needed from others

| Input | From | Needed by |
|---|---|---|
| CFX Theory Guide: discretisation, coupled solver, residuals | Ansys documentation (the collaborator or the licence holder) | Phase 2 |
| CFX run setup and `.out` file; outlet flows and pressures as CSV | collaborator | Phase 3 |
| H100 node time (milestone 0 job submitted) | Andrew | Phase 0 |
| Licence choice (BSD-3 proposed) | Andrew | Phase 0 |

---

## First two weeks

1. Put `zvcfd.lumped` (RCR and open-loop coronary, with implicit coupling
   coefficients) under both solvers, so the LBM gains coronary outlets at
   once. Validate against the 0-D analytic steady state.
2. Obtain the Theory Guide and the CFX `.out` file; build AmgX; submit
   milestone 0.
3. Start Phase 1 with the vectorised Fluent reader and element
   reconstruction. Confirm the 6.55 M tetrahedra and 8.24 M prisms, and
   measure couplings per node.

---

## References for this plan

- G. E. Schneider, M. J. Raw, *Control volume finite-element method for
  heat transfer and fluid flow using colocated variables*, Numer. Heat
  Transfer 11, 363 (1987).
- C. M. Rhie, W. L. Chow, *Numerical study of the turbulent flow past an
  airfoil with trailing edge separation*, AIAA J. 21, 1525 (1983).
- T. J. Barth, D. C. Jespersen, *The design and application of upwind
  schemes on unstructured meshes*, AIAA paper 89-0366 (1989).
- B. R. Hutchinson, G. D. Raithby, *A multigrid method based on the
  additive correction strategy*, Numer. Heat Transfer 9, 511 (1986).
- M. J. Raw, *Robustness of coupled algebraic multigrid for the
  Navier–Stokes equations*, AIAA paper 96-0297 (1996).
- C. Cecka, A. J. Lew, E. Darve, *Assembly of finite element methods on
  graphics processors*, Int. J. Numer. Meth. Eng. 85, 640 (2011).
- P. J. Roache, *Code verification by the method of manufactured
  solutions*, J. Fluids Eng. 124, 4 (2002); K. Salari, P. Knupp, *Code
  verification by the method of manufactured solutions*, Sandia report
  SAND2000-1444 (2000).
- L. I. G. Kovasznay, *Laminar flow behind a two-dimensional grid*, Proc.
  Camb. Phil. Soc. 44, 58 (1948).
- U. Ghia, K. N. Ghia, C. T. Shin, *High-Re solutions for incompressible
  flow using the Navier–Stokes equations and a multigrid method*, J.
  Comput. Phys. 48, 387 (1982).
- C. R. Ethier, D. A. Steinman, *Exact fully 3D Navier–Stokes solutions
  for benchmarking*, Int. J. Numer. Meth. Fluids 19, 369 (1994).
- H. J. Kim et al., *Patient-specific modeling of blood flow and pressure
  in human coronary arteries*, Ann. Biomed. Eng. 38, 3195 (2010).
- M. Esmaily Moghadam et al., *A comparison of outlet boundary treatments
  for prevention of backflow divergence with relevance to blood flow
  simulations*, Comput. Mech. 48, 277 (2011).
- NVIDIA AmgX (BSD-3), <https://github.com/NVIDIA/AMGX>.
- Ansys CFX-Solver Modeling Guide, Release 2025 R1 (MG), and CFX Theory
  Guide; the Schäfer–Turek, Womersley, White and Cho–Kensey references
  are in [References](../references.md).
