# Plan: a CFX-style finite-volume solver

**Status: in progress (29 September 2026).** Phase 1 is built and gate A
is passed on the real mesh; the Phase 2 CPU reference solver is built and
validated; the GPU solver is not started. See [Progress](#progress). This page plans a
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

**Gate A: passed.** Measured on the coronary mesh (`benchmarks/fv/mesh_report.py`):
18.7 blocks per row (median 18, p99 27, max 46),
106.0 M blocks, 227 M flux points (two per ip
face). By the itemisation below: **38.8 GB** in double precision with ILU(0),
24.8 GB with DILU, 23.2 GB mixed, 16.0 GB mixed with DILU.
One H100 holds it in double precision.

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
- [ ] GPU block-matrix assembly; element colouring against node gather
- [x] Rhie–Chow coupling; upwind, then High Resolution by deferred
  correction
- [ ] AmgX binding and own ACM behind one interface
- [ ] False-timestep outer loop, CFX-style residuals and imbalances
- [x] Walls, velocity inlets, static pressure outlets (reference solver)
- [x] Symmetry (for
  quasi-2-D cases)
- [ ] `zvcfd run` with `solver.method: fv`

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
- [ ] Mass-flow inlets, average static pressure, openings
- [x] `zvcfd.lumped`: RCR and open-loop coronary (Kim et al. 2010), with
  implicit coupling. The LBM `PatchController` uses it too
- [ ] Backflow stabilisation
- [ ] Wall shear stress on the wall store
- [ ] The steady coronary case on one H100

**Validation:** V8–V9, then X1–X3.

**Exit (gate C):** see [Decision gates](#decision-gates).

### Phase 4 — transient flow (3–5 weeks)

- [x] BDF1 and second-order backward Euler (BDF2) with coefficient loops
  and their own convergence test, constant step (reference solver)
- [ ] Variable time step
- [x] Time-dependent inlets and pressures (`f(x, t)`)
- [ ] Lumped outlets coupled implicitly in time in the FV solver
- [ ] Time-step check: rerun at double and half the step (MG §16.4.2)
- [ ] Transient snapshots and TAWSS

**Validation:** V10–V13, then X2 (transient) and X4.

**Exit:** second order in space and time on Womersley and Ethier–Steinman;
one cardiac cycle of the coronary case on one H100, timed.

### Phase 5 — multi-GPU (6–10 weeks; optional, see gate D)

- [ ] Partition nodes by whole store chunks along a Morton curve,
  balanced by nodes. Overlap nodes kept under 10 %, CFX's own guideline
- [ ] Halo exchange by peer copies or NCCL, the pattern `MultiLBM` uses
- [ ] Distributed assembly; distributed AMG (AmgX distributed mode, or own
  ACM with per-partition aggregation and a gathered coarse level)
- [ ] Global reductions for residuals and patch flows
- [ ] Parallel snapshot writes, one writer per GPU

**Exit:** agreement with the one-GPU run to the solver tolerance; weak
scaling ≥ 70 % from 1 to 8 H100s at ≥ 2 M nodes per GPU. The coronary
mesh has 0.7 M nodes per GPU on eight, which is too small to scale well
on its own.

### Phase 6 — performance and write-up (2–3 weeks)

- [ ] Profile; tune the kernels; measure how often the hierarchy has to
  be rebuilt; mixed precision
- [ ] Benchmarks against OpenFOAM on the same mesh and against the LBM
  (`docs/benchmarks/`)
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
   large case.
3. **Prism layers of high aspect ratio.** Thin sub-volumes can make the
   system stiff and anisotropic, which is why CFX coarsens
   anisotropically. *Retire:* aggregate along the strongest couplings;
   run the quality report of Phase 1 on the coronary mesh early.
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
