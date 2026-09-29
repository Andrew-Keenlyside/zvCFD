# Ansys context

How an Ansys CFX hemodynamics study is set up and solved, how CFX and
Fluent cut compute time on large models, and what each step and technique
becomes in a voxel-native GPU solver. CFX specifics follow the *CFX-Solver
Modeling Guide*, Release 2025 R1 (cited as MG with section and page). They
also follow notes from a collaborator who runs the HiP-CT coronary case in
CFX. Fluent specifics follow the Fluent theory and user guides.

---

## The pipeline, side by side

The CFX workflow for the coronary case is geometry → volume mesh with
boundary layers → boundary conditions and solver setup → solve. zvCFD
removes the meshing stage:

| Stage | CFX (collaborator's setup) | zvCFD |
|---|---|---|
| Geometry | segmented lumen surface, smoothed (here: Simpleware ScanIP, B-spline smoothing, MeshMixer) | the segmentation itself, as an OME-Zarr label or threshold at a chosen pyramid level |
| Volume mesh | unstructured tetrahedra with prismatic boundary layers (14.8 M cells for this tree) | **none**: active 8³ bricks of the voxel grid, built chunk by chunk in seconds to minutes |
| Near-wall resolution | inflation layers | voxel size, plus interpolated bounce-back from sub-voxel wall distance (roadmap) |
| Equations | 3-D incompressible, isothermal Navier–Stokes | lattice-Boltzmann, which recovers the same equations weakly compressibly; compressibility error ~Ma², kept ≲ 1 % by holding lattice velocity ≤ 0.1 |
| Discretisation | element-based finite volumes; High Resolution advection; second-order backward Euler in time | D3Q19 lattice, second order in space and time in the bulk; walls first to second order depending on the boundary scheme |
| Pressure–velocity coupling | coupled, implicit (u, v, w, p solved together with AMG) | none needed: pressure is a moment of the populations, and the scheme is explicit |
| Fluid | density; rheology model (Newtonian or non-Newtonian) | density, kinematic viscosity; non-Newtonian via a local relaxation time (roadmap) |
| Boundary conditions | velocity / mass-flow inlet, pressure outlets (openings for many outlets) | flag + patch table from Fluent zones or labels; reservoirs today, velocity, pressure and RCR outlets on the roadmap |
| Solve | CPU, MPI partitions; 3–5 coefficient loops per time step | GPU, one worker per device; one explicit update per step |
| Output | results and backup files, written by the leader process | per-GPU parallel brick stores, published atomically in an RFC-8 run collection |

---

## Solver settings a comparison has to respect

From the Modeling Guide, so that a CFX run and a zvCFD run of the same
case are compared fairly:

**Advection.** High Resolution varies the blend factor β locally to keep
the solution bounded: near 1 (second order) in smooth regions, dropping
towards 0 (upwind) at steep gradients. Final results should use "a
Specified Blend of at least 0.75 or the High Resolution scheme" (MG §16.5,
§16.10.5.1, pp. 579–590). Upwind is for robustness only.

**Time.** Second Order Backward Euler is the default and "generally
recommended for most transient runs". It is implicit and not monotonic;
the first step is solved first-order by default. The guide suggests three
coefficient loops per time step for single-phase flow ("values higher than
5 are unlikely to improve accuracy"). It also says to check the time step
by re-running at double and half the step (MG §16.4.2, pp. 574–578). The
guide gives **no pulsatile or blood-flow guidance**.

**Convergence.** RMS residual target 10⁻⁴ by default ("relatively
loose"), 10⁻⁵ is "good", 10⁻⁶ "very tight". Global imbalances should be
below 1 %. Derived quantities (flow splits, WSS) should be checked for
sensitivity to the target (MG §16.10, pp. 585–589). A zvCFD comparison
should converge CFX to 10⁻⁵, and run LBM to a matching change in outlet
flows per cycle.

**Solver.** Fully implicit and coupled. The u, v, w, p system is solved by
algebraic multigrid with anisotropic coarsening, and each linear solve
reduces the residual about 10× (MG §18.3.2, pp. 621–623). CFX's AMG builds
its coarse levels from the matrix: see [what we measured](#multiresolution-coarsening-and-scale).

---

## Chunked: partitioning and parallel solving

**What CFX does** (MG chapter 17, pp. 597–611). CFX runs as one leader and
several follower processes. Partitioning is node-based. MeTiS (graph
partitioning) is the default. The alternatives are recursive coordinate
bisection (low memory, larger overlaps), optimized RCB (MeTiS-like quality
at low memory, for very large cases), user-specified and directional,
radial and circumferential, and junction-box partitioning. The guide asks
for at least ~30,000 nodes per partition on tetrahedral meshes. Overlap
nodes should stay under 10 % of the total, and over 20 % is not
recommended. Performance "may be limited by memory bandwidth".

**Two limits CFX users meet**, confirmed by the guide:

- *Memory.* Every process holds its share of the assembled system. The
  guide says to "avoid using swap space" (p. 605) and documents no
  out-of-core mode. When memory runs out, the run either stops with
  `INSUFFICIENT MEMORY ALLOCATED` (p. 699) or the operating system pages,
  and the run slows to the speed of the disk.
- *I/O is not parallelised.* Setup, control and all input and output go
  through the leader. Reading input and writing results and backup files
  "are highly I/O dependent and not parallelized" (pp. 597, 610). So
  writing gets slower relative to solving as core counts rise.

Fluent partitions with METIS, Cartesian-axis, principal-axis or strip
bisection, with a pre-test that picks the fewest interface faces. It
can weight partitions by model cost and re-balance during a run.

**In zvCFD.**

| Ansys | zvCFD |
|---|---|
| MeTiS / (optimized) RCB partitions of a mesh | Morton-curve partition of whole store chunks, balanced by fluid cells (`BrickDomain.partition`); an RCB analogue aligned to the storage unit |
| Overlap regions exchanged every coefficient loop | One-brick ghost layer; only crossing populations (5 of 19 per face) over NVLink, overlapped with interior work |
| Assembled system in memory | **No matrix.** LBM is explicit; memory is the state only (157 B per stored cell fp32, 81 B fp16), known exactly before the run (`zvcfd plan`) |
| Swap / `INSUFFICIENT MEMORY` | No silent paging: a domain that does not fit is refused at plan time, and the routes beyond one node are explicit (fp16, sub-volumes, more nodes) |
| Leader-only result writing | **Parallel by construction**: each GPU writes the chunks it owns, no lock; one presence rebuild per snapshot. A single writer was 1.4–4× slower in our benchmark (0.28 vs 0.38–0.57 GB/s ZV; 0.03 vs 0.13 GB/s dense Zarr) |
| Dynamic load balancing (Fluent) | Re-partition at a checkpoint (designed; ownership is computed, so a restart may use a new partition) |

---

## Multiresolution: coarsening and scale

**What Ansys does.**

Algebraic multigrid (AMG)
: The linear systems inside each outer iteration are solved with AMG.
  Fine cells are agglomerated with their most strongly connected
  neighbours. Fluent coarsens by 2 for scalar equations and by 8 in 3-D
  for the coupled system, with ILU smoothing; CFX coarsens
  anisotropically. The key property: coarse levels are built **from the
  matrix**, so they stay consistent with the fine problem whatever the
  geometry.

FMG initialisation (Fluent)
: Restricts the initial field to coarse agglomerated levels, runs FAS
  cycles on a first-order Euler problem there, and interpolates up level
  by level.

Hybrid initialisation (Fluent)
: A Laplace (potential-flow) solve for velocity and pressure, ~10
  iterations, as the starting field.

Timescale control for steady runs (CFX)
: Auto timescale, or a physical timescale of ¼–⅓ of L/U; typically
  50–100 outer iterations to a steady solution (MG §16.4.1).

Coarse mesh, then interpolate; adaption
: Converge on a coarse mesh and interpolate onto the fine one, or refine
  locally.

Ansys Discovery
: A GPU solver on Cartesian voxels, with no mesh and a **fidelity slider**
  that trades resolution for speed. Architecturally the closest commercial
  relative of zvCFD. It publishes no throughput figures.

**What we measured** ([Multiresolution benchmarks](../benchmarks/multiresolution.md)):

| Ansys technique | zvCFD analogue | Result |
|---|---|---|
| Coarse mesh → interpolate | Solve a coarser OME-Zarr level, prolong to the fine one (diffusive scaling) | **0.8–1.6×**, and slower to 0.1 % on a long domain: geometry-coarsened levels mispredict flux by 25–80 % in thin vessels (conductance ∝ r⁴), and the fine solve must unlearn the bias |
| Hybrid initialisation | Fine-level lubrication (local-Poiseuille) pressure solve as the start | **0.7–1.1×**: flux within 1–13 % at the start, but time-marching LBM spends its time relaxing the start-up transient, not the pressure field |
| Discovery fidelity slider | Solve on a coarse level only, as a preview | **8–43× cheaper** per solve (level 1: 8–10×; level 2: 33–43×), with 25–80 % flux error in 5–9-voxel vessels; better in wide ones |
| AMG (operator coarsening) | Smoothed-aggregation AMG on the lubrication system | **Iterations flat at 12–14** as the domain grows 4×; Jacobi-CG grows 1,333 → 3,853 |

**The lesson** is the one Ansys already applies. Coarsen the *operator*,
not the geometry. Geometric coarsening of a segmentation changes the
physics, because thin channels close or widen. Algebraic coarsening of the
fine operator cannot. For zvCFD that means:

1. Use pyramid levels to **choose resolution** and to run **previews**
   (the fidelity slider), and label the preview's error as such.
2. Put multigrid where the equations are elliptic: the lubrication /
   network pressure solve now, and a Stokes solver for steady flows later.
   Both use AMG built from the fine operator.
3. Do not spend effort on coarse-to-fine initialisation of time-marching
   LBM. It measured as neutral to harmful.

---

## Physics a CFX user will expect

What the collaborator sets in CFX, and where zvCFD stands:

| Setting | CFX (MG reference) | zvCFD |
|---|---|---|
| Density | constant | constant (`physics.rho`) |
| Newtonian viscosity | constant | `physics.nu` → τ |
| Non-Newtonian | Bird–Carreau, **Carreau–Yasuda** μ = μ∞ + (μ₀ − μ∞)[1 + (λγ̇)ᵃ]^((n−1)/a), Casson, Cross, Herschel–Bulkley, power law, Bingham; min/max shear-rate clips (MG §1.2.13, pp. 74–77) | roadmap: local τ from the strain rate, which LBM gets from the non-equilibrium populations with no finite differences; Carreau–Yasuda and Casson first |
| Inlet | velocity (normal speed, components, profile via CEL) or mass flow rate | roadmap: velocity profile or flow-rate waveform on a patch |
| Outlets | static pressure; average static pressure; mass flow; **openings** recommended with more than two outlets; artificial walls against backflow (MG §2.3.2, §2.5–2.6) | reservoirs today; pressure patches and **RCR (Windkessel) outlets** on the roadmap. CFX has no native Windkessel outlet, only CEL expressions; zvCFD would make it first-class for the 77 outlets |
| Transient inputs | CEL expressions in `t`, profile files (MG §2.3.4, §2.9) | waveform tables in the patch table (designed) |
| Walls | no slip; free, finite or specified shear | no slip (bounce-back); moving walls possible |
| Wall shear stress | monitor points; post-processing | from the non-equilibrium stress at wall voxels; needs interpolated bounce-back for accuracy (roadmap) |

---

## Ansys GPU solvers, for scale

CFX has no GPU solver. Fluent's native GPU solver (2022 R1 onwards) keeps
the CPU solver's discretisation. It needs 1.0–5.6 GB of GPU memory per
million cells, from single-precision segregated tetrahedra up to
double-precision coupled polyhedra. Ansys and NVIDIA report 1 A100 ≈
270–400 CPU cores, and 8 × H200 34× faster than 512 cores on a 250 M-cell
DrivAer. These are vendor figures; the independent OpenFOAM GPU port SPUMA
measures 1 A100 ≈ 200–300 cores. At ~1.8 GB per million cells, one
8 × H100 node holds a Fluent GPU model of roughly 3 × 10⁸ cells.
zvCFD's comparative advantage starts above that, and wherever building the
mesh is the bottleneck.
