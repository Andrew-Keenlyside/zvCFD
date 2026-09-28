# Ansys context

How Ansys Fluent and CFX cut compute time on large models, and what each
technique becomes in a voxel-native GPU solver. The CFX points follow the
Ansys CFX documentation: the coupled algebraic multigrid solver (CFX
theory guide §23.7.3, *Algebraic Multigrid*) and *Using the Solver in
Parallel* (CFX-Solver Manager, chapter 17).

---

## Chunked: partitioning and parallel solving

**What Ansys does.** Fluent and CFX split the mesh into partitions, one
per process. CFX partitions with MeTiS by default, or by recursive
coordinate bisection. Fluent offers METIS, Cartesian-axis bisection,
principal-axis bisection and strips, with a pre-test that picks the method
with the fewest interface faces. Fluent can weight partitions by model cost
and re-balance dynamically during a run. Partitions exchange interface
values every iteration.

**Two limits Ansys users meet** (noted from CFX practice):

- *Memory.* The solver must hold the whole assembled system matrix in RAM.
  When it does not fit, the operating system pages to disk, the run goes
  out of core, and it slows to the speed of the disk.
- *I/O is not parallelised.* Results and backup files are gathered and
  written through one process, so writing gets slower relative to solving
  as core counts rise.

**In zvCFD.**

| Ansys | zvCFD |
|---|---|
| MeTiS / RCB partitions of a mesh | Morton-curve partition of whole store chunks, balanced by fluid cells (`BrickDomain.partition`) — an RCB analogue aligned to the storage unit |
| Interface exchange each iteration | One-brick ghost layer; only crossing populations (5 of 19 per face) over NVLink, overlapped with interior work |
| Assembled system matrix in RAM | **No matrix.** LBM is explicit; memory is the state only (157 B per stored cell fp32, 81 B fp16), known exactly before the run (`zvcfd plan`) |
| Out-of-core cliff when RAM runs out | No silent paging: a domain that does not fit is refused at plan time, and the routes beyond one node are explicit (fp16, sub-volumes, more nodes) |
| Serial result writing | **Parallel by construction**: each GPU writes the chunks it owns, no lock; one presence rebuild per snapshot. A single writer was 1.4–4× slower in our benchmark (0.28 vs 0.38–0.57 GB/s ZV; 0.03 vs 0.13 GB/s dense Zarr) |
| Dynamic load balancing | Re-partition at a checkpoint (designed; ownership is computed, so a restart may use a new partition) |

---

## Multiresolution: coarsening and scale

**What Ansys does.**

Algebraic multigrid (AMG)
: The linear systems inside each outer iteration are solved with AMG.
  Fine cells are agglomerated with their most strongly connected
  neighbours (Fluent's default coarsens by 2 for scalar equations; for the
  coupled system by 8 in 3-D, with ILU smoothing). CFX solves pressure and
  velocity as one coupled system with additive-correction AMG, and scales
  close to linearly with mesh size. The key property: coarse levels are
  built **from the matrix**, so they are consistent with the fine problem
  by construction, whatever the geometry.

FMG initialisation
: Fluent's full-multigrid initialisation restricts the initial field to
  coarse agglomerated levels, runs FAS cycles on a first-order Euler
  problem there, and interpolates up level by level. It is cheap, and it
  gives a better start than a uniform field.

Hybrid initialisation
: A Laplace (potential-flow) solve for velocity and pressure, ~10
  iterations, as the starting field.

Coarse mesh, then interpolate; adaption
: Converge on a coarse mesh, `file/interpolate` onto the fine one. Or
  refine where needed with polyhedral (PUMA) or hanging-node adaption.

Ansys Discovery
: A GPU solver on Cartesian voxels, with no mesh and a **fidelity slider**
  that trades resolution for speed. Architecturally it is the closest
  commercial relative of zvCFD. It publishes no throughput figures.

**What we measured** ([Multiresolution benchmarks](../benchmarks/multiresolution.md)):

| Ansys technique | zvCFD analogue | Result |
|---|---|---|
| Coarse mesh → interpolate | Solve a coarser OME-Zarr level, prolong to the fine one (diffusive scaling) | **0.8–1.6×**, and slower to 0.1 % on a long domain: geometry-coarsened levels mispredict flux by 25–80 % in thin vessels (conductance ∝ r⁴), and the fine solve must unlearn the bias |
| Hybrid initialisation | Fine-level lubrication (local-Poiseuille) pressure solve as the start | **0.6–1.1×**: flux within 1–13 % at the start, but time-marching LBM spends its time damping the initial transient, not the pressure field |
| Discovery fidelity slider | Solve on a coarse level only, as a preview | **7–35× cheaper** per solve (level 1: ~8×; level 2: ~35×), with 25–80 % flux error in 5–9-voxel vessels; better in wide ones |
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

## Ansys GPU solvers, for scale

Fluent's native GPU solver (2022 R1 onwards) keeps the CPU solver's
discretisation. It needs 1.0–5.6 GB of GPU memory per million cells
(single-precision segregated tetrahedra up to double-precision coupled
polyhedra). Ansys and NVIDIA report 1 A100 ≈ 270–400 CPU cores, and
8 × H200 34× faster than 512 cores on a 250 M-cell DrivAer. These are
vendor figures; the independent OpenFOAM GPU port SPUMA measures 1 A100 ≈
200–300 cores. At ~1.8 GB per million cells, one 8 × H100 node holds a
Fluent GPU model of roughly 3 × 10⁸ cells. zvCFD's comparative advantage
starts above that, and wherever building the mesh is the bottleneck.
