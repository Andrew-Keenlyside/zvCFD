# Multiresolution

**Question.** Ansys speeds up large solves with coarsening and scale:
coarse-mesh solutions interpolated up, FMG and hybrid initialisation,
algebraic multigrid. Which of these help a voxel lattice-Boltzmann solver
working from an OME-Zarr pyramid?

**Test case.** Pressure-driven flow through a synthetic vessel network:
26 capsule vessels of radius 5–9 voxels, about half along x, only the part
connected to both x faces kept. Fixed-density reservoirs sit on the two x
faces. The box is 128 × 128 × 512 voxels (14 % fluid), and 128 × 128 × 1024
for the long variant. This is the slow case for LBM: a steady pressure
field has to form along long, narrow channels. τ = 1, BGK, dense kernel.

**Metric.** Flux through the mid-plane, against a fine reference run to
steady state from rest (relative change < 2 × 10⁻⁶ over 1,000 steps). Cost
is counted in fine-step equivalents: a level-*k* step costs 8⁻ᵏ of a fine
step, and the lubrication solve is converted by wall time. Scripts:
`benchmarks/multires_init.py`, `benchmarks/amg_check.py`.

---

## 1. Coarse levels as previews (the fidelity slider)

The pyramid's coarser levels (2× and 4× per axis, `majority` rule), solved
on their own with diffusive scaling:

| Domain | Level | Steps to settle | Wall time | Fine solve to 0.1 % | Cheaper by | Flux error |
|---|---|---:|---:|---:|---:|---:|
| 512 long | 1 (2×) | 4,400 | 0.8 s | 6.6 s | **8×** | +28 % |
| 512 long | 2 (4×) | 5,800 | 0.2 s | 6.6 s | **33×** | +79 % |
| 1024 long | 1 (2×) | 8,800 | 2.9 s | 30 s | **10×** | +25 % |
| 1024 long | 2 (4×) | 14,200 | 0.7 s | 30 s | **43×** | +73 % |

Coarse levels are fast and systematically wrong in thin vessels. A vessel
of radius 5–9 voxels becomes 2.5–4.5 at level 1 and 1.25–2.25 at level 2.
Conductance goes as r⁴, so voxelisation errors of half a voxel become tens
of percent of flux. A preview is useful for layout, boundary-condition
checks and rough numbers, and should say which level it came from.

## 2. Coarse-to-fine initialisation (nested iteration)

Each level starts from the prolonged solution of the one below. Density
and velocity are interpolated trilinearly and rescaled (u ÷ 2, Δρ ÷ 4 per
level), and the fine run goes from there:

| Domain | Start | Fine steps to 1 % | Total to 1 % | vs from rest | Total to 0.1 % | vs from rest |
|---|---|---:|---:|---:|---:|---:|
| 512 | from rest | 3,200 | 3,200 | 1× | 5,600 | 1× |
| 512 | 2 levels, ρ and **u** | 1,800 | 2,441 | 1.3× | 4,641 | 1.2× |
| 512 | 2 levels, ρ only | 2,000 | 2,716 | 1.2× | 5,316 | 1.1× |
| 1024 | from rest | 10,600 | 10,600 | 1× | 14,000 | 1× |
| 1024 | 2 levels, ρ and **u** | 5,200 | 6,522 | **1.6×** | 17,722 | **0.8×** |

At best a modest gain, and on the longer domain the start was *slower* to
reach 0.1 %. The prolonged field carries the coarse levels' +25 % flux
bias, and the fine solve has to unlearn it. Most of the time from rest is
spent damping the start-up transient, not building the pressure field, and
a biased start does not remove that.

## 3. Fine-level "hybrid initialisation"

Fluent's hybrid initialisation solves a Laplace problem for a starting
field. The geometry-aware version is the lubrication pressure solve on the
*fine* mask. It has no coarse-level bias: its starting flux was within
1.4 % (512) and 13 % (1024) of the answer.

| Domain | Solve | Total to 1 % | vs from rest | Total to 0.1 % | vs from rest |
|---|---|---:|---:|---:|---:|
| 512 | 736 CG iterations, 1.5 s | 3,828 | 0.8× | 8,228 | 0.7× |
| 1024 | 1,093 CG iterations, 3.8 s | 9,760 | 1.1× | 19,960 | 0.7× |

A near-correct start did **not** help either. An equilibrium start with the
right mean flux still carries no non-equilibrium stresses, and the
lattice-Boltzmann field has to relax acoustically from it.

## 4. Operator coarsening: AMG

The same lubrication system, solved to 10⁻⁸ as the network grows 4×:

```{figure} ../_static/figures/amg_iterations.png
:width: 80%
:figclass: zv-figure

**Iterations to convergence against problem size.** Log scale. AMG's count
is flat; Jacobi-CG's grows with the domain.
```

| Length | Unknowns | Jacobi-CG | SA-AMG-CG | AMG levels | AMG setup + solve (CPU) |
|---:|---:|---:|---:|---:|---:|
| 256 | 674,086 | 1,333 it, 10.7 s | **12 it** | 4 | 1.4 + 1.4 s |
| 512 | 1,188,892 | 2,296 it, 33.1 s | **13 it** | 4 | 2.7 + 2.4 s |
| 1024 | 2,242,494 | 3,853 it, 110 s | **14 it** | 4 | 5.5 + 4.6 s |

Smoothed aggregation builds its coarse levels from the fine matrix
(operator complexity 1.5). They stay consistent with the fine problem,
whatever the vessels look like, and the iteration count barely moves. This
is the Ansys AMG result reproduced: CFX's coupled AMG scales near-linearly
for the same reason.

## Conclusions

| Technique | Verdict for zvCFD |
|---|---|
| Coarse level as preview | **Use**, labelled. 8–43× cheaper, 25–80 % flux error in thin vessels |
| Coarse-to-fine LBM initialisation | **Drop.** 0.8–1.6×, sometimes slower |
| Fine lubrication initialisation | **Drop** as an initialiser (0.7–1.1×). Keep as a seconds-scale estimator |
| AMG on elliptic sub-problems | **Use.** Flat iteration counts. Move to the GPU (AmgX or cupy SA) for 10⁸–10⁹ unknowns |
| Steady flows fast | Needs an elliptic (Stokes) solver with AMG; time-marching LBM will not get there by initialisation tricks |
