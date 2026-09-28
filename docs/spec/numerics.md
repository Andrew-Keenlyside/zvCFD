# Numerics

## Terms

**D3Q19**
: The lattice with 19 discrete velocities in 3-D: rest, 6 face neighbours,
  12 edge neighbours.

**Population** *fᵢ*
: The distribution function along velocity **c**ᵢ. Density and momentum
  are its moments.

**BGK**
: Single-relaxation-time collision towards equilibrium at rate `ω = 1/τ`.

**Lubrication model**
: Local Poiseuille approximation: every fluid voxel conducts like the
  centre of a channel whose half-width is its distance to the wall.

---

## Introduction

Two solvers are built. The lattice-Boltzmann solver is the main one: an
explicit, local, memory-bound time-marching scheme that works directly on
voxels. The lubrication solver is one elliptic equation, solved with
multigrid. It gives a fast approximate pressure and flow field and
boundary estimates. Both are written from the textbook and paper sources
below ([Clean-room policy](../how_to/cleanroom.md)).

---

## Technical reference

### Lattice-Boltzmann (`zvcfd.lbm`)

**Scheme.** Krüger et al., *The Lattice Boltzmann Method* (Springer 2017),
chapters 3, 5 and 6.

- Velocities, weights: D3Q19, `w₀ = 1/3`, `w₁…₆ = 1/18`, `w₇…₁₈ = 1/36`.
- Equilibrium: second order in **u**,
  `fᵢᵉ = wᵢ ρ (1 + 3 cᵢ·u + 9/2 (cᵢ·u)² − 3/2 u²)`.
- Collision: BGK, `fᵢ* = fᵢ + ω (fᵢᵉ − fᵢ) + Fᵢ`, viscosity `ν = (τ − ½)/3`.
- Forcing: Guo. `u = (Σ fᵢ cᵢ + F/2)/ρ`, and
  `Fᵢ = (1 − ω/2) wᵢ [3 (cᵢ − u) + 9 (cᵢ·u) cᵢ]·F`.
- Streaming: *pull*, two population buffers swapped each step.
  `fᵢ(x, t+1)` is read from `x − cᵢ`.
- Walls: half-way bounce-back. If `x − cᵢ` is solid, the pulled value is
  `f*_{ī}(x)`, the post-collision population of the opposite direction at
  `x`.
- Reservoirs: populations set to `wᵢ ρ_b` each step.

**Layout.** Structure of arrays: population `i` of cell `n` at `i·N + n`,
32-bit indices, `19·N < 2³¹` per buffer (checked at construction; larger
domains are partitioned). In fp16 mode the stored value is `fᵢ − wᵢ` in
IEEE half precision, and all arithmetic is fp32.

**Sparse bricks.** One CUDA block of 512 threads per active brick. The
27-entry neighbour table is staged in shared memory. A source outside the
brick is found through the table, and a missing neighbour is a wall. The
sparse kernel's output is bit-identical to the dense kernel's
(`tests/test_lbm_gpu.py`).

**Verification.** Plane Poiseuille flow between half-way bounce-back walls
(H = 32, τ = 1, Guo forcing) matches `u(z) = F/(2ν) z (H − z)` to within
1 % of the maximum. fp16 storage tracks fp32 within 5 % after 300 steps on
a porous sample.

**Bytes per update.** `2 · 19 · s + 1`, `s` the storage size: 153 B (fp32),
77 B (fp16). The kernel is memory-bound: 91–96 % of device copy
bandwidth dense fp32, 74–78 % sparse fp32, on the A2000.

**Known limits** (roadmap): BGK is unstable at low τ and high Reynolds
number (TRT/cumulant planned). Stair-case walls limit wall-shear accuracy
(interpolated bounce-back planned). Compressibility error scales with Ma²
(keep lattice velocity ≤ 0.1).

### Lubrication pressure solve (`zvcfd.solvers.lubrication`)

`∇·(k ∇p) = 0` on fluid voxels, `k = ¾ d²` (`d` the Euclidean distance to
the half-way wall, from `scipy.ndimage.distance_transform_edt`). This is
exact for a circular tube: `∫ k dA = πR⁴/8`, Poiseuille's conductance. Face
conductances are harmonic means. Walls carry no flux. Dirichlet values
are set on any voxel set. Velocity is `u = −(k/μ) ∇p`.

Solvers: smoothed-aggregation AMG as a CG preconditioner (pyamg, CPU),
12–14 iterations to 10⁻⁸ independent of domain length in our tests; or
Jacobi-preconditioned CG on the GPU (cupyx), whose iteration count grows
with the domain.

Accuracy: −17 % in flux on a 6-voxel-radius voxelised tube (the discrete
distance transform underestimates d near the wall). The error falls as the
resolution rises. It is an estimator, not a replacement for the 3-D solve.
