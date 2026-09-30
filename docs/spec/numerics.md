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
- Collision: BGK, `fᵢ* = fᵢ + ω (fᵢᵉ − fᵢ) + Fᵢ`, viscosity `ν = (τ − ½)/3`;
  or **TRT** (two relaxation times; Ginzburg): ω on the symmetric parts
  of each pair (i, ī), ω⁻ on the antisymmetric parts, with the magic
  parameter `Λ = (1/ω − ½)(1/ω⁻ − ½) = 3/16`. That puts half-way
  bounce-back walls exactly mid-link for straight walls, and makes steady
  solutions independent of τ in the bulk and at walls (the pressure
  boundaries add a small τ dependence; see below).
- Rheology (optional): **Carreau–Yasuda**,
  `ν(γ̇) = ν∞ + (ν₀ − ν∞)[1 + (λγ̇)ᵃ]^((n−1)/a)`, applied per voxel through a
  local ω. The shear rate comes from the non-equilibrium second moment,
  `γ̇ = √2 · (3ω/2ρ)·|Π⁽ⁿᵉᑫ⁾|`, with no finite differences; ω and γ̇ are
  solved by fixed-point iteration from the zero-shear value until ω
  settles to 10⁻⁶ (at most 12 iterations; a fixed three left a 0.6 %
  profile error that did not fall with resolution), and ω is clipped to
  [0.05, 1.95].
- Forcing: Guo. `u = (Σ fᵢ cᵢ + F/2)/ρ` from the pre-collision
  populations, and `Fᵢ = (1 − ω/2) wᵢ [3 (cᵢ − u) + 9 (cᵢ·u) cᵢ]·F` (with
  TRT, split into its even and odd parts and relaxed with ω and ω⁻). The
  buffers hold *post*-collision populations, whose momentum already
  carries the whole step's force, so output fields use
  `u = (Σ f*ᵢ cᵢ − F/2)/ρ`.
- Streaming: *pull*, two population buffers swapped each step.
  `fᵢ(x, t+1)` is read from `x − cᵢ`.
- Walls: half-way bounce-back. If `x − cᵢ` is solid, the pulled value is
  `f*_{ī}(x)`, the post-collision population of the opposite direction at
  `x`.
- Inlets and outlets: pressure outlets on surface caps by anti-bounce-back
  on the links that cross the cap. If `x − cᵢ` lies across the cap, the
  pulled value is `−f*_{ī}(x) + 2 wᵢ ρ_b [1 + 4.5 (cᵢ·u)² − 1.5 u²]`.
  Velocity inlets, and patches on the box faces, by Guo non-equilibrium
  extrapolation (flags 4 and 3). See [Boundary conditions](boundary_conditions.md).
- Reservoirs (legacy): populations set to `wᵢ ρ_b` each step.

**Layout.** Structure of arrays: population `i` of cell `n` at `i·N + n`,
32-bit indices, `19·N < 2³¹` per buffer (checked at construction; larger
domains are partitioned).

**Shifted populations.** Buffers hold `gᵢ = fᵢ − wᵢ`, and every kernel
computes with `g`: `ρ = 1 + Σ gᵢ`, `j = Σ gᵢ cᵢ`, and the equilibrium as
`gᵢᵉ = wᵢ [δρ + ρ (3 cᵢ·u + 9/2 (cᵢ·u)² − 3/2 u²)]`. Patch and reservoir
densities reach the kernels as `ρ − 1`. In slow flows, populations differ
from `wᵢ` only in the fourth or fifth significant digit, so fp32
arithmetic on the full `fᵢ` loses most of the signal. Before the change,
steady Stokes fluxes wandered by ±0.1 % for tens of thousands of steps and
drifted with τ by up to 12 % in a porous sample. With it, they are
τ-independent to 10⁻⁵ in tubes. fp16 mode stores `g` in IEEE half
precision; all arithmetic is fp32.

**Sparse bricks.** One CUDA block of 512 threads per active brick. The
27-entry neighbour table is staged in shared memory. A source outside the
brick is found through the table, and a missing neighbour is a wall. The
sparse kernel's output is bit-identical to the dense kernel's
(`tests/test_lbm_gpu.py`).

**Verification** (`tests/test_lbm_gpu.py`, `tests/test_physics.py`,
`tests/test_geometry.py`):

| Check | Result |
|---|---|
| sparse kernel = dense kernel | bit-identical |
| multi-partition = one partition (3 partitions, patches, TRT) | bit-identical |
| BGK plane Poiseuille, τ = 1 | within 1 % of analytic |
| TRT plane Poiseuille (body force), τ = 0.55 / 0.8 / 2 | exact to 2 × 10⁻⁶ (test tolerance 10⁻⁴) |
| TRT steady flux vs τ (0.6–2), pressure-driven voxel pipes and ducts | independent of τ to 10⁻⁵ |
| TRT permeability vs τ, periodic body-force porous sample | independent of τ to 2 × 10⁻⁴ |
| TRT steady flux vs τ, pressure-driven porous and vessel samples (4 open buffer layers) | 0.4–0.6 % spread from τ = 0.6 to 2 |
| pressure-driven channel (pressure patches) | within 2 % of analytic |
| Carreau–Yasuda channel vs the exact stress-balance solution | 1.6 × 10⁻⁴, second order ([Validation](../validation/exact_solutions.md)) |
| Womersley, Taylor–Green, sphere arrays, DFG cylinder | see [Validation](../validation/index.md) |
| straight voxel pipes vs Poiseuille (area-equivalent radius) | 4.2 % / 1.9 % / 1.0 % low at radius 6 / 12 / 24 voxels (staircase); OpenFOAM on the same voxels 1.6 / 1.2 / 0.8 % low ([OpenFOAM comparison](../benchmarks/openfoam.md)) |
| square duct vs exact series solution | +0.37 % at 16 voxels across, +0.09 % at 32 |
| fp16 storage vs fp32 | within 5 % after 300 steps (porous sample) |

**Steady flows and τ.** Time-marching LBM reaches a steady state only as
fast as its slowest modes decay. With TRT the steady answer does not
depend on τ, so in Stokes-regime steady runs τ is purely a convergence
setting. Two processes compete:

- **Momentum diffusion** across a vessel or pore, time `~ r²/ν`. It is
  faster at large τ.
- **Pressure equilibration** along the sample. In a weakly compressible
  solver this behaves like diffusion with coefficient `c_s² K/ν`
  (`K` the permeability), so it is faster at small τ.

Straight tubes are limited by the first and porous or branched samples by
the second. Measured on the benchmark geometries
(`benchmarks/openfoam/tau_scan.py`, time to 0.1 % of the converged flux,
RTX A2000):

| Case | τ = 0.6 | 0.8 | 1.0 | 1.5 | 2.0 |
|---|---:|---:|---:|---:|---:|
| pipe, radius 24 voxels (0.69 M cells) | 14.2 s | 4.7 s | 3.2 s | 2.2 s | **1.7 s** |
| porous sample (0.92 M cells) | **2.4 s** | **2.4 s** | 3.5 s | 6.6 s | 10.1 s |
| vessel network (1.19 M cells) | **7.8 s** | 12.8 s | 21.7 s | 43.7 s | 65.8 s |

τ = 0.8 is within 3× of the best setting in every case, and is the
default for steady Stokes benchmarks. At higher Reynolds numbers τ is
capped by the lattice Mach number (`u = Re·ν/D` in lattice units), so
convergence is also judged on the mass balance, not only on flux changes.

**Bytes per update.** `2 · 19 · s + 1`, `s` the storage size: 153 B (fp32),
77 B (fp16). The kernel is memory-bound: 98 % of device copy
bandwidth dense fp32 and 77 % sparse fp32 on open domains, on the A2000.

**Known limits** (roadmap): staircase walls limit wall-shear accuracy
(interpolated bounce-back planned). Compressibility error scales with Ma²
(keep lattice velocity ≤ 0.1). Steady convergence is bounded by pressure
equilibration in large low-permeability samples (a coarse-level pressure
initialisation, see [Multiresolution](multiresolution.md), is the planned
remedy). The pressure boundaries extrapolate the non-equilibrium part
from one neighbour, which assumes locally developed flow. A face cut
through a porous sample gave a 7 % τ spread; open buffer layers at the
ends reduce it to under 1 %.

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
