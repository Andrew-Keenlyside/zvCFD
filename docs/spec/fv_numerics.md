# Finite-volume numerics

**Status: in development.** This page specifies what
`zvcfd.fv.geometry`, the CPU reference solver `zvcfd.fv.reference`, the
GPU solver `zvcfd.fv.solver` (kernels in `zvcfd.fv.gpu`, linear solvers in
`zvcfd.fv.linear`) and its partitioned form `zvcfd.fv.multi` implement
today. The GPU solver evaluates the same discrete equations as the
reference, and reaches the reference's solutions to 10⁻¹¹. See the
[plan](../feasibility/fv_plan.md).

## Terms

**Control volume (CV)**
: The volume `V_i` belonging to mesh node `i`: one sub-control-volume
  (SCV) from each element around the node, bounded by surfaces through
  edge midpoints, face centroids and element centroids (the median dual).

**Integration point (ip)**
: The piece of CV surface inside one element between the SCVs of the two
  nodes of an element edge, and the point on it where fluxes are
  evaluated. There is one per element edge: 6 per tetrahedron, 8 per
  pyramid, 9 per wedge, 12 per hexahedron.

**Block**
: The 4 × 4 coupling between the (u, v, w, p) unknowns of two nodes.

**Rhie–Chow term**
: The pressure-redistribution part of the ip mass flow that couples
  pressure to continuity and removes checkerboard modes.

---

## Introduction

The solver is an element-based finite-volume method in the manner of
Ansys CFX (Schneider & Raw 1987). Unknowns sit at the mesh nodes, and each
node's control volume is assembled from pieces of the elements around it.
Inside an element, every quantity at an integration point comes from that
element's shape functions: pressure, velocity and their gradients. So
diffusion and pressure gradients need no non-orthogonality corrections,
and each element can be assembled on its own. Pressure and velocity are
solved together, as one system of 4 × 4 blocks.

---

## Technical reference

### Geometry (`zvcfd.fv.geometry`)

For element edge `(a, b)` the ip face is the loop `(m_ab, f_1, c, f_2)`:
edge midpoint, the centroids of the two element faces sharing the edge,
and the element centroid. A face centroid is the mean of the face's nodes,
and the element centroid the mean of the element's nodes. The ip area
vector, pointing from `a` to `b`, is

    A = ½ (c − m_ab) × (f_2 − f_1),

which every surface spanning the (generally non-planar) loop shares. The
orientation of each loop is fixed once, on the reference element. SCV
volumes come from the divergence theorem over the ip faces around the
node, each fanned into four triangles about its centre. The element-face
pieces of the SCV surface lie in planes through the node and add nothing.

Two identities hold to round-off, and `dual_geometry` checks them on every
mesh:

| Check | Generated meshes (all types, distorted) | HiP-CT coronary mesh |
|---|---|---|
| Σ node volumes = Σ element volumes (faces fanned about their centroids) | ≤ 1.5 × 10⁻¹⁶ relative | 9.0 × 10⁻¹⁶ |
| Each CV closed: outward area vectors (ip faces + boundary sub-faces) sum to 0, relative to `V^{2/3}` | ≤ 3 × 10⁻¹⁵ | 5.4 × 10⁻¹¹ (coordinates ~100 mm, smallest CV 6 × 10⁻⁸ mm³) |

**Shape functions** (isoparametric, linear-complete):

| Type | Reference element | Shape functions |
|---|---|---|
| tet | unit tetrahedron | `1 − ξ − η − ζ, ξ, η, ζ` |
| wedge | unit triangle × [−1, 1] | `(1 − ξ − η, ξ, η) × (1 ∓ ζ)/2` |
| hex | [−1, 1]³ | trilinear |
| pyramid | base [−1, 1]² at ζ = −1, apex (0, 0, 1) | collapsed hex: base `⅛(1 ± ξ)(1 ± η)(1 − ζ)`, apex `(1 + ζ)/2` |

Each ip sits at its loop's centre on the reference element, mapped back
to parametric coordinates. On affine elements (every tetrahedron, and
undistorted other types), the discrete gradient `(1/V_i) Σ_ip p_ip A` of a
linear field is then exact at interior nodes. On distorted hexahedra,
wedges and pyramids it carries the usual O(1) consistency error of
isoparametric integration points: 0.1–1.6 % at 25 % random node
displacement.

### Coupling pattern (`zvcfd.fv.pattern`)

A node's equations involve every node of every element around it. On the
coronary mesh that gives **18.7 blocks per row** on average (median 18,
p99 27, max 46), and 106.0 M blocks for its 5.67 M nodes.

### Discrete equations (`zvcfd.fv.reference`)

For each CV, with outward ip areas and the boundary sub-faces:

    ρ V_i (u_i − u_i°)/Δt + Σ_ip ṁ_ip u_ip + Σ_ip p_ip A − Σ_ip μ (∇u + ∇uᵀ)_ip · A + boundary = f_i V_i
    Σ_ip ṁ_ip + boundary mass flow = 0

| Term | Discretisation |
|---|---|
| `p_ip`, `(∇u)_ip` | shape functions of the element that holds the ip |
| `ṁ_ip` | `ρ [ū_ip · A + d_ip (∇̄p_ip − ∇p_ip) · A] + f_ip Σ_l (c_l/c_0)(ṁ_ip^l − ρ ū_ip^l · A)`, with `ū, ∇p` from shape functions, `∇̄p` interpolated nodal gradients, `d_ip` the mean of `V/(a_P + t)` at the edge's two nodes (`t` the time term's diagonal), and `f_ip = 1 − d_ip / mean(V/a_P)` (Rhie & Chow 1983; Choi 1999). Then `d_ip/(1 − f_ip) = mean(V/a_P)` exactly: a steady state does not depend on the (false or physical) time step, and halving a control volume at a symmetry plane changes nothing |
| nodal gradient `∇̄φ_i` | element gradients averaged with SCV weights: exact for linear fields at every node, boundary nodes included |
| advection `u_ip` | upwind in the lagged `ṁ`, plus a deferred correction to `u_up + β ∇u_up · (x_ip − x_up)`; `β` fixed (Specified Blend) or from a Barth–Jespersen limiter (High Resolution), frozen after 10 Picard iterations (left free, the non-differentiable limiter holds Picard in a limit cycle) |
| viscosity | constant, or `μ(γ̇)` at each ip from the shape-function strain rate (`zvcfd.rheology`: Carreau–Yasuda, Cross, power law, Casson, with CFX-style shear-rate clips); the `μ (∇u)ᵀ` stress term is kept for generalised-Newtonian fluids and dropped for constant viscosity, where it integrates to zero |
| time | a false time step (pseudo-transient) for steady runs, or BDF1 / BDF2 with Picard coefficient loops per step, the first BDF2 step BDF1. BDF2 takes a variable step: with `ω = Δt_n/Δt_{n−1}`, `c = (1 + 2ω)/(1 + ω), 1 + ω, −ω²/(1 + ω)` (`3/2, 2, −1/2` at `ω = 1`), exact for quadratics at any ratio; observed order 1.9–2.0 with ratios 0.6 and 1.67 alternating |

The reference solver takes `∇̄p` implicitly (a sparse operator product, which
widens the pressure stencil), so a Stokes problem is one linear solve. A
solver that lags it, as CFX does, converges to the same discrete
solution. For smooth pressure modes that fixed-point iteration contracts
slowly (≈ 0.7 per iteration on a hexahedral box), because `∇̄p` and `∇p`
nearly cancel.

### Boundary conditions

| Zone kind | Velocity | Pressure | Mass flow |
|---|---|---|---|
| wall | `u = 0` at the zone's nodes | from continuity | 0 |
| velocity | prescribed at the nodes (`f(x)` or `f(x, t)`); a moving wall is a velocity zone | from continuity | `ρ u · A_sub`, with `u` interpolated on the face to each sub-face's area centroid (11/18 and 7/36 on triangles; 9/16, 3/16, 1/16 on quadrilaterals): exact for linear profiles |
| symmetry (axis-aligned plane) | normal component 0; tangential components solved with no boundary flux | from continuity | 0 |
| velocity, flow rate | a profile shape along the inward normal, scaled every step so the discrete inflow is exactly `ρ Q(t)`: `plug`, `poiseuille` (below), or `f(x)` | from continuity | `ρ Q(t)` exactly |
| pressure | momentum solved; viscous flux zero-normal-gradient (only `μ (∇u)ᵀ · A`, lagged), or a prescribed traction from a given `∇u` (manufactured solutions); boundary momentum `ṁ_b u_node` in either direction (`backflow: consistent`, default) or out only; first order where the flow enters (below); optional backflow stabilisation (below) | prescribed at the nodes and, for the pressure force, at sub-face centroids (`f(x)` or `f(x, t)`) | the consistent flux: what the interior sends to those nodes |
| pressure, lumped (GPU) | as pressure | uniform, `p = a + r Q`, implicit (below) | the consistent flux |
| pressure, average static (GPU, one partition) | as pressure | the upstream-extrapolated profile (lagged) shifted so its area mean is the value; `blend` (0.05) of a uniform profile, as CFX | the consistent flux |
| pressure, opening (GPU) | as pressure | `value` where flow leaves; `value − ½ρ|u|²` where it enters (total pressure, lagged) | the consistent flux |

**Developed inlet profile.** `poiseuille` solves `−Δu = 1` on the inlet
face itself (linear triangles in the face's plane, `u = 0` on its rim):
the fully developed laminar profile of that cross-section, whatever its
shape, which on a circle is the parabola (to 3 % at the nodes of a
6 × 6 O-grid face). Only the shape matters; the flow rate fixes the scale.

**Momentum entering through a pressure boundary.** Inside the domain the
advected value at a flux point is second order, `u_up + β ∇u_up · (x_ip −
x_up)`. Momentum crossing a pressure boundary is `ṁ_b u_node`, first order.
Where the flow *enters* through a pressure boundary (`ṁ_b < 0`, lagged),
the faces leaving the node carry their correction, and in the node's own
row nothing balances it: in developed flow the inflow arrives at `u_node`
and leaves at `u_node` plus the cross-stream part of the correction. That
first row of control volumes was therefore inconsistent: a truncation error
of 0.0101 of the momentum throughput, against 0.0005 inside, on a 9.5 k-node
pipe in developed flow. A natural inflow barely constrains the incoming
profile, so the error grew into an inflow jet. A pressure-driven pipe's flow
rate came out 2–4 % high and did not shrink with refinement (+1.8 %, +3.4 %
and +3.8 % with 4, 6 and 12 cells across; the developed pipe in
[validation](../validation/fv_solver.md#developed-pipe-flow-through-pressure-boundaries)).

The treatment now has two parts:

1. **The corrections of the faces an inflow node is upwind of stay out of
   the node's own row** and go to the downstream rows only. The node's
   boundary flux carries exactly that momentum in, so the scheme stays
   conservative (`GPUSolver._own_inflow`; the CPU reference's momentum
   balance counts it as momentum entering through the zone). Where the
   faces leaving the node mirror its boundary faces, as on prism and
   hexahedral inflow rows, that is the consistent boundary flux.
2. **Where they do not mirror them, the cross-stream part of the
   difference is added:** `|ṁ_b| β ∇u_node · Δd`, on the right-hand side.
   `Δd` is the offset between the mean flux point of the faces leaving the
   node (weighted by their area along the inward normal `n`) and the mean
   boundary sub-face centroid (weighted by area), with its component along
   `n` removed (`zvcfd.fv.boundary_advection.inflow_offsets`). It is pure
   geometry, computed once, and zero on extruded meshes.

The part along `n` is left out deliberately. The faces leaving an inflow
node lie half a cell downstream, so their correction includes `½ h ∂u/∂n`
with the node's gradient. At a boundary node that gradient is one-sided,
from the next node downstream, so the half-cell would be differenced
centrally. Central differencing is unstable at cell Péclet numbers above
about 2. That is what made the full second-order boundary reconstruction
(below) diverge with 6 cells across (cell Péclet number about 8) and
converge with 12 and 18 (about 4 and 2.8, with the limiter's help).
Developed flow has no gradient along `n`, so the cross-stream part is all
that consistency in developed flow needs.

Results against Stokes on the developed pipe:

- **extruded triangles:** +0.56 %, +0.20 % and +0.12 % with 6, 12 and 18
  cells across (`Δd = 0`);
- **structured tetrahedra (wedge wall layers around a split-hexahedron
  core):** +0.60 %, +0.66 %, +0.08 %, −0.44 % and −0.58 % with 3, 4, 6, 12
  and 18 cells across. With part 1 alone it was −0.85 %, −1.53 %, −1.71 %,
  −1.46 % and −1.33 %. About 0.5 % does not shrink, for every second-order
  treatment including the full reconstruction (+0.51 %, +0.43 %): the
  split's diagonals all lean the same way;
- **Delaunay tetrahedra (unstructured connectivity):** −7.8 %, −0.59 %,
  −0.46 % and −0.23 % with 6, 12, 18 and 24 cells across, converging.

Nodes with fixed velocity are left out: their rows are replaced, and their
reactions keep the corrections. So are zones that count only outflowing
momentum (`backflow: outflow`).

Two simpler treatments were tried and rejected. Dropping the corrections at
inflow nodes altogether (`β = 0` there) only moves the mismatch one row in:
the extruded pipe then gives +1.30 % and +1.25 %, and the tetrahedral pipe
−4.1 % and −2.9 %, the jet reversed. First-order upwind everywhere is exact
on the extruded pipe, where every edge is axial or in the inlet plane, but
inconsistent on the tetrahedral core, whose split diagonals all lean the same
way: on the developed flow its advection residual stays at 0.4 ρU²/R RMS
however fine the mesh, and the flow rate is −15 % at every level.

The full second-order boundary reconstruction remains as an option
(`boundary_reconstruction = True`, `zvcfd.fv.boundary_advection`). It
reconstructs each boundary sub-face's value from the node's gradient, with
the sub-face's share of the boundary mass flow. Where it converges it is
accurate: +0.25 %, +0.51 % and +0.43 % against Stokes on the tetrahedral
pipe with 6, 12 and 18 cells across, −0.03 % and +0.003 % on the extruded
one with 12 and 18. But with 6 cells across, the extruded pipe diverges with
it: with exact or inexact linear solves, explicit or implicit, limited or
not, and with false time steps down to 0.03. The growing mode sits at the
inlet and flattens the inflow profile. That is the central differencing
along `n` described above. It is off by default.

**Backflow stabilisation.** `backflow_stabilisation: β` adds
`β (ṁ_b)₋ u_node` to the outflowing momentum at pressure-zone nodes where
the boundary flow enters (`(ṁ_b)₋ = max(−ṁ_b, 0)`): the node-based form of
the term of Esmaily Moghadam et al. (2011), which removes that fraction
of the kinetic energy that backflow carries in. `β = 1` removes all of
it, as `backflow: outflow` does; SimVascular's default is 0.2, and so is
`zvcfd run`'s.

A node on several zones takes velocity from a wall first, then an inlet,
and pressure from an outlet. At symmetry-plane nodes, nodal gradients are
made mirror-symmetric: zero normal derivative for pressure and for the
tangential velocity components, zero tangential derivatives for the
normal component. Without that, one-sided gradients break the symmetry. With it,
half a channel with a symmetry plane reproduces the full channel to
round-off (`tests/test_fv_reference.py`). With no pressure zone, one node's pressure is
pinned. Because outlet mass flows are the consistent fluxes, inflow equals
outflow to round-off on every mesh.

### Lumped outlets, implicitly coupled (`zvcfd.fv.solver`)

A zone with a 0-D model (`zvcfd.lumped`: RCR, open-loop coronary) has a
uniform pressure `p = a + r Q`, where `Q` is the zone's volume outflow and
`(a, r)` the model's coefficients for the step (trapezoidal or backward
Euler), or its steady pair (`r` the total resistance) in a steady solve.
`Q` is linear in the unknowns: it is minus the sum of the zone nodes' raw
continuity rows, `Q = w · x + c`. So each of the zone's pressure rows
becomes `p_j − r (w · x + c) = a`: the sparse identity row plus the
rank-one term `u wᵀ` (`u = −r` at those rows). The Krylov solver applies
the rank-one terms in every product; the multigrid preconditioner sees
only the sparse part, which costs about one extra FGMRES iteration per
zone (with AmgX, zvCFD's FGMRES runs with one AmgX V-cycle as the
preconditioner). The pressure force on the zone, `p_j A_j`, goes into the
matrix too. Coupled this way, two outlets with resistances 1000 and 2000
times the domain's converge in 40–70 outer iterations (a lagged pressure
update diverges once `r` exceeds the domain's resistance), and after each
time step the model's state is committed with that step's final flow.

### The outer loop on the GPU (`zvcfd.fv.solver`)

Each outer iteration assembles the linearisation about the current state
(Picard in the lagged mass flows, deferred correction, lagged `∇̄p`),
solves it to a relative residual reduction `linear_rtol` (0.1 by default,
as CFX's coupled solver does), and updates the Rhie–Chow mass flows from
the new state. Three details make that robust:

- **Dimensionless rows.** Before the linear solve, momentum rows are
  divided by `a_P U_ref`, continuity rows by `ρ U_ref V^{2/3}`, a lumped
  zone's rows by `ρ U_ref A_zone √n` (its patch-level flux, repeated at its
  `n` nodes), and fixed rows by their variable's scale. FGMRES minimises
  the 2-norm of the residual; with rows in SI units, one equation's units
  dominate it, a single Krylov step removes that equation's residual, and
  the solve "converges" with the rest unsolved. Scaling rows does not
  change the solution, and the block smoothers are invariant to it. It
  took a steady RCR pipe in SI units from 130 outer iterations to 38.
- **Pressure level.** The interior starts at the area-weighted mean of the
  pressure zones' values (a lumped zone counts as `a + r Q` with the inflow
  shared in proportion to `1/r`), not at zero. The lagged `∇̄p` of a zero
  interior next to an outlet at `p_out` is a spurious jump of `p_out/Δx`
  that the iterations then take long to forget; the implicit reference is
  level-invariant, the lagged form only at convergence. With the level set,
  shifting every pressure by 1000 changes velocities by 7 × 10⁻¹³.
- **Convergence test.** Converged means the largest relative changes of
  `u` (over `max |u|`) and `p` (over its range) are below the tolerance
  *and* the last linear solve met its tolerance: a stalled linear solve
  also makes the changes small. With large lumped resistances, round-off
  in `Q` amplified by `r` floors the pressure change near 10⁻⁹ of the
  pressure range; use tolerances of 10⁻⁸ or looser there.
- **Boundary rows in one pass.** On one GPU, after the element
  kernels, one kernel (`zvcfd.fv._cuda_bc`) finishes every row. Its
  thread `(node, component)` owns that row in every block of the row, so
  no two threads write the same entry and the result is deterministic.
  It adds the known terms (time, body force, outlet pressure force,
  inflow, outflowing momentum, a lumped outlet's implicit pressure
  force), then copies the rows the solver keeps for reactions and lumped
  flows. It replaces fixed rows by identity rows, and moves the columns
  of known values to the right-hand side (**symmetric elimination**: the
  same solution, and multigrid no longer mixes identity rows into its
  coarse operators). It accumulates the residual statistics of the report
  and applies the row scaling above, with `U_ref` and `p_ref` read from
  device memory. The iteration then reads back to the host once: the
  residual report, the changes of `u` and `p`, and the zone flows,
  together. Without elimination this gives bit-identical iterates to the
  staged path it replaces; with it, the same solution to 2 × 10⁻¹⁵. A
  partitioned solver keeps the staged path, since it exchanges halos
  between the stages.

AmgX rebuilds its hierarchy for every matrix (`structure_reuse_levels: 0`).
Reusing the whole hierarchy (`−1`) through `AMGX_solver_setup` keeps the
coarse operators of the first matrix, and FGMRES stalls as soon as the
linearisation moves away from it. With `reuse` set, zvCFD instead calls
`AMGX_solver_resetup`, which keeps aggregates and colourings and
recomputes the operators, and rebuilds when the linear iterations grow by
half. That works on small systems, but AmgX 2.5 fails on the first
re-set-up of larger ones ("Matrix was not initialized"), so it is off by
default.

**Element geometry: stored or recomputed.** The kernels need each
element's flux-point areas and centroids, sub-volumes and shape-function
gradients. They can recompute them from the node coordinates in every
kernel (no memory, most of the double-precision arithmetic) or read them
from arrays filled once, with the assembly's block positions
(`store_geometry`: 0.8 KB per tetrahedron, whose gradients are constant;
2.8, 3.6 and 6.1 KB per pyramid, wedge and hexahedron). The default,
`"auto"`, stores everything if it fits in a fifth of the free device
memory, else tetrahedra only; on the coronary mesh that is the 6.5 M
tetrahedra (5.0 GB), not its 8.2 M wedges (30 GB). The results are the
same to round-off either way.

### Linear solvers on thin cells (`zvcfd.fv.linear`)

The coupled system is a saddle point: a pressure row's diagonal is only
the Rhie–Chow term, while its couplings to the neighbouring velocities
(the divergence) are much larger. Point-block smoothers (block Jacobi,
AmgX's DILU) see only each node's own 4 × 4 block, and damped block
Jacobi on that system has an iteration matrix of spectral radius above
one: more sweeps make it worse. On cells of comparable size the coarse
grids hide this; on cells thin in one direction across the whole domain
(a one-cell quasi-2-D slab, a slab of thin cells) they do not, because
`d = V/a_P` is set by the strong wall-normal viscous coupling and the
pressure diagonal shrinks further. Thin prism layers at a wall slow them
down without stopping them. Measured on an idle GPU, iterations of FGMRES
to 10⁻⁶ ("—" = not converged in 300 iterations for AmgX, 400 for SIMPLE
and ACM; systems from `benchmarks/fv/capture_hard_systems.py`):

| System | AmgX symmetric GS, pressure-weighted (`gs`, default) | AmgX DILU (the usual set-up) | AmgX ILU(0), pairwise, pressure-weighted (`robust`) | AmgX DILU, pressure-weighted | SIMPLE block preconditioner | zvCFD ACM |
|---|---:|---:|---:|---:|---:|---:|
| gate B tube (wedge layers, finest at the wall) | 26 | 24 | 13 | 35 | 55 | 39 |
| DFG 2D-2 slab, one cell deep, Δt = 0.005 s | 17 | — | 11 | — | — | — |
| tube with strongly graded wall layers (growth 0.45) | 17 | 24 | 9 | 28 | 41 | 51 |
| slab with cells 10× thinner than wide | — | — | 283 | — | — | — |

The two tubes were captured again on 1 October, after a fix to the tube
generator, whose `growth` had graded the layers toward the core. On the
old tubes, coarsest at the wall, DILU had not converged on the graded
layers (200) and `robust` had needed 90–127 iterations.

The outer loop asks for a reduction of 0.1, not 10⁻⁶. At 0.1, `gs` takes
2–3 iterations on the first three systems and `robust` 1–2. `robust` also
solves the thin slab (12 iterations), where `gs` and DILU fail. DILU with
two post-sweeps stalls on the DFG slab: its post-sweeps amplify the
saddle-point modes that a symmetric forward-and-back sweep damps. Script: `benchmarks/fv/amgx_smoothers.py`;
results: `benchmarks/results/fv/amgx_smoothers.json`. On three systems of
the SimVascular coronary model (398 k nodes, 2.1 M tetrahedra), `gs` takes
1–4 iterations and 0.18–0.31 s. The old DILU default takes 13–53 iterations
and 0.9–3.5 s, and `robust` 1–3 iterations and 0.32–0.42 s, its ILU set-up
being the larger cost. Two earlier coronary systems in that file came from
a defective crop (singular) and are kept for the record only. `gs` smooths
the coarsest level (4 DILU sweeps) instead of using AmgX's dense LU, which
fails in mixed precision ("Fail to get info from cudense").

Several things help. A symmetric block Gauss–Seidel sweep treats the
velocity–pressure couplings of each colour implicitly. ILU smoothing sees the velocity–pressure couplings a
point-block smoother misses (ILU(1) is stronger still, but its fill
overflows AmgX's per-row shared memory on tetrahedral rows). Measuring
coupling strength on the pressure–pressure entry
(`aggregation_edge_weight_component: 3`) aggregates along the Rhie–Chow
Laplacian, which follows the mesh's anisotropy. The `SIMPLE` solver
splits the system (an AMG V-cycle on the momentum block, one on the
approximate pressure Schur complement `C − B diag(F)⁻¹ Bᵀ`, then the
velocity correction) inside FGMRES on the exact coupled operator. No one
configuration wins everywhere, so `linear: auto` (`zvcfd.fv.linear.Auto`)
tries `gs`, then `robust`, then pressure-weighted DILU, then SIMPLE, and
keeps the first that reduces the residual; switches are logged. On meshes
of good quality `robust` converges in about half the iterations of `gs`,
so the ladder also solves its third system with `robust` and keeps
whichever was faster by at least 20 %. If a probed-in `robust` later
fails, the ladder returns to `gs`. The last row is an
extreme case (aspect ratio 10 in every cell). `robust` solves it to 0.1 in
12 iterations, but to 10⁻⁶ only slowly. zvCFD's own ACM keeps block-Jacobi
smoothing and is the check solver for meshes without thin cells: on the
169 k-node gate B tube, whose wall layers are now thin, it no longer
converges.

### Wall shear stress

`wall_shear()` gives `τ` at wall nodes as Ansys CFX evaluates laminar
walls (`zvcfd.fv.wall`). Each wall face is split among its nodes into
sub-faces, and each sub-face carries a boundary integration point at its
area centroid. There the owning element's shape functions give the
velocity gradient `∇u = Σ_n u_n ⊗ ∇N_n(ξ_ip)`, and `τ` is the tangential
part of the fluid's traction on the wall, `−μ (∇u + ∇uᵀ) · n`. For a
shear-thinning fluid, `μ` is taken at the integration point's shear rate.
A node's value is the area-weighted mean over its wall sub-faces.

The shape functions reproduce linear fields, so `τ` is exact for any
affine velocity on every element type, distorted or not
(`tests/test_fv_wall_shear.py`). For curved profiles it is first order in
the wall cell's thickness, and it reads low. On tetrahedra the gradient is
the element's constant one, the same as a P1 finite-element gradient. Its
accuracy therefore rests on near-wall resolution, as in CFX, which pairs
it with inflation layers.

`wall_shear("reaction")` instead takes `τ` from consistent reactions:
minus the residual of the node's momentum equation before no-slip
replaced it, less the inlet sub-faces' pressure force and momentum where
the node also borders an inlet, projected tangentially on the node's
area-weighted normal, over its wall area. It closes the axial force
balance of developed pipe flow `τ P = −(dp/dz) A` to 10⁻⁴, and
`zone_forces()` uses it. Locally, though, it absorbs the discretisation
error of the advective flux into the wall control volumes. That error is
small with thin wall cells and large on coarse ones.

Measured bias against exact solutions
(`benchmarks/fv/wall_shear_estimators.py`, steady Navier–Stokes,
ρ = 1060 kg/m³, μ = 4 mPa·s):

| Case | Mesh | `gradient` (default) | `reaction` |
|---|---|---:|---:|
| Poiseuille, R = 1 mm, Re 106 | Delaunay tetrahedra, R/h = 2.8 / 5.6 / 8.3 | −13.3 / −10.8 / −9.8 % | +44.7 / +4.5 / +2.2 % |
| | prism layers, wall cell R/10 / R/32 / R/129 | −5.6 / −2.1 / −1.5 % | +0.4 / −0.2 / −1.0 % |
| Stagnation flow (Hiemenz), δ = √(ν/a) | Delaunay tetrahedra, h = 1.5 / 0.5 / 0.25 δ | −48 / −19 / −9.5 % | +108 / +3.6 / +0.7 % |
| | hexahedral layers, wall cell 0.5 / 0.2 / 0.1 δ | −14.8 / −5.9 / −3.0 % | +3.1 / +1.2 / +0.8 % |

So `gradient` needs a wall cell of about R/30 in a vessel, or about 0.1 δ
at a stagnation point, to come within 2–3 %. With inflation layers both
methods converge. Without them, `gradient` reads low and `reaction`
reads high. Over a transient, `zvcfd.fv.wss.WallShearStats` accumulates
TAWSS, OSI and RRT by the trapezoidal rule.

### Partitions (`zvcfd.fv.partition`, `zvcfd.fv.multi`)

Nodes are owned by whole store chunks along a Morton curve, cut into
pieces of nearly equal node count. Each partition holds every element
with an owned node, so its owned nodes' control volumes, flux points and
boundary sub-faces are complete; the other nodes of those elements are its
halo. Per outer iteration the halo receives the owners' nodal gradients,
limiter blends and momentum diagonals (after stage 1 of assembly), the new
state, and the new pressure gradient (before the mass-flow update).
Elements in several partitions compute identical mass flows from
identical inputs. The linear solve is one FGMRES over the distributed
vector (global dot products, halo exchange in every product), with a
block-Jacobi preconditioner: a V-cycle of ACM or AmgX on each partition's
owned block. The lumped outlets' rank-one terms are global (their `w · x`
sums over partitions). One partition is the single-GPU solver to
round-off; several reach the same solution to the solver tolerance, and
with High Resolution to the limiter's path dependence (10⁻⁴).

### Forces and balances

The force of the fluid on a zone comes from **consistent reactions**: the
residual of each fixed-velocity node's momentum equation before its
boundary row replaced it, which is exactly the boundary flux the discrete
equations need there. Each reaction component is shared among the zones
that fix that component (walls and inlets fix all three, a symmetry plane
only its normal), by sub-face area. Pressure zones are excluded, because
their pressure force and traction are already in the assembled rows.
Drag and lift therefore need no differentiation of the solution at the
wall. `balances()` checks, on every solve:

- **mass**: net outflow over gross flow (round-off, because outlet flows
  are consistent fluxes);
- **momentum**: Σ_zones (force + momentum outflow) − ∫f dV + momentum
  change, which is zero at convergence;
- **identity**: the same less the residual left in the solved rows, which is zero to
  round-off at any state. This checks the bookkeeping.

### Known properties

- The High Resolution limiter acts on Cartesian velocity components, as
  CFX's does, so results are not exactly rotation-invariant (8 × 10⁻⁴ on a
  test pipe; exactly invariant with a fixed blend).
- Freezing the limiter after 10 iterations makes High Resolution results
  depend slightly on the iteration path, for example on the false time step
  (7 × 10⁻⁵ in velocity on a test pipe). The linear solver changes the path
  as well: on the 9.5 k-node benchmark pipe at Re = 50, the flow rate with
  AmgX `gs` and with `robust` differs by 0.3 % (0.03 % at 72 k nodes).
  Fixed blends are path-independent to round-off (4 × 10⁻¹² there).
- Equal-order linear elements carry first-order pressure errors at
  boundary nodes (L2 pressure order 1.5–1.7 in the no-slip manufactured
  solution), as stabilised equal-order methods do. Velocity is second
  order.

### References

G. E. Schneider, M. J. Raw, Numer. Heat Transfer 11, 363 (1987);
C. M. Rhie, W. L. Chow, AIAA J. 21, 1525 (1983); S. K. Choi, Numer. Heat
Transfer B 36, 545 (1999); T. J. Barth, D. C. Jespersen, AIAA paper
89-0366 (1989); J. Bey, Computing 55, 355 (1995) (uniform refinement);
M. Esmaily Moghadam et al., Comput. Mech. 48, 277 (2011) (backflow
stabilisation); X. He, D. N. Ku, J. Biomech. Eng. 118, 74 (1996) (OSI);
F. M. White, *Viscous Fluid Flow*, 3rd ed. (developed duct flow).
