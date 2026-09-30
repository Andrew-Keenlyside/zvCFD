# Finite-volume numerics

**Status: in development.** This page specifies what
`zvcfd.fv.geometry`, the CPU reference solver `zvcfd.fv.reference` and the
GPU kernels `zvcfd.fv.gpu` implement today. The GPU kernels evaluate the
same discrete equations (gradients, momentum diagonal, residual), checked
against the reference to 3 × 10⁻¹⁶; the GPU linear solve is not built yet.
See the [plan](../feasibility/fv_plan.md).

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
| time | a false time step (pseudo-transient) for steady runs, or BDF1 / BDF2 (`c = 1, 1, 0` / `3/2, 2, −1/2`) with Picard coefficient loops per step, the first BDF2 step BDF1 |

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
| pressure | momentum solved; viscous flux zero-normal-gradient (only `μ (∇u)ᵀ · A`, lagged), or a prescribed traction from a given `∇u` (manufactured solutions); boundary momentum `ṁ_b u_node` in either direction (`backflow: consistent`, default) or out only | prescribed at the nodes and, for the pressure force, at sub-face centroids (`f(x)` or `f(x, t)`) | the consistent flux: what the interior sends to those nodes |

A node on several zones takes velocity from a wall first, then an inlet,
and pressure from an outlet. At symmetry-plane nodes, nodal gradients are
made mirror-symmetric: zero normal derivative for pressure and for the
tangential velocity components, zero tangential derivatives for the
normal component. Without that, one-sided gradients break the symmetry. With it,
half a channel with a symmetry plane reproduces the full channel to
round-off (`tests/test_fv_reference.py`). With no pressure zone, one node's pressure is
pinned. Because outlet mass flows are the consistent fluxes, inflow equals
outflow to round-off on every mesh.

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
  (7 × 10⁻⁵ in velocity on a test pipe). Fixed blends are path-independent
  to round-off.
- Equal-order linear elements carry first-order pressure errors at
  boundary nodes (L2 pressure order 1.5–1.7 in the no-slip manufactured
  solution), as stabilised equal-order methods do. Velocity is second
  order.

### References

G. E. Schneider, M. J. Raw, Numer. Heat Transfer 11, 363 (1987);
C. M. Rhie, W. L. Chow, AIAA J. 21, 1525 (1983); S. K. Choi, Numer. Heat
Transfer B 36, 545 (1999); T. J. Barth, D. C. Jespersen, AIAA paper
89-0366 (1989); J. Bey, Computing 55, 355 (1995) (uniform refinement).
