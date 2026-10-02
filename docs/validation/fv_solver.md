# Finite-volume solver

**Status (30 September 2026).** These cases verify and validate
`zvcfd.fv.reference`, the double-precision CPU reference solver, and the
GPU solver (`zvcfd.fv.solver`, its kernels `zvcfd.fv.gpu`, its linear
solvers `zvcfd.fv.linear`, and the partitioned form `zvcfd.fv.multi`). The
GPU solver is held to the reference solver itself, on the same discrete
equations, and to exact statements for the boundary conditions only it
has. The evidence is
organised on [Verification and validation](vv_plan.md); this page has the
numbers. See also the [plan](../feasibility/fv_plan.md) and
[numerics](../spec/fv_numerics.md).

The exact solutions, the manufactured solutions and the reference data
(Ghia et al. 1982; DFG 2D-1) are in `benchmarks/validation/exact.py` and
`benchmarks/validation/data/`, independent of the solver. The cases are in
`benchmarks/validation/fv_cases.py`, raw results in
`benchmarks/results/validation/fv/`, and fast versions with recorded
thresholds run in the test suite (`tests/test_fv_validation.py`,
`tests/test_fv_properties.py`, `tests/test_fv_gpu.py`,
`tests/test_fv_reference.py`).

Meshes come from `zvcfd.mesh.generate`: structured boxes of hexahedra,
wedges, tetrahedra or pyramids (optionally *warped*, a smooth mapping, or
*distorted*, interior nodes moved at random by up to 20 % of the local
edge); unstructured *Delaunay* tetrahedra of jittered lattices (slivers
included, minimum orthogonality 0.2°); butterfly O-grid tubes, with a
"mixed" variant built like the HiP-CT coronary mesh (wall wedges with
triangles parallel to the wall, around a tetrahedral core); the DFG
cylinder channel; and the FDA nozzle.

---

## Summary

| Case | Reference | Exercises | Result |
|---|---|---|---|
| Hydrostatics, Couette | exact | pressure gradient, viscous fluxes, boundary mass flows | round-off on every affine element |
| Manufactured, slip walls | Roache (2002) | every term, all element types | velocity **second order**; pressure 1.8–2.3 (large constant on tets and wedges) |
| Manufactured, no-slip walls | Roache (2002) | walls, all element types, Delaunay | velocity **second order**; pressure 1.5–1.7 (boundary pressure first order, the equal-order limit) |
| Manufactured, crossing a pressure outlet | Roache (2002) | outlets with traction, Carreau–Yasuda with full stress, warped meshes | velocity and pressure **second order** (final orders 1.82–2.07) |
| Hagen–Poiseuille pipe | exact | walls, velocity inlet, pressure outlet | **second order** in velocity, dp/dz, inlet pressure; mass exact |
| Kovasznay, Re = 40 | exact | steady Navier–Stokes | **second order**, 5 × 10⁻⁴ at n = 32 |
| Womersley pipe, α = 4 | exact | transient, BDF1/BDF2 | **second order** in space; BDF1 order 1, BDF2 order 2 |
| Ethier–Steinman | exact | unsteady 3-D Navier–Stokes | hex **second order** in space, BDF1/BDF2 orders 1/2 in time; tets 1.8 (sliding-velocity boundaries) |
| Invariances and balances | the transformed solution | rotation, reflection, renumbering, similarity, time-step independence | round-off (High Resolution limiter: 8 × 10⁻⁴ under rotation) |
| GPU kernels | the CPU reference | gradients, momentum diagonal, residual, block matrix, mass flows | **3 × 10⁻¹⁶** relative on every element type |
| GPU solver | the CPU reference | whole steady and transient solves, Carreau–Yasuda, symmetry, forces | **10⁻¹¹** with host-direct, ACM and AmgX linear solves |
| Flow-rate inlet; lumped outlets | exact: `ρ Q`; `p = a + r Q`; the standalone 0-D model | implicit RCR coupling, steady and transient, two stiff outlets | round-off; to the solver tolerance |
| Backflow stabilisation, openings, average pressure | the CPU reference; exact relations | pressure-zone options | 10⁻⁸ |
| Variable time step | exact for quadratics; the CPU reference | BDF2 at step ratios 0.6–1.7 | **order 1.9–2.0**; GPU = CPU to 10⁻⁹ |
| Wall shear stress | affine fields; Poiseuille; stagnation flow | velocity gradients at the wall integration points (CFX); consistent reactions as an option | exact for affine fields on every element type; first order in the wall cell, within 2 % with prism layers at R/32 ([numerics](../spec/fv_numerics.md#wall-shear-stress)); reactions close the force balance to 5 × 10⁻⁴ |
| Partitions | the one-GPU solver | halo exchange, global FGMRES | one partition: round-off; 2–4: 10⁻⁸ |
| DFG 2D-2 cylinder | Schäfer & Turek (1996) | vortex shedding, Re = 100 | *(results pending)* |
| DFG 2D-1 cylinder | Schäfer & Turek (1996) | drag, lift, Δp | c_D +0.23 %, Δp +0.35 % at 21,680 nodes |
| Lid-driven cavity, Re = 100, 400 | Ghia et al. (1982) | recirculation, moving wall | centreline RMS 0.003–0.007 at 64² |
| Developed pipe, pressure inflow | exact (Re-independent) | momentum entering through pressure boundaries | **converging** on extruded (+0.12 % at 18 cells across, order 1.2–1.5) and Delaunay tetrahedral meshes (−0.23 % at 24); a 0.5 % floor on structured tetrahedra; the opt-in full reconstruction unstable on coarse meshes |
| Same mesh as OpenFOAM | OpenFOAM v2506 | the whole solver; the Fluent writer | writer passes `checkMesh`; at Re = 50 zvCFD is 0.5–0.7 % below the exact flow rate and OpenFOAM 5.8–6.8 % (nc = 6, 12) |

---

## Exact solutions the scheme must reproduce

Two flows lie inside the discrete solution space, so the solver must get
them to round-off. Box of 5³ cells:

| Element | Distorted | Hydrostatic max \|u\| | Hydrostatic p error (of range) | Couette u error (of max) |
|---|---|---|---|---|
| tet | no | 4 × 10⁻¹⁶ | 4 × 10⁻¹⁴ | 2 × 10⁻¹⁵ |
| tet | yes | 1 × 10⁻¹⁶ | 9 × 10⁻¹⁵ | 2 × 10⁻¹⁵ |
| pyramid | no | 2 × 10⁻¹⁶ | 2 × 10⁻¹⁴ | 7 × 10⁻¹⁵ |
| pyramid | yes | 1.4 × 10⁻⁵ | 1.1 × 10⁻⁴ | 4.5 × 10⁻⁵ |
| wedge | no | 9 × 10⁻¹⁷ | 4 × 10⁻¹⁵ | 2 × 10⁻¹⁵ |
| wedge | yes | 3.4 × 10⁻⁶ | 2.0 × 10⁻⁵ | 1.4 × 10⁻⁵ |
| hex | no | 2 × 10⁻¹⁶ | 2 × 10⁻¹⁴ | 8 × 10⁻¹⁶ |
| hex | yes | 1.0 × 10⁻⁴ | 4.2 × 10⁻⁴ | 1.3 × 10⁻⁴ |

Tetrahedra are affine however they are distorted, so they stay exact.
Distorted hexahedra, wedges and pyramids are not affine. There, shape-
function gradients at flux points carry the usual isoparametric
consistency error, which is small but does not vanish.

---

## Manufactured solutions

### Slip walls

`exact.manufactured`: `u = (sin πx cos πy cos πz, cos πx sin πy cos πz,
−2 cos πx cos πy sin πz)`, `p = sin πx sin πy sin πz`, on the unit cube
with the exact velocity on every face and pressure pinned at the centre.
The body force makes it an exact solution (checked against finite
differences). Errors are relative L2 norms weighted by control volume.

#### Stokes (ρ = 1, μ = 1)

| Element | n | Velocity | order | Pressure | order | Pressure, interior nodes | order |
|---|---|---|---|---|---|---|---|
| hex | 4 | 5.55 × 10⁻² | | 0.453 | | 0.015 | |
| | 8 | 1.29 × 10⁻² | 2.11 | 0.215 | 1.07 | 0.084 | |
| | 16 | 3.12 × 10⁻³ | **2.05** | 0.043 | **2.31** | 0.028 | 1.62 |
| wedge | 4 | 6.62 × 10⁻² | | 3.60 | | 0.437 | |
| | 8 | 1.85 × 10⁻² | 1.84 | 1.30 | 1.47 | 0.337 | 0.38 |
| | 16 | 4.49 × 10⁻³ | **2.04** | 0.370 | **1.82** | 0.109 | 1.63 |
| tet | 4 | 8.46 × 10⁻² | | 9.37 | | 1.03 | |
| | 8 | 2.51 × 10⁻² | 1.76 | 3.14 | 1.58 | 0.542 | 0.93 |
| | 16 | 5.85 × 10⁻³ | **2.10** | 0.858 | **1.87** | 0.164 | 1.72 |
| pyramid | 4 | 5.04 × 10⁻² | | 0.420 | | 0.175 | |
| | 8 | 1.30 × 10⁻² | 1.96 | 0.133 | 1.67 | 0.082 | 1.09 |
| | 16 | 3.23 × 10⁻³ | **2.00** | 0.027 | **2.27** | 0.022 | 1.93 |

**Velocity is second order on every element type.** Pressure converges
at 1.8–2.3, but on tetrahedra and wedges its error starts large: it is
many times the pressure itself on the coarse grids. It sits at boundary
nodes, where velocity is prescribed and only the Rhie–Chow term
determines pressure. The error is smooth, not a checkerboard. On the
structured tetrahedral box it follows the direction in which the hexes
were cut. Three candidate causes were tested and ruled out:

- *iteration error* from the lagged Rhie–Chow gradient: the reference
  solver now takes that gradient implicitly, and nothing changed;
- *boundary mass flows* evaluated at nodes rather than at sub-face
  centroids: fixed, and exact for Couette flow, but this solution has
  `u · n = 0` on the boundary, so it did not change;
- *the Gauss form of the nodal pressure gradient* in the Rhie–Chow term:
  identical to the volume-averaged form at interior nodes of linear
  elements.

Dropping the `μ (∇u)ᵀ` part of the viscous stress, which integrates to
zero for constant viscosity, cuts the tetrahedral pressure error by a
third (9.4 against 13.9 at n = 4). The reference solver drops it for
Newtonian fluids. The rest is **open**, and the pipe below, with
realistic boundaries, does not show it.

#### Navier–Stokes (ρ = 1, μ = 0.1, Re ≈ 10)

| Element, advection | n | Velocity | order | Pressure | order | Picard iterations |
|---|---|---|---|---|---|---|
| hex, high-resolution | 4 | 9.23e-02 |  | 0.571 |  | 12 |
|  | 8 | 1.29e-02 | 2.84 | 0.207 | 1.47 | 14 |
|  | 16 | 2.58e-03 | **2.32** | 0.034 | 2.58 | 14 |
| hex, upwind | 4 | 1.06e-01 |  | 0.715 |  | 13 |
|  | 8 | 4.58e-02 | 1.21 | 0.403 | 0.83 | 12 |
|  | 16 | 2.75e-02 | **0.73** | 0.146 | 1.47 | 12 |
| tet, high-resolution | 4 | 1.10e-01 |  | 1.735 |  | 14 |
|  | 8 | 3.07e-02 | 1.84 | 0.524 | 1.73 | 22 |
|  | 16 | 6.42e-03 | **2.26** | 0.119 | 2.13 | 20 |
| wedge, high-resolution | 4 | 9.92e-02 |  | 0.924 |  | 12 |
|  | 8 | 1.93e-02 | 2.36 | 0.287 | 1.69 | 14 |
|  | 16 | 4.18e-03 | **2.21** | 0.058 | 2.30 | 18 |

**Second order in velocity with High Resolution** on hexahedra,
tetrahedra and wedges. Upwind is first order, as it must be. On
hexahedra the limiter is inactive in this smooth field: a fixed blend of
1 gives the same errors to three digits. On tetrahedra it acts, and
changes the velocity error by 3 % (3.07 × 10⁻² against 2.98 × 10⁻² at
n = 8). Picard converges to
10⁻¹¹ in 12–22 iterations, with the limiter frozen after 10.

### No-slip walls

`exact.manufactured_noslip`: `u = 100 ∇ × (0, φ, φ)`, `φ = g(x) g(y) g(z)`,
`g(s) = s² (1 − s)²`: zero on every face with its tangential derivatives,
as at a wall. `p = cos πx cos πy cos πz` is not zero there. Stokes flow, all
faces walls.

| Mesh | n | Velocity | order | Pressure | order | Boundary pressure (max, of range) | order |
|---|---|---|---|---|---|---|---|
| hex | 4 | 4.63 × 10⁻¹ |  | 0.560 |  | 0.264 |  |
|  | 8 | 9.28 × 10⁻² | 2.32 | 0.126 | 2.15 | 0.082 | 1.69 |
|  | 16 | 2.22 × 10⁻² | 2.06 | 0.045 | 1.47 | 0.044 | 0.91 |
| wedge | 4 | 3.53 × 10⁻¹ |  | 0.557 |  | 0.291 |  |
|  | 8 | 7.15 × 10⁻² | 2.30 | 0.137 | 2.02 | 0.103 | 1.49 |
|  | 16 | 1.72 × 10⁻² | 2.05 | 0.045 | 1.60 | 0.045 | 1.21 |
| tet | 4 | 2.07 × 10⁻¹ |  | 0.593 |  | 0.326 |  |
|  | 8 | 4.36 × 10⁻² | 2.25 | 0.154 | 1.94 | 0.102 | 1.68 |
|  | 16 | 1.02 × 10⁻² | 2.09 | 0.047 | 1.72 | 0.044 | 1.22 |
| pyramid | 4 | 2.55 × 10⁻¹ |  | 0.267 |  | 0.145 |  |
|  | 8 | 5.77 × 10⁻² | 2.14 | 0.097 | 1.47 | 0.072 | 1.01 |
|  | 16 | 1.42 × 10⁻² | 2.03 | 0.033 | 1.55 | 0.036 | 1.00 |
| Delaunay tet | 4 | 2.15 × 10⁻¹ |  | 0.573 |  | 0.314 |  |
|  | 8 | 6.43 × 10⁻² | 1.74 | 0.174 | 1.72 | 0.158 | 0.99 |
|  | 16 | 1.67 × 10⁻² | 1.95 | 0.056 | 1.63 | 0.065 | 1.27 |

With walls like the real case's, **every element type behaves the same**,
unstructured Delaunay tetrahedra included. Velocity is second order.
Pressure converges at 1.5–1.7 in L2 and about 1 in the maximum at boundary
nodes. That is the known limit of equal-order linear elements with
pressure stabilisation, where the formal pressure order is 1 and 1.5 is
typical. It is not a defect. The large tetrahedral constant of the slip-wall
case above appears only with prescribed *sliding* velocity on every face:
fixing the exact pressure at the boundary nodes there cuts the pressure
error 37-fold and the velocity error 5-fold (structured tets, n = 4). The
defect is characterised, and the real case's walls and inlets do not
trigger it.

### Crossing a pressure outlet, Newtonian and Carreau–Yasuda

`exact.manufactured_gn`: the slip-wall field shifted by (0.2, 0.1, 0.3),
plus a mean flow (1.5, 0, 0), so `u · n ≠ 0` on every face and flow leaves
through `xmax` everywhere. That face is a pressure outlet with the exact
pressure and viscous traction; the rest have prescribed velocity. The
Carreau–Yasuda case (μ₀ 0.5, μ∞ 0.05, λ 2, a 2, n 0.3568) keeps the full
stress `μ(γ̇)(∇u + ∇uᵀ)`. Its source term comes from complex-step
derivatives of the analytic stress, exact to round-off. Meshes are warped
by 3 %, so distortion shrinks with refinement.

| Mesh, fluid | n | Velocity | order | Pressure | order | Picard |
|---|---|---|---|---|---|---|
| hex, newtonian | 4 | 3.22 × 10⁻² |  | 0.756 |  | 17 |
|  | 8 | 9.27 × 10⁻³ | 1.80 | 0.247 | 1.61 | 21 |
|  | 16 | 2.38 × 10⁻³ | 1.96 | 0.063 | 1.97 | 19 |
| hex, carreau-yasuda | 4 | 2.57 × 10⁻² |  | 0.898 |  | 17 |
|  | 8 | 8.90 × 10⁻³ | 1.53 | 0.327 | 1.46 | 22 |
|  | 16 | 2.52 × 10⁻³ | 1.82 | 0.079 | 2.05 | 25 |
| tet, newtonian | 4 | 5.92 × 10⁻² |  | 1.524 |  | 19 |
|  | 8 | 1.77 × 10⁻² | 1.74 | 0.527 | 1.53 | 21 |
|  | 16 | 4.54 × 10⁻³ | 1.97 | 0.125 | 2.07 | 22 |
| tet, carreau-yasuda | 4 | 5.06 × 10⁻² |  | 1.778 |  | 18 |
|  | 8 | 1.57 × 10⁻² | 1.69 | 0.676 | 1.40 | 23 |
|  | 16 | 4.22 × 10⁻³ | 1.90 | 0.162 | 2.06 | 25 |

**Second order in velocity and pressure** for both fluids and both element
types, with mass conserved to round-off: outlets, traction, variable
viscosity and the transpose stress are all verified.

---

## Hagen–Poiseuille pipe

Radius 0.5, length 3 (six diameters), μ = 1: a parabolic inlet profile
with centreline speed 2, a pressure outlet at p = 0, and no-slip walls.
The tube's wall is a polygon through `4 nc` wall nodes, and the errors
are measured against the exact solution in the true circle. They
therefore include the polygon's geometric error, which is also second
order.

| Mesh | nc | Nodes | Mid-pipe velocity (L2) | order | dp/dz error | order | Inlet pressure error | order | Mass imbalance |
|---|---|---|---|---|---|---|---|---|---|
| hex | 2 | 325 | 1.47e-02 |  | +16.61 % |  | +16.76 % |  | -2e-15 |
|  | 4 | 2,225 | 3.10e-03 | 2.24 | +3.84 % | 2.11 | +3.91 % | 2.10 | -4e-15 |
|  | 8 | 16,513 | 7.29e-04 | 2.09 | +0.94 % | 2.03 | +0.97 % | 2.01 | 1e-14 |
| tet | 2 | 325 | 1.70e-02 |  | +14.78 % |  | +15.18 % |  | -8e-15 |
|  | 4 | 2,225 | 3.96e-03 | 2.11 | +3.48 % | 2.09 | +3.58 % | 2.08 | -1e-14 |
|  | 8 | 16,513 | 9.34e-04 | 2.08 | +0.86 % | 2.01 | +0.89 % | 2.01 | 1e-14 |
| wall wedges + tet core | 2 | 325 | 1.80e-02 |  | +15.91 % |  | +16.21 % |  | -2e-15 |
|  | 4 | 2,225 | 3.99e-03 | 2.18 | +3.73 % | 2.09 | +3.81 % | 2.09 | -1e-14 |
|  | 8 | 16,513 | 9.68e-04 | 2.04 | +0.92 % | 2.02 | +0.95 % | 2.01 | -8e-15 |

**Second order in velocity, pressure gradient and inlet pressure on all
three meshes**, including the inlet, where velocity is prescribed. The
pressure errors at nc = 8 are 0.86–0.97 %. At that size the wall polygon has 32 sides and
0.64 % less cross-section than the circle, so much of what remains is
geometry. **Inflow equals outflow to round-off** on every mesh, because
outlet flows are the consistent fluxes. The mixed mesh, built like the
coronary mesh, is as accurate as the pure ones.

---

## Developed pipe flow through pressure boundaries

Radius 0.5, length 4, ρ = 1, μ = 0.02, fixed pressure 2.56 at the inlet and
0 at the outlet, natural velocity at both ends, no-slip walls: Re = 50 on
the developed mean velocity. Developed Poiseuille flow satisfies the
Navier–Stokes equations and both boundary conditions, so the flow rate does
not depend on Re. Its exact value is Poiseuille's on the wall polygon
(`exact.polygon_poiseuille_flux`). The advective error is the flow rate
against the discrete Stokes solution on the same mesh, which converges to
the exact rate at second order. The case tests momentum entering through a
pressure boundary, where the inflow profile is free
([numerics](../spec/fv_numerics.md)).

Three mesh families, all with the same wall polygon of `4 nc` sides and
`8 nc` layers:

- **extruded**: triangles extruded along the axis, rings graded toward the
  wall, so every edge is axial or in a cross-section;
- **mixed**: wedge layers graded toward the wall around a structured
  tetrahedral core, built like the coronary mesh. This is the mesh of the
  OpenFOAM comparison below. Its tetrahedra come from splitting hexahedra,
  so their diagonals all lean the same way;
- **Delaunay**: the Delaunay tetrahedralisation of a jittered lattice in the
  same prism, with no wall layers. Its connectivity is unstructured, as in
  vessel meshes.

Flow rate against Stokes at Re = 50, with outer iterations to 10⁻⁹ in brackets:

| Mesh | nc | Nodes | High Resolution (default) | Blend 1 | Upwind | Second-order reconstruction (opt-in) |
|---|---:|---:|---:|---:|---:|---:|
| extruded | 6 | 9,457 | +0.560 % (36) | +0.560 % | 0 (round-off) | diverges |
| | 12 | 72,265 | +0.204 % (32) | +0.204 % | 0 | −0.026 % (107) |
| | 18 | 240,265 | +0.124 % (31) | +0.124 % | 0 | +0.003 % (57) |
| mixed | 3 | 1,300 | +0.60 % (42) | +2.23 % | −21.6 % | −0.52 % (not converged in 400) |
| | 6 | 9,457 | +0.08 % (34) | +0.14 % | −14.9 % | +0.25 % (stalls at 10⁻¹⁰) |
| | 12 | 72,265 | −0.44 % (32) | −0.44 % | −14.5 % | +0.51 % (124) |
| | 18 | 240,265 | −0.58 % (33) | −0.57 % | −15.3 % | +0.43 % (69) |
| Delaunay | 6 | 3,035 | −7.84 % (41) | −5.73 % | −49.6 % | −8.49 % (90) |
| | 12 | 22,567 | −0.59 % (37) | −0.58 % | −34.7 % | −0.84 % (46) |
| | 18 | 72,624 | −0.46 % (36) | −0.47 % | −28.1 % | −0.24 % (38) |
| | 24 | 168,050 | −0.23 % (30) | −0.22 % | −23.9 % | −0.14 % (32) |

**On the extruded mesh the default is consistent and converges**, at an
observed order of 1.46 (nc 6 → 12) and 1.23 (12 → 18). Against the exact
rate it is within 0.19 %, 0.015 % and 0.006 %. Upwind is exact there to
round-off, because every edge is either along the developed flow or across
it. The inflow shows no jet: the centreline speed at the inlet is within
10⁻⁴ of mid-pipe.

**On the Delaunay mesh it converges too:** −0.59, −0.46 and −0.23 % with
12, 18 and 24 cells across, an observed order of 1.4 over that range (the
random meshes make the step-to-step order noisy). Against the exact rate:
−1.05, −0.66 and −0.35 %. The reconstruction converges as well. Upwind
converges there, slowly (order 0.5).

**On the mixed mesh about 0.5 % does not shrink**, for every second-order
treatment of the inflow: −0.44 % and −0.58 % for the default at 12 and 18
cells across, +0.51 % and +0.43 % for the full reconstruction. Against the
exact rate the default stays at −0.66 to −0.70 % from 6 to 18 cells across.
The floor comes from the mesh's structured tetrahedra, not from the inflow
treatment: the Delaunay mesh has none. Randomly moving the mixed mesh's
interior nodes by up to 20 % of the local edge does not remove it (−0.09,
−0.54, −0.64 %), because the split pattern stays. Upwind shows the same bias
at full strength. On the developed flow its advection residual stays at
0.4 ρU²/R RMS at every level, and its flow rate at −15 %. The second-order
schemes reduce that residual to 0.01–0.03 ρU²/R, where the square core meets
the rings, but it does not shrink.

How the inflow treatment was chosen, with the flow rate against Stokes:

| Inflow treatment | extruded, nc 6 | 12 | mixed, nc 6 | 12 | 18 | stable |
|---|---:|---:|---:|---:|---:|---|
| corrections on faces leaving an inflow node, none at its boundary face (before 1 October) | not converged | — | +3.4 % | +3.8 % | — | yes |
| no correction at inflow nodes (`β = 0`) | +1.30 % | +1.25 % | −4.09 % | −2.93 % | −2.36 % | yes |
| corrections out of the inflow node's own row, carried by its boundary flux | +0.56 % | +0.20 % | −1.71 % | −1.46 % | −1.33 % | yes |
| **the same, plus the cross-stream part of the faces' offset from the boundary (default)** | **+0.56 %** | **+0.20 %** | **+0.08 %** | **−0.44 %** | **−0.58 %** | **yes** |
| full second-order boundary reconstruction | diverges | −0.03 % | +0.25 % | +0.51 % | +0.43 % | no, on coarse meshes |

- **Dropping the corrections only moved the inconsistency one row in.**
  The inflow from the boundary nodes then arrived uncorrected and left
  corrected, which reversed the jet.
- **The full reconstruction keeps the part of the correction along the
  boundary normal.** The faces leaving an inflow node lie half a cell
  downstream, and its gradient there is one-sided, so the half-cell is
  differenced centrally. That is unstable at cell Péclet numbers above
  about 2: it diverges on the extruded mesh at 6 cells across (Péclet about
  8) and converges at 12 and 18. Developed flow has no gradient along the
  normal, so the default keeps only the cross-stream part.
- **On the Delaunay mesh the mirrored treatment alone gives −0.55, −0.54
  and −0.18 %** at 12, 18 and 24 cells across. The cross-stream term
  matters where the tetrahedra are structured: it takes the mixed mesh from
  −1.5 % to about −0.5 %.

GPU (fused and classic kernels) and CPU solvers agree on the default to
2 × 10⁻¹¹ with a fixed blend, and the CPU reference's momentum balance
closes to 10⁻¹². Tests: `tests/test_fv_developed_pipe.py`. Script:
`benchmarks/validation/fv_gpu_cases.py developed_pipe [mixed|wedge|delaunay]`.
Results: `benchmarks/results/validation/fv/developed_pipe{,_wedge,_delaunay}.json`
(the perturbed mixed mesh and the mirrored treatment alone were one-off
runs on 2 October and are not in the results files).

---

## Exact Navier–Stokes flows

### Kovasznay flow (steady, Re = 40)

On [−0.5, 1] × [−0.5, 1.5], one cell deep, with symmetry planes on the z
faces and the exact velocity elsewhere.

| Mesh | n (cells across y / 2) | Nodes | Velocity | order | Pressure | order |
|---|---|---|---|---|---|---|
| hex | 8 | 442 | 5.99 × 10⁻³ |  | 4.80 × 10⁻² |  |
|  | 16 | 1,650 | 1.60 × 10⁻³ | 1.90 | 1.07 × 10⁻² | 2.17 |
|  | 32 | 6,370 | 4.75 × 10⁻⁴ | 1.75 | 2.76 × 10⁻³ | 1.95 |
| wedge | 8 | 442 | 3.85 × 10⁻² |  | 1.08 × 10⁻¹ |  |
|  | 16 | 1,650 | 8.14 × 10⁻³ | 2.24 | 2.55 × 10⁻² | 2.08 |
|  | 32 | 6,370 | 1.26 × 10⁻³ | 2.69 | 4.74 × 10⁻³ | 2.42 |
| tet | 8 | 442 | 3.12 × 10⁻² |  | 1.03 × 10⁻¹ |  |
|  | 16 | 1,650 | 6.44 × 10⁻³ | 2.28 | 2.36 × 10⁻² | 2.12 |
|  | 32 | 6,370 | 1.01 × 10⁻³ | 2.67 | 4.96 × 10⁻³ | 2.25 |

### Womersley flow (pulsatile, α = 4)

A pipe driven by an oscillating pressure difference `L cos 2πt` between
fixed-pressure ends (natural velocity conditions at both). The flow is
fully developed, so two axial cells suffice. Started from the exact
state; Navier–Stokes for the spatial study, Stokes (identical here) for
the temporal one.

| nc | Nodes | Velocity at t = T (L2) | order | Coefficient loops per step | Mass, momentum balance |
|---|---|---|---|---|---|
| 2 | 75 | 8.26 × 10⁻² |  | 4.9 | 2e-16, 9e-17 |
| 4 | 267 | 2.18 × 10⁻² | 1.92 | 4.9 | 2e-16, 8e-16 |
| 8 | 1,011 | 5.39 × 10⁻³ | 2.01 | 4.7 | 2e-16, 3e-16 |

| Scheme | steps per period | Error against dt = T/800 | order |
|---|---|---|---|
| BDF1 | 25 | 1.81 × 10⁻¹ |  |
|  | 50 | 9.10 × 10⁻² | 1.00 |
|  | 100 | 4.33 × 10⁻² | 1.07 |
| BDF2 | 25 | 2.61 × 10⁻² |  |
|  | 50 | 7.47 × 10⁻³ | 1.81 |
|  | 100 | 1.95 × 10⁻³ | 1.94 |

**Second order in space, and BDF1 first and BDF2 second order in time**,
with mass and momentum balances at round-off, the time term included.
Getting there needed a fix: pressure boundaries dropped inflowing
momentum, which oscillating flow exposes at both ends every half-cycle
(see defects).

### Ethier–Steinman flow (unsteady, fully 3-D)

The exact solution of Ethier & Steinman (1994) on [−1, 1]³ (a = π/4,
d = π/2, ν = 1), exact velocity on every face, pressure pinned to its exact
value, to t = 0.1.

| Mesh | n | Nodes | Velocity at t = 0.1 | order | Pressure | order | loops per step |
|---|---|---|---|---|---|---|---|
| hex | 4 | 125 | 2.60 × 10⁻² |  | 6.12 × 10⁻¹ |  | 5.1 |
|  | 8 | 729 | 6.02 × 10⁻³ | 2.11 | 1.50 × 10⁻¹ | 2.03 | 6.0 |
|  | 16 | 4,913 | 1.14 × 10⁻³ | 2.40 | 3.48 × 10⁻² | 2.11 | 6.0 |
| tet | 4 | 125 | 4.45 × 10⁻² |  | 9.29 × 10⁻¹ |  | 5.8 |
|  | 8 | 729 | 1.25 × 10⁻² | 1.83 | 3.97 × 10⁻¹ | 1.23 | 6.0 |
|  | 16 | 4,913 | 3.69 × 10⁻³ | 1.76 | 1.58 × 10⁻¹ | 1.33 | 6.0 |

| Scheme | steps to t = 0.1 | Error against dt = T/160 | order |
|---|---|---|---|
| BDF1 | 5 | 8.26 × 10⁻⁴ |  |
|  | 10 | 4.11 × 10⁻⁴ | 1.01 |
|  | 20 | 1.95 × 10⁻⁴ | 1.08 |
| BDF2 | 5 | 1.33 × 10⁻⁴ |  |
|  | 10 | 2.68 × 10⁻⁵ | 2.31 |
|  | 20 | 6.05 × 10⁻⁶ | 2.15 |

On hexahedra, **second order in velocity and pressure**. BDF1 converges at
first order and BDF2 at second in time, with all nonlinear terms active.
Tetrahedra reach 1.8 in velocity and 1.2–1.3 in pressure. This solution
prescribes non-zero tangential velocity on every face, the sliding-velocity
case where structured tetrahedral boundary pressure is poor (see
[manufactured solutions](#no-slip-walls)). The coefficient loops were
capped at six per step, short of the 10⁻¹¹ tolerance at the finest levels.

---

## Invariances and balances

On an unstructured (Delaunay) pipe with an inlet, a pressure outlet and
walls, Navier–Stokes at Re = 20:

| Transformation | max velocity change (relative) | max pressure change (of range) |
|---|---|---|
| random rotation + translation, fixed blend | 3.0e-15 | 1.0e-14 |
| reflection + translation, fixed blend | 3.0e-15 | 1.3e-14 |
| rotation, High Resolution | 7.6e-04 | 6.6e-04 |
| reflection, High Resolution | 8.0e-04 | 7.3e-04 |
| random node renumbering | 6.7e-15 | 8.9e-15 |
| lengths × 3, speed × 0.2, viscosity × 0.6 (Re fixed) | 8.4e-15 | 1.4e-14 |
| false time step 0.05 against none | 6.8e-13 | 3.7e-14 |
| false time step 0.5 against none | 1.1e-13 | 1.3e-14 |

Balances on the same pipe (Navier–Stokes): mass 6e-16, momentum 3e-16, bookkeeping identity 4e-16.

Exact invariance holds for every fixed-blend run. High Resolution departs
by 8 × 10⁻⁴ because its limiter works on Cartesian velocity components, as
CFX's does. The steady state does not depend on the false time step, and
marching a transient to steady state reproduces the steady solve to
10⁻¹³. That needed the transient Rhie–Chow term, with its coefficients
built from per-node ratios (see defects).

**Lagged Rhie–Chow gradient.** The GPU solver will take the interpolated
nodal pressure gradient from the previous outer iteration, as CFX does,
instead of implicitly. On a mixed pipe the lagged form converges to the
implicit solution to 10⁻¹²
(`tests/test_fv_reference.py`).
It takes 61 outer iterations against 2 for Stokes flow, and 58 against 16
for Navier–Stokes. That cost is inside the 50–150 outer iterations a
CFX-style steady solve takes anyway.

---

## GPU kernels against the reference

The CUDA kernels (`zvcfd.fv._cuda`) are written from the specification,
not ported from the CPU code. On random states (velocities, pressures,
lagged mass flows, limiter blends) and distorted meshes of every type,
the nodal gradients, the momentum diagonal and the full ip-flux residual
must match the CPU reference (`tests/test_fv_gpu.py`, 26 cases with and
without advection and the transpose stress):

| Mesh | Nodes | Colours | Residual (max, relative) | Gradients (max, relative) |
|---|---|---|---|---|
| delaunay | 64 | 36 | 4.7e-16 | 2.8e-15 |
| hex | 64 | 8 | 2.8e-16 | 7.6e-16 |
| mixed | 100 | 34 | 4.5e-16 | 7.1e-16 |
| pyramid | 35 | 24 | 1.7e-16 | 6.0e-16 |
| tet | 64 | 28 | 2.3e-16 | 6.2e-16 |
| wedge | 64 | 13 | 3.2e-16 | 8.6e-16 |

Agreement is at machine precision. Element colouring (no two elements of
a colour share a node) keeps the kernels free of atomics, and repeated runs
are bit-identical. The block matrix, right-hand side, limiter and mass
flows of the GPU assembly match the reference's lagged form to 10⁻¹¹ of
their scale, with and without a time term and for Carreau–Yasuda fluid.

---

## The GPU solver against the reference

The GPU solver lags the Rhie–Chow gradient and solves each linear system
only to a tenfold residual reduction, as CFX does. At convergence it must
reach the reference's discrete solution (`tests/test_fv_solver_gpu.py`):

| Case | Linear solver | Agreement with the reference |
|---|---|---|
| Mixed tet–wedge pipe, fixed blend | host direct, ACM, AmgX | `u` 10⁻⁹, `p` 10⁻⁹ of its range; mass 10⁻¹² |
| Same, High Resolution (reference lagged too, same path) | host direct | 10⁻⁸ |
| Hexahedral half channel, symmetry planes, false time step | ACM | 10⁻⁹ |
| Carreau–Yasuda manufactured solution crossing an outlet with traction | host direct | 10⁻⁸ |
| Womersley pipe, BDF2 | host direct | 10⁻⁹ |
| Womersley pipe, variable steps (ratios 0.5–1.6) | host direct | 10⁻⁹ |
| Zone forces (walls, inlet, outlet) from consistent reactions | host direct | 10⁻⁸ of the largest |

With the pressure level initialised from the boundaries, shifting every
pressure by 1000 changes the GPU velocities by 7 × 10⁻¹³, as it does the
reference's. Gate B timings are on the [plan](../feasibility/fv_plan.md#progress).

---

## Boundary conditions of the GPU solver

Each is checked against an exact statement (`tests/test_fv_boundaries.py`):

| Check | Exact statement | Result |
|---|---|---|
| Developed profile of a circular cap | `1 − (r/R)²` | 0.03 at the nodes of a 6 × 6 O-grid face |
| Flow-rate inlet, tabulated `Q(t)` | discrete inflow `= ρ Q(t)` | 10⁻¹³ relative |
| Steady RCR outlet (host direct, ACM, AmgX) | `p_out = P_v + (R_p + R_d) Q`; the field equals the fixed-pressure solution at that pressure | 10⁻⁹ |
| Transient RCR outlet, sinusoidal inflow | outflow = inflow; `p_out(t)` = the standalone RCR driven by that outflow | 10⁻⁹ |
| Two outlets at 1000× and 2000× the domain's resistance (ACM, AmgX) | `p_i = r_i Q_i`; `Q_a/Q_b → 2` | 10⁻⁸; 2.000 ± 0.002, in < 80 outer iterations |
| Backflow stabilisation (β = 0.5), outlet with backflow | GPU = CPU reference; momentum identity | 10⁻⁸; 10⁻¹⁰ |
| Average static pressure | area mean = the value; ≈ the fixed-pressure drop in developed flow | 10⁻¹⁰; within 5 % |
| Opening driven by total pressure | `p = p₀ − ½ρ|u|²` at inflow nodes; mass balance | 10⁻⁹; 10⁻¹⁰ |
| Wall shear in developed pipe flow, reactions | `τ P = −(dp/dz) A`; Poiseuille `4μQ/(πR³)` | 5 × 10⁻⁴; 5.6 % on a 16-sided section, 1.5 % on 32 (the section's area deficit, `A/πR²` = 0.974, 0.994) |
| Wall shear, wall-ip gradients (default) | affine velocity on perturbed and warped tet, wedge, hex, pyramid meshes; Poiseuille against the section's force balance | 10⁻¹²; 7.4 → 3.2 → 1.2 % as the wall cell shrinks (first order) |
| BDF2, variable steps (ratios 0.6 and 1.67 alternating), Womersley | observed temporal order | 1.9–2.0 |
| Time-step study helper on BDF2 decay | order 2; the extrapolate beats the fine run | 2.00 |

## Linear solvers on thin cells

`tests/test_fv_linear.py` holds the linear solvers to a direct solve on
systems where point-block smoothing struggles: the DFG slab (one cell
deep, Δt = 0.005 s) and a tube with strongly graded wall layers. The
default AmgX configuration (`gs`: symmetric multicolour Gauss–Seidel,
aggregation weighted on the pressure entry) needs 17 iterations to 10⁻⁶
on each, in double and mixed precision alike. The `robust` preset (ILU(0), pairwise
aggregation) needs 11 on the slab, where the usual DILU set-up stalls. The
SIMPLE block preconditioner solves the layered tube, and the `auto` ladder
switches to a working solver when one fails (`gs` → `robust` → DILU →
SIMPLE, checked with a 3-iteration limit that no rung can meet). The numbers and the reason are
on [numerics](../spec/fv_numerics.md#linear-solvers-on-thin-cells-zvcfdfvlinear).

## Partitions

`tests/test_fv_multi.py` runs the partitioned solver as 1–4 partitions of
one GPU. After the halo exchange, nodal gradients equal the global ones
at every local node to 10⁻¹²; one partition reproduces the one-GPU ACM
solver's iteration history to 10⁻⁹; with two to four, the converged
fields match it to 10⁻⁸ (fixed blend), to 10⁻³ with High Resolution (the
frozen limiter's path dependence), for two lumped outlets and for a
transient with variable steps, whose wall shear matches to 10⁻⁸.

---

## Benchmarks against published data

### DFG 2D-1: cylinder in a channel (Re = 20)

Quasi-2-D (one cell deep, symmetry planes), parabolic inflow U_max = 0.3,
ν = 10⁻³, butterfly O-grid around the cylinder. Forces come from consistent
reactions.

| m | Nodes | c_D | error | c_L | error | Δp | error | Picard |
|---|---|---|---|---|---|---|---|---|
| 8 | 2,080 | 5.8069 | +4.07 % | 0.00986 | -7.1 % | 0.12274 | +4.44 % | 32 |
| 16 | 6,698 | 5.6216 | +0.75 % | 0.01142 | +7.5 % | 0.11913 | +1.37 % | 25 |
| 32 | 21,680 | 5.5922 | +0.23 % | 0.01079 | +1.7 % | 0.11794 | +0.35 % | 25 |
| reference | | 5.5795 | | 0.01062 | | 0.11752 | | |
| interval | | 5.57–5.59 | | 0.0104–0.011 | | 0.1172–0.1176 | | |

Drag and pressure difference converge at order 1.7–2.4 and reach the
published intervals' edges at m = 32 (c_D 0.04 % above the upper bound,
Δp 0.3 %). Richardson extrapolation of c_D gives 5.587, 0.13 % from the
reference. Lift is small and converges non-monotonically, as in most
codes. An early run gave −7.7 % drag at m = 16: the cylinder nodes' x and
y reactions were being shared with the symmetry planes, which fix only z
(see defects).

### DFG 2D-2: vortex shedding (Re = 100)

The same channel at `U_max = 1.5` m/s: periodic shedding. GPU solver, High
Resolution, BDF2 at Δt = 0.005 s, from the inflow profile with a small
asymmetric kick in the wake, marched to 6 s; the maximum drag and lift
coefficients over the last periods, the Strouhal number from the lift
period, and Δp half a period after maximum lift, against the benchmark's
intervals (`benchmarks/validation/fv_gpu_cases.py`):

*(results pending)*

### Lid-driven cavity (Re = 100, 400)

One cell deep with symmetry planes; the lid is a velocity zone, and its
corner nodes take the side walls' no-slip. Deviations from Ghia et al.'s
centreline tables (u on x = 0.5, v on y = 0.5), with their probable typo
at Re = 400, x = 0.9063 excluded:

| Re | n | u RMS dev | u max dev | v RMS dev | v max dev | Picard |
|---|---|---|---|---|---|---|
| 100 | 16 | 0.0069 | 0.0171 | 0.0103 | 0.0187 | 23 |
|  | 32 | 0.0031 | 0.0076 | 0.0060 | 0.0099 | 20 |
|  | 64 | 0.0026 | 0.0059 | 0.0054 | 0.0101 | 19 |
| 400 | 16 | 0.0565 | 0.0925 | 0.0754 | 0.1565 | 37 |
|  | 32 | 0.0139 | 0.0245 | 0.0226 | 0.0449 | 44 |
|  | 64 | 0.0034 | 0.0066 | 0.0071 | 0.0126 | 42 |

At Re = 100 the deviation settles at about 0.003–0.005, the precision of
Ghia's own 129² solution. At Re = 400 it is still falling at 64².

---

## Cross-code: the same mesh in OpenFOAM

A wall-wedge / tet-core pipe written by zvCFD's Fluent writer and imported
with `fluent3DMeshToFoam` (OpenFOAM v2506, official image). Pressure-driven
flow at Re = 50, with fixed pressure at both ends and natural velocity there,
in both codes. OpenFOAM is cell-centred (`simpleFoam`, `linearUpwind`),
zvCFD vertex-centred (High Resolution). It is the mixed mesh of the
[developed pipe](#developed-pipe-flow-through-pressure-boundaries), whose
exact flow rate is developed Poiseuille flow on the polygonal section.

zvCFD measured on 2 October, after the fix to momentum entering through
pressure boundaries; OpenFOAM on 1 October, after the fix to the tube
generator (its `growth` had graded the wall layers toward the core).

| nc | Nodes | Cells | zvCFD Q | OpenFOAM Q | difference | exact Q | zvCFD vs exact | OpenFOAM vs exact | checkMesh |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 1,300 | 5,328 | 0.69746 (CPU reference) | 0.65988 | +5.7 % | 0.71267 | −2.1 % | −7.4 % | Mesh OK |
| 6 | 9,457 | 38,016 | 0.76175 (CPU reference) | 0.71498 | +6.5 % | 0.76711 | −0.70 % | −6.8 % | Mesh OK |
| 12 | 72,265 | 304,128 | 0.77668 (GPU) | 0.73530 | +5.6 % | 0.78085 | −0.53 % | −5.8 % | Mesh OK |

OpenFOAM's `checkMesh` passes every mesh the writer produces: an
independent check of the Fluent export. "Exact" is developed Poiseuille
flow on the wall polygon (`4 nc` sides), from a converged finite-element
solution of `−Δw = 1` on it (`exact.polygon_poiseuille_flux`, which
reproduces the square-duct series to 2 × 10⁻⁶): 0.9074, 0.97671 and
0.99421 of the circle's 0.78540. That is not the circle's rate times the
area ratio squared, which holds only for similar sections (0.9119, 0.9774,
0.9943). zvCFD's Stokes solutions converge to it at second order: −0.77,
−0.22 and −0.12 % at nc = 6, 12 and 18.

**Both codes lie below the exact rate and converge toward it, and zvCFD is
much the closer: −0.5 to −0.7 % at nc = 6 and 12, against OpenFOAM's −5.8
to −6.8 %.** The gap between them is now almost all OpenFOAM's
discretisation error. zvCFD's remainder is the Stokes solution's own error
(0.2–0.8 %) and this mesh's structured-tetrahedron floor of about 0.5 %
(see the developed pipe). The GPU value at nc = 12 differs from the
developed-pipe run's by 0.13 %, because the two use different linear solvers
(`auto` here, AmgX `gs` there). High Resolution freezes its limiter after
10 iterations, so the answer depends slightly on the iteration path.

Before 2 October the codes differed by 10 % at nc = 6 and 12, with zvCFD
3.4–3.8 % *above* the exact rate and an inflow jet 14–28 % faster on the
axis than Poiseuille. The cause was the deferred High Resolution correction
on the faces leaving inflow-boundary nodes. Momentum entering through the
pressure boundary carried no counterpart, so in the first row of control
volumes the correction moved momentum toward the faster fluid. Switching it
off at the inlet nodes alone removed the excess; switching it off at outlet
or wall nodes changed nothing. Both zvCFD solvers showed it, so it was in
the shared discretisation, not in the speed work. On the earlier,
wall-coarse meshes, a scan in Reynolds number had shown the codes agreeing
at Re = 0.5 and 10 (−0.24 % and +0.07 % at nc = 6) and parting at Re = 50
(`benchmarks/results/fv/openfoam_same_mesh_re_scan.json`), as an advective
inflow error would.

---

## Solution verification on a fixed mesh

The coronary tree cannot be remeshed, so its grid study will refine the
given mesh. `zvcfd.mesh.refine` splits every element into eight
(conformingly, keeping the faceted geometry), and `zvcfd.verification`
computes the observed order, the Richardson extrapolate and the grid
convergence index (Roache; Celik et al. 2008). Demonstrated on a
wall-wedge / tet-core pipe, pressure-driven Stokes flow:

| Level | Nodes | Elements | Flow rate | Centreline u | Wall force − Δp × area |
|---|---|---|---|---|---|
| 0 | 175 | 528 | 0.006145 | 0.018665 | 0e+00 |
| 1 | 1,157 | 4,224 | 0.006372 | 0.018554 | -6e-16 |
| 2 | 8,425 | 33,792 | 0.006475 | 0.018567 | -3e-15 |

Flow rate: observed order 1.15, extrapolated 0.006559, GCI (fine) 1.6 %, asymptotic ratio 1.016. Centreline velocity: oscillatory convergence (flagged), GCI 0.012 %.

The flow rate is in the asymptotic range (the ratio test is 1.02), with a
1.6 % fine-grid GCI. Its observed order, 1.15, is below the formal 2
because the sequence starts very coarse. A hexahedral square duct refined
the same way gives 1.79, with the extrapolate within 0.5 % of White's
exact series. The wall force equals Δp × inlet area to round-off at every
level, as momentum conservation requires in Stokes flow.

---

## Prepared, not run

- **The FDA benchmark nozzle** (`generate.fda_nozzle`: 12 mm pipe, 20°
  cone, 4 mm × 40 mm throat, sudden expansion; FDA dimensions). The default
  mesh has 46,000 nodes, beyond the direct-solver reference, so it waits for
  the GPU solver. The multi-laboratory PIV data (Hariharan et al. 2011) were
  published on the FDA's hub, `nciphub.org`, which no longer resolves; a
  copy is needed.
- **The coronary tree**: the same cases on the real mesh (OpenFOAM, CFX,
  GCI by refinement, the time-step study) wait for the H100 run: its
  106 M blocks need about 25–39 GB.

---

## Defects the validation found

| Found by | Defect | Symptom | Fix |
|---|---|---|---|
| Hydrostatic case | the lagged Rhie–Chow gradient converges slowly for smooth pressure (≈ 0.7 per iteration) | pressure not converged after 50 iterations | the reference solver takes it implicitly; the GPU solver lags it, as CFX does, and spends outer iterations on it |
| Pyramid gradient check | integration points placed by averaging parametric points of the collapsed-hex pyramid | 8 % gradient error on undistorted pyramids | place each point physically on the reference element, then map back |
| Couette flow | boundary mass flows evaluated with each node's own velocity | velocity error 3.7 % of the maximum, spurious pressures up to 7–10 where the exact pressure is zero | interpolate on the face to each sub-face's area centroid (11/18, 7/36 on triangles; 9/16, 3/16, 1/16 on quadrilaterals) |
| Couette flow, tetrahedra | one flux point per non-planar ip face | boundary continuity error 6 × 10⁻⁵ on undistorted tetrahedra | two flux points per ip face, one per planar half (a deviation from the plan) |
| VTK round trip | wedge node order assumed opposite to VTK's | negative VTK cell volumes | VTK's wedge order is zvCFD's; no swap |
| Symmetry-plane test | outflowing momentum at pressure outlets indexed per node against per sub-face arrays | crash in the first Navier–Stokes outlet run (every earlier outlet run was Stokes) | index by the sub-face's node |
| Symmetry-plane test | one-sided nodal gradients at symmetry nodes | half channel off the full one by 3 × 10⁻³ | mirror the gradients (zero normal derivative of p and tangential u; zero tangential derivatives of normal u) |
| Navier–Stokes manufactured solution | the Barth–Jespersen limiter left free | Picard held in a limit cycle at a relative change of 5 × 10⁻³ (tetrahedra, wedges) | freeze the limiter after 10 iterations; converges to 10⁻¹¹, error unchanged to 0.1 % |
| Womersley flow | pressure boundaries counted outflowing momentum but dropped inflowing | 20–31 % velocity error in oscillating flow | boundary momentum `ṁ_b u` in either direction (`backflow: consistent`) |
| False-time-step runs | Rhie–Chow `d` and the time share averaged separately | steady answers moved by 10⁻⁴ with the false time step | `d_ip = mean(V/(a + t))`, `f_ip = 1 − d_ip/mean(V/a)`: exact independence |
| Symmetry test after that fix | a volume-averaged `d_ip` (V̄/ā) is not invariant when a symmetry plane halves a control volume | half and full channels off by 8 × 10⁻⁹ | build `d_ip` from per-node ratios |
| DFG 2D-1 | reactions at nodes on a symmetry plane shared across all their zones | drag −7.7 % at m = 16, converging slowly | share each reaction component only among zones that fix it |
| Transient balances | the momentum check evaluated after the time state was cleared | a spurious 0.4 imbalance | keep the last time terms with the assembled system |
| No-slip manufactured solution | wrong curl in the manufactured velocity (a test defect) | divergence 0.19 | corrected; every exact solution is now checked against finite differences |
| Manufactured solution, slip walls | tetrahedral boundary pressure with prescribed sliding velocity | see above | **characterised, open** |
| Gate B | AmgX reusing its whole hierarchy (`structure_reuse_levels: −1`) | 200 iterations at 0.96 per iteration from the second outer iteration | rebuild per matrix (`0`) |
| Gate B | outer convergence tested on the change only | "converged" after 5 outer iterations while the linear solve stalled | also require the linear solve to have met its tolerance |
| Gate B | FGMRES residuals reported against ‖b‖ by one back end and against the initial residual by the other; `rtol=0` read as the default | inconsistent reports; a round-off start never "converged" | both relative to the initial residual, with a round-off floor |
| Two stiff lumped outlets | lagged Rhie–Chow gradient of a zero interior next to outlets at 1600 Pa | fixed-pressure outlets at that level diverged; the GPU solver was not level-invariant | start from the boundaries' pressure level |
| Two stiff lumped outlets (AmgX) | lumped rows of size `r Q` dominating the Krylov norm | residual 10⁻⁷ with the solution 0.25 off | scale the rows (then all rows dimensionless, next) |
| RCR pipe in SI units | rows in SI units: one equation's units dominate the 2-norm | transient coefficient loops drifted apart; one Krylov step per solve | dimensionless rows; steady RCR 130 → 38 outer iterations |
| RCR pipe, transient | each step reset lumped zones' pressure to 0 | NaN in the first coefficient loops | keep the solver's own pressure there |
| Two partitions | a halo node's local control volume used in `V/a_P` | mass imbalance stuck at 1.7 × 10⁻⁵ | use the whole control volume |
| Gate B at 329 k nodes | a second AmgX hierarchy and a whole-matrix copy each iteration | out of memory (AmgX code 5) | release the handle; keep only the needed raw rows |
| DFG 2D-2 run | block-Jacobi and DILU smoothing of the coupled saddle-point system on thin cells (the smoother alone diverges: more sweeps, faster) | every linear solve stalled at 200 iterations; c_D 17 instead of 3.2 | ILU(0) with pressure-weighted aggregation by default; SIMPLE; the `auto` ladder |
| Smoother sweep | ILU(1)'s fill on tetrahedral rows (about 20 blocks) | AmgX setup error (shared memory per row) | ILU(0) as the default |
| SimVascular segment (zvcfd-be) | wall shear at inlet rims took part of the inlet's pressure force, tangential to the wall | up to 3,950 Pa against a median 4.4 Pa at cut rims | subtract the inlet sub-faces' pressure force and momentum explicitly |
| Same mesh as OpenFOAM; developed pipe | deferred corrections on the faces leaving a pressure-inflow node, with nothing for them in its boundary momentum `ṁ_b u_node` | an inflow jet; the flow rate 3.4–3.8 % above the exact one, not converging; 10 % from OpenFOAM | the corrections stay out of the inflow node's own row and its boundary flux carries them, plus the cross-stream part of the faces' offset (`β = 0` there only moved the mismatch one row in; the full reconstruction is unstable on coarse meshes) |

## Reproducing

```bash
PYTHONPATH=. python benchmarks/validation/fv_cases.py            # every case, hours on one core each
PYTHONPATH=. python benchmarks/validation/fv_cases.py kovasznay  # one case
FOAM_SIF=esi2506.sif PYTHONPATH=. python benchmarks/fv/openfoam_same_mesh.py
ZVCFD_AMGX_LIB=.../libamgxsh.so PYTHONPATH=. python benchmarks/validation/fv_gpu_cases.py dfg2d2 16 32
ZVCFD_AMGX_LIB=.../libamgxsh.so PYTHONPATH=. python benchmarks/fv/gate_b.py 8 12 16 20
pytest tests/test_fv_validation.py tests/test_fv_properties.py tests/test_fv_gpu.py \
       tests/test_fv_solver_gpu.py tests/test_fv_boundaries.py tests/test_fv_multi.py tests/test_fv_run.py
```
