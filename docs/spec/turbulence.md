# Turbulence: the k-kL model

## Terms

**RANS**
: Reynolds-averaged Navier–Stokes. The equations for the mean flow, in
  which turbulence acts through an eddy viscosity `μ_t` added to the
  molecular viscosity.

**k**
: The turbulent kinetic energy per unit mass (m²/s²).

**Φ = kL**
: `k` times a turbulent length scale `L` (m³/s²): the second transported
  variable.

**k-kL-MEAH2015m**
: The model as implemented: NASA's k-kL-MEAH2015 in its incompressible
  form.

---

## Introduction

The finite-volume solver can solve turbulent flow with one optional
Reynolds-averaged model: the **k-kL** two-equation model. It transports
`k` and `Φ = kL` (Rotta's formulation, in the form Menter derived) and
closes them as Abdol-Hamid did. NASA's Turbulence Modeling Resource
defines it as **k-kL-MEAH2015** (Abdol-Hamid, Carlson & Rumsey,
NASA/TM-2015-218968). It is a low-Reynolds-number model: it needs the
viscous sublayer resolved (`y⁺ ≲ 1` at the first node), with no wall
functions, damping functions or blending between two models. Its length
scale equation uses the von Kármán length scale, built from the second
derivatives of the velocity, which also lets it relax towards resolved
structures in unsteady runs.

zvCFD's solver is incompressible, so it uses the Resource's
incompressible variant, **k-kL-MEAH2015m**: production `P = μ_t S²`, and
no `⅔ρk` in the Reynolds stress (it is absorbed in the pressure).

Turn it on in the `fv` section of a run configuration:

```yaml
fv:
  turbulence: {model: k-kl, speed: 1.0, mach: 0.15}   # the Resource's farfield values
  # or explicit freestream values: {model: k-kl, k_inf: 2e-6, kl_inf: 1e-11}
```

---

## Technical reference

### Equations

```text
∂(ρk)/∂t + ∇·(ρuk) = P_k − C_μ^¾ ρ k^{5/2}/Φ − 2μ k/d² + ∇·[(μ + σ_k μ_t)∇k]
∂(ρΦ)/∂t + ∇·(ρuΦ) = C_φ1 (Φ/k) P_k − C_φ2 ρ k^{3/2} − 6μ (Φ/d²) f_φ + ∇·[(μ + σ_φ μ_t)∇Φ]

μ_t  = C_μ^¼ ρ Φ / √k
P    = μ_t S²,     S = √(2 S_ij S_ij)
P_k  = min(P, 20 C_μ^¾ ρ k^{5/2}/Φ)
C_φ1 = ζ1 − ζ2 (Φ/(k L_vK))²,   C_φ2 = ζ3
f_φ  = (1 + C_d1 ξ)/(1 + ξ⁴),   ξ = ρ d √(0.3 k)/(20 μ)
L_vK = κ S / |∇²u|,   Φ/(k C11) ≤ L_vK ≤ C12 κ d f_p
f_p  = min(max(P_k Φ/(C_μ^¾ ρ k^{5/2}), 0.5), 1)
```

`d` is the distance to the nearest wall, computed exactly to the wall
triangles. The constants are:

| σ_k | σ_φ | κ | C_μ | ζ1 | ζ2 | ζ3 | C11 | C12 | C_d1 |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.0 | 0.41 | 0.09 | 1.2 | 0.97 | 0.13 | 10.0 | 1.3 | 4.7 |

With them, an equilibrium log layer has `k = u_τ²/√C_μ` and `L = κy`.
The `Φ` equation balances only with its diffusion term included:
`ζ1 − ζ2 − ζ3 C_μ^{-¾} + κ²/√C_μ = 0` to three figures.

### Boundary values

| Zone | k | Φ |
|---|---|---|
| wall | 0 | 0 (and `μ_t = 0`) |
| velocity inlet | the inflow value (`k_inf`, or a profile) | the inflow value |
| pressure | carried out where flow leaves (zero gradient); the inflow value where it enters | as k |
| symmetry | no flux | no flux |

The Resource's farfield values are `k∞ = 9 × 10⁻⁹ a∞²` and
`Φ∞ = 1.5589 × 10⁻⁶ μ∞ a∞/ρ∞`, `a∞` being the speed of sound. For an
incompressible run, `fv.turbulence: {speed, mach}` uses `a∞ = U∞/M∞` at the
case's nominal Mach number (`zvcfd.fv.turbulence.tmr_freestream`). The 2015
memorandum gives `9 × 10⁻¹⁰ a∞²` for `k∞`; the Resource's page, used here,
gives `9 × 10⁻⁹`.

### Discretisation

`k` and `Φ` live at the mesh nodes on the same control volumes as the
flow, and are solved after each outer iteration of the coupled
velocity–pressure solve: `k` first, then `Φ` with the new `k`. That is the
loose coupling CFL3D and FUN3D use (`zvcfd.fv.turbulence.KkLModel`, through
the solver's `turbulence` hook).

- **Advection**: first-order upwind with the solver's own mass flows at
  the integration points, in the bounded form `∇·(ρuφ) − φ∇·(ρu)`. Once
  continuity is met the two forms are the same. Until then, the bounded form
  keeps each row diagonally dominant where a node takes in more mass than it
  gives out.
- **Diffusion**: edge-based, as FUN3D's. Through the sub-face `s` on
  the element edge `a → b`, `∇φ·A_s` splits into a two-point part
  `c_s (φ_b − φ_a)`, with `c_s = max(A_s·e, 0)/|e|²` (implicit), and the
  remainder `∇φ·(A_s − c_s e)` from the shape functions (lagged). The
  remainder vanishes where the sub-face is normal to its edge, as on the
  hexahedra of a structured boundary layer. The diffusivity
  `μ + σ μ_t` is interpolated to the integration point by the shape
  functions. The full shape-function diffusion, which the momentum
  equations use, is not monotone on stretched elements: its neighbour
  coefficients take the wrong sign wherever an element is much longer
  than it is thick. On the NASA flat-plate slab, 94 % of the rows had one,
  and `k` ahead of the plate came out as a sawtooth that jumped from the
  floor to 10⁴ times its freestream value from one node to the next.
- **Sources**: at the nodes, from nodal gradients; `∇²u` comes from the
  gradients of the nodal gradients. The sinks are linearised by Newton's
  method. A sink `g ∝ φ^m` contributes `m g/φ` to the diagonal and
  `(m − 1) g` to the right side: `m = 5/2` for `C_μ^¾ ρ k^{5/2}/Φ`,
  `−1/2` for the unlimited production `P_k ∝ k^{−1/2}`, and the `Φ`
  exponents of `ζ2 (Φ/(k L_vK))² (Φ/k) P_k`. The destruction `ζ3 ρ k^{3/2}`,
  which does not depend on `Φ`, keeps the positive form
  `(ζ3 ρ k^{3/2}/Φ) Φ`. The production `ζ1 (Φ/k) P_k` is written
  `(ζ1 (Φ/k) P_k/Φ) Φ` on the diagonal, as far as the sinks leave the
  diagonal at least half its size. Every coefficient stays positive, and
  near equilibrium each update contracts: the plain `(k^{3/2}/Φ) k` form
  of `k^{5/2}/Φ` gives an update gain of about −2, Newton's about −0.2.
- **Relaxation**: the sources are under-relaxed (`relax`, 0.7). That is a
  pseudo-time step of about `relax/(1 − relax)` turbulent time scales,
  since the sources' diagonal is the sinks' rate. Relaxing the whole
  diagonal, as is usual for the momentum equations, would scale it by the
  diffusion across the thinnest cells (wall layers, one-cell slabs). That
  coupling is large and limits nothing, and the step would be orders of
  magnitude shorter than the turbulence's own time scale. On the flat plate
  the turbulence then took thousands of iterations to settle. `μ_t` passes
  to the flow under-relaxed (`relax_mu_t`, 0.7).
- **False time step**: a steady run with `fv.dt` marches `k` and `Φ` with
  the same pseudo-time step as the flow, from the last iterate.
- **Positivity**: in one update no node's `k` or `Φ` may fall below a
  tenth of its previous value, nor below 10⁻⁶ of its freestream value. A
  converged solution is unaffected.
- **Linear solve**: FGMRES preconditioned by an AmgX V-cycle (aggregation,
  Gauss–Seidel smoother), on rows scaled to a unit diagonal. Without AmgX,
  GMRES is used instead. A wall layer couples its nodes far more strongly
  across the thin direction than along it, and Jacobi-preconditioned
  Krylov methods stall on that anisotropy.
- **Coupling**: `μ_t` reaches the momentum equations through
  `GPUSolver.mu_t`, which adds it to the viscosity at every integration
  point and switches on the `μ(∇u)ᵀ` term. A steady run converges when the
  relative change of `k`, `Φ` and `μ_t` also falls below `fv.tolerance`.
- **Transient runs**: backward Euler for `k` and `Φ`.

Wall shear stress in turbulent runs is the molecular viscous traction,
since `μ_t = 0` at the wall ([wall shear stress](fv_numerics.md#wall-shear-stress)).

### Meshes

The model needs `y⁺ ≲ 1` at the first node off every wall: prism or
hexahedral layers at the wall, as CFX meshes have. Two-dimensional cases
run as one-cell-deep slabs, with a span of a few wall spacings.

On such slabs the flow's iterative linear solvers stall once the eddy
viscosity is on. Away from the wall the cells are tens of times taller
than the span, and `μ_t` makes the coupling across the span the dominant
one. The AmgX presets, the `auto` ladder and SIMPLE then make no progress:
FGMRES ends each outer iteration at its iteration limit with the residual
where it started. The outer loop still settles, onto a state that does not
satisfy continuity: on the flat plate, a tenth of the nodes missed it by
more than 10 % of their throughflow, and `c_f` came out 40 % low. With the
host direct solver (`linear: host-direct`) the same case converges in
about 150 iterations, with continuity met to round-off. The
two-dimensional validation cases use it. The AmgX presets were tuned on
meshes whose only strong anisotropy is across the wall layers (the
SimVascular coronary model's prism layers). The ONERA M6 C-H grid of
[Turbulent flow](../validation/turbulence.md#onera-m6-at-low-speed) has far-field
cells that are long in the plane of the sections and thin across the span,
and there the iterative solvers stall even for laminar flow.

### Validation

[Turbulent flow](../validation/turbulence.md) has the NASA flat plate and
NACA 0012 cases, and two independent programs that solve the same
equations (a 1-D channel and boundary-layer marching).

### Limits

- Incompressible only (the `m` variant). The Resource's free-shear and
  compressibility corrections (k-kL-MEAH2015+J) are not implemented.
- One GPU: the partitioned solver refuses a turbulence model for now.
- On one-cell-deep slabs, and on the ONERA M6 C-H grid, the flow's iterative
  linear solvers stall ([Meshes](#meshes)); the two-dimensional cases use the
  direct solver.
