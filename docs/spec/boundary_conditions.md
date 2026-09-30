# Boundary conditions

## Terms

**Flag**
: The per-voxel uint8 that tells the kernel how to treat a voxel.

**Patch**
: A named set of boundary voxels sharing one condition, e.g. an inlet or
  one of 77 coronary outlets (`zvcfd.boundary.Patch`).

**Interior neighbour**
: For each patch voxel, the fluid voxel one lattice link inward (the link
  closest to the inward normal) that the boundary scheme extrapolates from.

**Patch link**
: A lattice link from a fluid voxel across a surface cap to a non-fluid
  node. Pressure outlets on caps carry their condition on these links
  (anti-bounce-back), the way walls carry theirs on the links into solid.

**Zone** (Fluent)
: A named face or cell group in an Ansys mesh, with a type such as
  `wall`, `velocity-inlet` or `pressure-outlet`.

---

## Introduction

In a voxel solver a boundary condition is a property of voxels and their
links. A wall is wherever a fluid voxel meets a solid one. Pressure
outlets on surface caps (pressure, RCR and coronary patches from a mesh)
are the lattice links that cross the cap: an anti-bounce-back condition
on each link imposes the pressure half a link out, at the cap. Velocity
inlets, and patches on the faces of the bounding box, are layers of
voxels where values are imposed: their voxels are flagged, and after
every stream-and-collide step a boundary kernel rewrites them from the
imposed state and the state of their interior neighbour. Patches carry physical
values (Pa, m/s, m³/s, Windkessel parameters); the run converts them to
lattice units at every check interval, which is also when flow-rate control
and Windkessel outlets update.

---

## Technical reference

### Flags

| Flag | Name | Kernel behaviour |
|---:|---|---|
| 0 | fluid | pull-stream from neighbours, collide (BGK or TRT, optionally Carreau–Yasuda) |
| 1 | solid | never updated; a fluid voxel's link into it bounces back half-way |
| 2 | reservoir | legacy: populations reset to equilibrium at a fixed density, zero velocity (x faces only) |
| 3 | pressure patch (box faces, explicit voxel sets) | Guo non-equilibrium extrapolation: prescribed density, velocity from the interior neighbour. Pressure patches on surface caps use patch links instead, and their voxels stay fluid |
| 4 | velocity patch | Guo non-equilibrium extrapolation: prescribed velocity × a per-voxel profile factor, density from the interior neighbour |

The main kernel skips flags 3 and 4. The boundary kernel (`boundary_neem`)
then writes them:

```text
f*(b) = feq(rho_b, u_b) + [ f*(n) - feq(rho_n, u_n) ]
```

Here *b* is the patch voxel and *n* its interior neighbour. The result is
read from the freshly written buffer (Guo, Zheng & Shi 2002). A patch voxel
with no fluid neighbour on any link inward takes the equilibrium alone.

The copied non-equilibrium part is exact only where the flow does not
change between *b* and *n*. That holds at a vessel cut square to its axis,
where steady fluxes are τ-independent to 10⁻⁵. It fails at a face cut
straight through a porous sample, where the pressure-driven flux varied by
7 % between τ = 0.6 and τ = 2. Four open voxel layers between the sample
and the patch reduce the spread below 1 %. Place patches where the flow is
locally developed. Voxelised mesh inlets and outlets already are.

The extrapolation also fails where the flow crosses a patch off the
lattice axes, the more so as τ approaches ½. On a straight pipe along
(1,1,1) with patches normal to it, it lost 9 % and 7 % of the imposed
pressure difference at the two patches at τ = 0.55. Every outlet of a
patient vessel tree is oblique, and the SimVascular comparison found a
5–110 Pa jump at each coronary outlet
([Against SimVascular](../validation/simvascular.md)). That is why
pressure patches on caps now use patch links.

### Pressure outlets on caps: patch links

For each cap, `zvcfd.geometry.cap_links` finds every link from a fluid
voxel centre to a non-fluid voxel whose segment crosses the cap's
triangles, including diagonals from the second layer that cross a
staircase. Each link's population arriving at the fluid voxel *i* along
`c_q` is set by anti-bounce-back (Ginzburg; Krüger et al. 2017, ch. 5):

```text
f_q(i, t+1) = −f*_q̄(i, t) + 2 w_q ρ_b [1 + 4.5 (c_q·u)² − 1.5 u²]
```

Here ρ_b is the patch density and u the velocity of voxel *i*. The
condition acts half a link out, as half-way bounce-back does for walls,
and with TRT (Λ = 3/16) it holds there independently of τ. The kernel
`abb_links` writes the value into the slot the step's bounce-back reads,
just before each step. That slot is read by nothing else, so the main
kernel is unchanged and the buffer after a step is clean. The link
voxels stay fluid. Their mask (bit q per link) and patch index live in
`BoundarySet.link_cells`, `link_mask` and `link_patch`. Patch voxels are
still found for every cap, so the same zone can be a velocity patch
instead.

| Straight pipe, R = 8, L = 160 (`benchmarks/simvascular/oblique_outlet.py`) | Extrapolation (before) | Patch links (now) |
|---|---:|---:|
| (1,1,1) axis, Δp lost at the two patches, τ = 0.55 / 0.8 | 16.6 / 2.2 % | −1.1 / −1.1 % |
| (1,2,3) axis, the same | 6.6 / 0.5 % | −1.1 / −1.1 % |
| axis along z, the same | −0.6 / −0.6 % | −0.9 / −0.9 % |

The remaining 0.5 % per patch is the staircase placement of the cap,
fixed for a given geometry and independent of τ.

### Patch kinds

| Kind | Flag | Prescribed | Updated each check from |
|---|---:|---|---|
| `pressure` | 3, or links | pressure (Pa) → density | the pressure × its waveform factor |
| `rcr` | 3, or links | Windkessel pressure | `C dPd/dt = Q − (Pd − Pv)/Rd`, `P = Pd + Rp·Q`, with Q the measured outflow; advanced over each check interval by the trapezoidal rule (`zvcfd.lumped.RCR`) |
| `coronary` | 3, or links | open-loop coronary pressure (Kim et al. 2010) | `Ra`, `Ca`, `Ram`, `Cim`, `Rv` (`coronary: [Ra, Ca, Ram, Cim, Rv]`, SI) and the intramyocardial pressure `pim` (Pa, or a periodic `[[t, Pa], ...]` table); venous pressure from `pressure`. `zvcfd.lumped.Coronary`, trapezoidal |
| `velocity` | 4 | mean normal velocity or flow rate, plug or parabolic | the waveform; with `flow_rate` and flow control on, a gain that drives the **measured** flux to the target |

### Measuring flow

`patch_flux` counts, for every patch voxel, the populations it sends into
fluid voxels minus those fluid voxels send into it: the exact lattice mass
flux through the patch. For patch links it counts, per link, what arrives
(the anti-bounce-back population) less what leaves; `abb_flux` computes
that from the same buffer, so it is the flux of the next step exactly. It is summed per patch, and flows, splits and the
mass balance all use it. Velocity × area is not used: a node-based
velocity condition delivers about 1 % less flux than its nominal plug
(measured on a plane channel), which flow control then removes.

| Check | Result (`tests/test_physics.py`, `tests/test_geometry.py`) |
|---|---|
| patch flux = interior mass flux, plane channel | within 0.2 % |
| flow control reaches target | within 0.2 % |
| inflow = outflow, voxelised capped cylinder | within 0.5 % |
| oblique-pipe patch links hold their pressure, τ = 0.55 and 0.8 (`tests/test_outlets.py`) | within 1.5 % per patch; flux τ-independent to 0.2 % |
| patch links across partitions | bit-identical fields |
| pressure-driven plane Poiseuille | within 2 % of analytic |

### Where patches come from

From a **Fluent mesh** (`zvcfd.geometry.voxelize_fluent`)
: Each non-wall zone's triangles are sampled at a third of a voxel. The
  fluid voxel half a voxel inside each sample joins the patch, and the
  zone's area, outward normal and centroid come with it; so do the
  zone's patch links. Inlets become velocity patches (on the voxels) and
  outlets pressure patches (on the links); a run's
  `boundaries.patches` rules (matched by substring of the zone name) then
  set the values. The HiP-CT coronary mesh gives 78 patches (1 inlet, 77
  outlets) at 50 µm, with 21–902 voxels each.

From the **bounding box** (`zvcfd.boundary.face_patches`)
: Fluid voxels on a face (`xmin` … `zmax`). Configured as
  `boundaries.faces`.

From **explicit voxel sets** (`zvcfd.boundary.from_cells`)
: Any global cell ids, e.g. from an OME-Zarr label image with one label
  per patch.

### Walls

Half-way bounce-back puts the wall midway between the last fluid voxel and
the first solid one. With TRT and Λ = 3/16 that location is exact for
straight walls, independent of τ (plane channel: exact to 2 × 10⁻⁶ at
τ = 0.55, 0.8 and 2). The wall is still a staircase, so wall shear
stress needs interpolated bounce-back ([Risks](../feasibility/risks.md#3-staircase-walls-and-wall-shear-stress)).

### Not yet built

- Backflow stabilisation at pressure outlets.
- Patch tables persisted in the domain store (patches are rebuilt from the
  mesh or configuration on each run).
- Closed-loop 0-D models (heart and systemic circulation).

### Lumped outlet models

`zvcfd.lumped` holds the 0-D outlet models for both solvers, as linear
state-space systems `dx/dt = A x + Bq Q + Bu u(t)`, `P = Cx x + Dq Q`,
with `u = [P_v, P_im(t)]`. A step of length `dt` ending at `t` gives the
patch pressure as an affine function of that step's end flow,
`P(t) = a + r Q(t)` (`coefficients`). An implicit solver enters `r` into
its equations; the LBM patch controller measures `Q` and calls `advance`,
which commits the step. The coronary state is `[P_1, P_2 − P_im]`, so
`P_im` is never differentiated. `Coronary.from_svsolver` inverts
svSolver's `cort.dat` ODE coefficients to the five circuit parameters
(the redundant coefficient checks the inversion), and
`read_svsolver_cort` / `read_svsolver_rcrt` read SimVascular's outlet
files in SI.

| Check (`tests/test_lumped.py`) | Result |
|---|---|
| steady state under constant flow | `P = P_v + (total resistance) Q` to 10⁻⁶ |
| periodic flow against the exact impedance `Z(iω)`, dt = 4, 2, 1 ms | trapezoidal: max error 8.6 × 10⁻⁶ → 5.4 × 10⁻⁷, order **2.00**; backward Euler: order **1.00** |
| zero flow, sinusoidal `P_im` against `b₁ s / (1 + p₁ s + p₂ s²)` | within 10⁻³ |
| VMR 0066 `cort.dat` (24 outlets) | total resistance = svSolver's `q0` × 10⁵ to 10⁻⁹ |
