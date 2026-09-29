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

**Zone** (Fluent)
: A named face or cell group in an Ansys mesh, with a type such as
  `wall`, `velocity-inlet` or `pressure-outlet`.

---

## Introduction

In a voxel solver a boundary condition is a property of voxels. A wall is
wherever a fluid voxel meets a solid one. Inlets and outlets are layers of
voxels where values are imposed: their voxels are flagged, and after every
stream-and-collide step a boundary kernel rewrites them from the imposed
state and the state of their interior neighbour. Patches carry physical
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
| 3 | pressure patch | Guo non-equilibrium extrapolation: prescribed density, velocity from the interior neighbour |
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

### Patch kinds

| Kind | Flag | Prescribed | Updated each check from |
|---|---:|---|---|
| `pressure` | 3 | pressure (Pa) → density | the pressure × its waveform factor |
| `rcr` | 3 | Windkessel pressure | `C dPd/dt = Q − (Pd − Pv)/Rd`, `P = Pd + Rp·Q` (explicit, over the check interval), with Q the measured outflow |
| `velocity` | 4 | mean normal velocity or flow rate, plug or parabolic | the waveform; with `flow_rate` and flow control on, a gain that drives the **measured** flux to the target |

### Measuring flow

`patch_flux` counts, for every patch voxel, the populations it sends into
fluid voxels minus those fluid voxels send into it: the exact lattice mass
flux through the patch. It is summed per patch, and flows, splits and the
mass balance all use it. Velocity × area is not used: a node-based
velocity condition delivers about 1 % less flux than its nominal plug
(measured on a plane channel), which flow control then removes.

| Check | Result (`tests/test_physics.py`, `tests/test_geometry.py`) |
|---|---|
| patch flux = interior mass flux, plane channel | within 0.2 % |
| flow control reaches target | within 0.2 % |
| inflow = outflow, voxelised capped cylinder | within 0.5 % |
| pressure-driven plane Poiseuille | within 2 % of analytic |

### Where patches come from

From a **Fluent mesh** (`zvcfd.geometry.voxelize_fluent`)
: Each non-wall zone's triangles are sampled at a third of a voxel. The
  fluid voxel half a voxel inside each sample joins the patch, and the
  zone's area, outward normal and centroid come with it. Inlets become
  velocity patches and outlets pressure patches; a run's
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
- 0-D outlet models beyond RCR (open-loop coronary, closed loop), through
  svZeroDSolver.
