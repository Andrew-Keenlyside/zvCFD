# Boundary conditions

## Terms

**Flag**
: The per-voxel uint8 that tells the kernel how to treat a voxel.

**Patch**
: A named set of boundary voxels sharing one condition, e.g. an inlet or
  one of 77 coronary outlets.

**Zone** (Fluent)
: A named face or cell group in an Ansys mesh, with a type such as
  `wall`, `velocity-inlet` or `pressure-outlet`.

---

## Introduction

In a voxel solver a boundary condition is a property of voxels: a wall is
wherever a fluid voxel meets a solid one, and inlets and outlets are sets
of voxels on the domain's surface where values are imposed. The flag
volume encodes the kind; a patch table encodes the values. Today the
kernel implements walls (half-way bounce-back) and fixed-density
reservoirs on the two x faces. General inlet/outlet patches are
[milestone 2](../feasibility/roadmap.md).

---

## Technical reference

### Flags

| Flag | Name | Kernel behaviour | Status |
|---:|---|---|---|
| 0 | fluid | pull-stream from neighbours, collide | built |
| 1 | solid | never updated; a fluid voxel's link into it bounces back half-way | built |
| 2 | reservoir | populations reset each step to equilibrium at fixed density, zero velocity; density `rho_in` for `x < nx/2`, `rho_out` above | built (x faces) |
| 3 | pressure patch | as 2, density from the voxel's patch entry | designed |
| 4 | velocity patch | regularised / Zou–He velocity condition from the patch entry (profile or flow rate) | designed |
| 5 | outlet with lumped model | pressure from a Windkessel (RCR) updated from the patch's flux | designed |

Flags ≥ 3 carry a patch id. It is stored in a parallel uint16
`vertex_attributes/patch` array of the domain store (0 = none).

### Patch table *(designed)*

The domain store's user metadata namespace `zvcfd` gains:

```json
"patches": [
  {"id": 1, "name": "cor_inlet_001", "kind": "velocity", "normal_zyx": [0.1, -0.2, 0.97],
   "area_mm2": 1.94, "waveform": "inlet_flow.csv"},
  {"id": 2, "name": "cor_outlet_001", "kind": "rcr", "R1": 1.2e9, "C": 3.1e-11, "R2": 9.8e9}
]
```

### Where patches come from

From an **Ansys Fluent mesh**
: `zvcfd.io.fluent_msh.read_fluent_boundary` returns every boundary zone
  with its Fluent type and name. The HiP-CT coronary mesh has one
  `velocity-inlet` (1.94 mm², equivalent diameter 1.57 mm), 77
  `pressure-outlet` zones (0.24–1.12 mm), and four wall zones including the
  crop planes. Voxelising a zone's faces onto the fluid mask gives the
  patch voxels. The zone's type gives the kind.

From a **label image**
: An OME-Zarr label volume (RFC-8 `labels.source` pointing at the image)
  with one label per patch, as BRIDGE writes masks.

From the **domain box**
: Fluid voxels on a face of the bounding box, the case the kernel handles
  today.

### Walls

Half-way bounce-back places the wall midway between the last fluid voxel
and the first solid one. Flow rate is second-order accurate for straight
walls, but the wall is a staircase. Interpolated bounce-back (Bouzidi)
uses sub-voxel wall distances, from a signed distance field or from the
Fluent wall surface, and is the planned route to accurate wall shear stress
([Risks](../feasibility/risks.md#3-staircase-walls-and-wall-shear-stress)).
