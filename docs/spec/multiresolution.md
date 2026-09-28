# Multiresolution

## Terms

**Level**
: One resolution of an OME-Zarr input pyramid (`datasets[k]`), or of a
  zarr-vectors store. Level 0 is the finest.

**Restriction / prolongation**
: Moving a field from a fine level to a coarse one, or back.

**Diffusive scaling**
: Changing the voxel size while keeping the relaxation time τ fixed, so
  the lattice viscosity is unchanged and `dt ∝ dx²`.

**Preview**
: A solve on a coarse level whose result is used as is, with its error
  stated, rather than as a starting point.

---

## Introduction

Large images arrive as pyramids. The question for a solver is which of the
standard multiresolution tricks actually pay. The Ansys ones are: coarse
solve then interpolate; FMG initialisation; AMG. zvCFD measured each (see
[Multiresolution benchmarks](../benchmarks/multiresolution.md)). This page
fixes the rules that follow from the measurements.

---

## Technical reference

### Level selection

A run solves on one level, chosen by `source.level` or by the voxel size
closest to a target (`zvcfd.io.omezarr.Multiscale.nearest_level`). The
choice SHOULD resolve the narrowest channel that matters by at least
6 voxels across. At 20 µm the coronary tree's smallest outlet (0.24 mm)
is 12 voxels across. At 50 µm it is under 5.

### Unit scaling between levels

At fixed τ, one 2× coarsening multiplies, in lattice units:

| Quantity | Factor per coarsening |
|---|---:|
| velocity | 2 |
| density (pressure) difference | 4 |
| body force | 8 |
| time step (physical) | 4 |

`zvcfd.units.level_factors(k)` returns these for `k` coarsenings. A coarse
level's lattice Mach number is 2ᵏ times the fine one's, which limits how
coarse a level can be run at the same physical velocity.

### Mask coarsening

`zvcfd.phantoms.coarsen_mask(fluid, rule)` implements the two rules
compared in the benchmarks. `majority` makes a coarse voxel fluid if at
least 4 of its 8 children are. `any` makes it fluid if any child is (this
keeps connectivity, but widens every channel). Neither preserves hydraulic
conductance: flux scales with r⁴, so a half-voxel error in the radius of a
5-voxel vessel is a ~40 % flux error. Coarse levels of thin-vessel
geometries therefore mispredict flow by tens of percent (measured +25 %
at one level and +73–79 % at two, `majority` rule).

### What levels are used for

| Use | Status | Rule |
|---|---|---|
| Choosing the solve resolution | built | as above |
| Preview solves (the "fidelity slider") | built (any level, via `source.level`) | the result MUST be labelled with its level; its error in thin vessels is tens of percent |
| Multigrid for elliptic solves | built (lubrication, CPU AMG) | hierarchies MUST be algebraic (built from the fine operator), not from coarsened masks |
| Initialising a fine LBM run from a coarse one | **rejected** | measured 0.8–1.6×, and slower on long domains; not implemented in the solver |
| Field pyramids for viewing | roadmap | brick-averaged fields written as further store levels |
