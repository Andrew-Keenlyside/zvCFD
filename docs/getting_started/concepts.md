# Concepts

This page is the mental model: what a zvCFD run is made of, from the voxel
up to the run collection, and which of those things you choose.

---

## Voxels and flags

zvCFD solves on the voxel grid of the image, in OME-Zarr axis order
`(z, y, x)`. Every voxel carries a flag:

| Flag | Meaning |
|---:|---|
| `0` | fluid |
| `1` | solid (a wall sits half-way between a fluid and a solid voxel) |
| `2` | fixed-density boundary: a reservoir that sets the pressure |

Flags come from a segmentation (a label value or a threshold on an
OME-Zarr level), from a synthetic phantom, or — for Ansys users — from
voxelising the wall surface of a Fluent mesh. There is no body-fitted mesh
and no meshing step.

## Bricks

A **brick** is an 8 × 8 × 8 block of voxels, and the unit of storage.
Only bricks holding at least one non-solid voxel exist. Two numbers
describe how well bricks fit a geometry:

active fraction
: active bricks / all bricks in the bounding box. Low for vessels (a
  coronary tree at 20 µm: well under 1 %), near 1 for porous media.

fill
: fluid voxels / voxels in active bricks. Memory is `fluid / fill` cells.
  Thin vessels (radius ≲ brick) fill poorly (0.35 in our vessel phantom);
  wide ones fill well (≳ 0.8).

On the GPU one CUDA block processes one brick, one thread per voxel. A
neighbour table gives each brick its 26 neighbours; a missing neighbour
reads as solid. The dense and sparse kernels produce bit-identical results.

## Chunks and shards

Bricks group into **chunks** of `chunk_bricks³` bricks (32³ bricks = 256³
voxels by default). A chunk is one *cell* of the Zarr Vectors store — one
file per field, or one entry of a shard — and the unit a GPU owns.

A **shard** packs several chunk cells into one file (`shard_shape` cells
per side), which cuts file counts 5–6× in our benchmark. With shards, the
shard becomes the unit a worker must own whole: two workers writing
different cells of one shard is a lost update.

## Workers and ownership

A run on *N* GPUs has one worker process per GPU and one coordinator.
`BrickDomain.partition` assigns whole chunks (or whole shards) to workers
along a Morton curve, balancing fluid cells, so each worker's region is
compact and no two workers ever write the same file. Each worker also holds
a one-brick **halo** of its neighbours' bricks, refreshed every step over
NVLink. (The multi-GPU driver is designed but not built; see the
[roadmap](../feasibility/roadmap.md).)

## Brick stores

A **brick store** is a Zarr Vectors store of custom geometry type
`zvcfd:bricks`:

- `vertices` — one row per brick: its centre, in physical units;
- `vertex_attributes/<field>` — that brick's 512 voxel values;
- `vertex_attributes/flags` — in the domain store only.

It is written through `zarr_vectors.building` in three phases: the
coordinator creates the store and allocates every array, then workers write
the chunks they own with presence deferred, and finally the coordinator
rebuilds presence once. Because solid-only bricks are never stored, a store's size follows the fluid
volume. See the [brick store specification](../spec/brick_store.md).

## Snapshots and checkpoints

A **snapshot** is one brick store of macroscopic fields (`rho`, `ux`, `uy`,
`uz`) at one step. A **checkpoint** is one brick store of the populations,
enough to restart. Each is written as a new store and becomes visible only
when the coordinator adds it to the run collection, in one atomic replace
of the collection's `zarr.json`. That gives snapshots all-or-nothing
visibility without a transactional store.

## The run collection

A run is a directory `<name>-<hash>.zvcfd/` that is itself a Zarr group
whose `attributes.ome` is an OME-NGFF RFC-8 **collection**. Its nodes point,
by relative path, at the input image (never copied), the mask, the domain
store, every snapshot, and the resolved configuration. The layout and
conventions follow BRIDGE's collection plan, so a BRIDGE subject and a
zvCFD run on it can sit in one study index. See
[Run collections](../spec/run_collection.md).

## Resolution levels

OME-Zarr inputs come as multiscale pyramids. zvCFD uses them in three ways:
to pick the resolution a run solves at, to run cheap **previews** on a
coarse level (8–43× cheaper in our tests, with 25–80 % flux error in thin
vessels), and to build multigrid hierarchies for the *elliptic* solves. It
does **not** use a coarse level to initialise a fine lattice-Boltzmann run:
we measured that, and it does not pay (see
[Multiresolution](../benchmarks/multiresolution.md)).

## Lattice units

The solver works in lattice units. A lattice is fixed by the voxel size
`dx`, the fluid's kinematic viscosity `nu` and the relaxation time `tau`;
the time step follows as `dt = (tau - 1/2)/3 · dx²/nu`. `zvcfd.Lattice`
converts velocities, pressures and times, and `zvcfd.units.level_factors`
gives the factors between pyramid levels (velocity ×2, density difference
×4, body force ×8 per coarsening, at fixed `tau`).
