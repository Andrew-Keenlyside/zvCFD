# Brick stores

## Terms

**Brick**
: A cube of `B³` voxels, `B = 8`. The unit of storage and of GPU work.
  A brick is *active* if any of its voxels is not solid.

**Brick coordinate**
: The integer position `(bz, by, bx)` of a brick: voxel `(z, y, x)` lies in
  brick `(z // B, y // B, x // B)`.

**Chunk** (of a brick store)
: A cube of `C³` bricks (`C` = `chunk_bricks`, default 32). One chunk is
  one *cell* of the Zarr Vectors store: the unit of I/O and of ownership.

**Brick row**
: One row of a brick store's vertex arrays, describing one active brick.

**Voxel order**
: Within a brick, voxel `(lz, ly, lx)` is element `lx + B·(ly + B·lz)`:
  x fastest, as in a C-ordered `(z, y, x)` array.

---

## Introduction

A brick store maps a sparse voxel field onto the Zarr Vectors data model
without changing it. Each active brick becomes one vertex whose position
is the brick centre. Each field becomes a vertex attribute whose row is
the brick's 512 voxel values. Because Zarr Vectors chunks vertices
spatially, the bricks of one store chunk land in one cell, and cells hold
only the bricks that exist.

Any Zarr Vectors reader can open a brick store. It sees a point cloud of
brick centres carrying wide attributes. zvCFD reads it back as bricks.

---

## Technical reference

### Root

The store MUST be created with `zarr_vectors.building.create_store`
with:

| Argument | Value |
|---|---|
| `bounds` | `([0, 0, 0], [Z·dx, Y·dx, X·dx])` for a domain of `(Z, Y, X)` voxels of edge `dx` |
| `chunk_shape` | `(C·B·dx,)·3` |
| `axes` | `[{"name": "z"}, {"name": "y"}, {"name": "x"}]`, all `type: "space"` |
| `unit` | a UDUNITS-2 length, default `"micrometer"` |
| `geometry_types` | `["zvcfd:bricks"]` |
| `shard_shape` | optional; cells per shard side |

The root's user metadata namespace `zvcfd` (reached through
`zarr_vectors.open(path).metadata["zvcfd"]`) MUST hold:

```json
{
  "brick": 8,
  "chunk_bricks": 32,
  "voxel_size": 2.5,
  "unit": "micrometer",
  "shape_zyx": [1536, 1536, 1536],
  "fields": {"rho": "float32", "ux": "float32", "uy": "float32", "uz": "float32"}
}
```

### Level 0 arrays

| Array | dtype | Row width | Content |
|---|---|---:|---|
| `vertices` | float32 | 3 | brick centre `((bz, by, bx) + ½)·B·dx` |
| `vertex_attributes/<field>` | per `fields` | 512 · components | voxel values in voxel order |
| `vertex_attributes/flags` | uint8 | 512 | voxel flags (domain store only) |

Rows within a cell are in any order; the brick coordinate is recovered from
the vertex position as `round(position / (B·dx) − ½)`. A reader MUST NOT
assume rows are sorted. zvCFD writes them in Morton order of brick
coordinate, which keeps spatially close bricks close in memory.

Each cell is written as a single fragment (no `base_bin_shape`). Brick
stores are not coarsened with zarr-vectors' per-object pyramid builder;
field pyramids, when wanted, are computed by brick averaging and written as
further levels (see [Multiresolution](multiresolution.md)).

### Values of solid voxels

Solid voxels inside an active brick MUST be written as `0` in every field.
Readers SHOULD mask them with the domain store's `flags`.

### Codecs

Arrays inherit the codec pipeline of the write session that created them.
zvCFD creates all attribute arrays in the coordinator's session, with
the store's `compressor` (`None` or `"zstd"`). Uncompressed cells decode on
the GPU without nvCOMP. zstd cells decode on the host, or on the GPU with
`decode="device"`, which the zarr-vectors documentation reserves for
trusted stores.

### Sizes

A chunk cell of a float32 field holds `n_active · 2048` bytes. At the
default `C = 32` that is up to 64 MiB per field per cell (32,768 bricks). A
cell of a half-full chunk of a vessel tree at 20 µm is typically 1–10 MB,
which is inside the 1–4 MB Lustre stripe guidance at the low end and fine on
NVMe.

### Worked example

```python
from zvcfd import BrickDomain
from zvcfd.io import fields as zf

level = zf.create_brick_store("snap.zarrvectors", domain, voxel_size=2.5,
                              fields=["rho", "ux", "uy", "uz"], chunk_bricks=32)
zf.write_brick_chunks(level, domain, rows, voxel_size=2.5, chunk_bricks=32)
zf.finalize_brick_store(level)
```

On disk:

```text
snap.zarrvectors/
├── zarr.json                 # zarr_vectors root; user_metadata.zvcfd
└── 0/
    ├── zarr.json             # level metadata
    ├── vertices/c/<i>/<j>/<k>
    ├── vertex_fragments/c/<i>/<j>/<k>
    └── vertex_attributes/
        ├── rho/c/<i>/<j>/<k>
        ├── ux/…  uy/…  uz/…
```
