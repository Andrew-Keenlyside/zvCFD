# Choosing brick and chunk sizes

Three sizes shape a run: the **brick** (the GPU's unit), the **chunk** (the
store's cell and the unit a GPU owns), and optionally the **shard** (the
file). Only the brick is fixed today (8³).

---

## Brick: 8³

The sparse kernel is compiled for 8³ bricks, with one 512-thread CUDA
block per brick. Smaller bricks (4³) would fill thin vessels better, but
64-thread blocks and a 27-entry neighbour lookup per 64 voxels would cost
kernel efficiency. Larger bricks (16³) fill worse and exceed the CUDA
block limit. 8³ is the standard compromise (NanoVDB leaf nodes, GVDB, and
waLBerla's blocks are comparable).

## Chunk: `chunk_bricks`, default 32 (256³ voxels)

| Consideration | Smaller chunks | Larger chunks |
|---|---|---|
| Load balance across 8 GPUs | better | worse for small domains |
| Files per snapshot (flat) | more: one per chunk per field | fewer |
| Cell size on disk | smaller; below 1–4 MB loses Lustre stripe parallelism | larger |
| Halo surface per GPU | unaffected (ownership is a union of chunks) | unaffected |
| Partial reads of a region | finer | coarser |

A rule of thumb: aim for **≥ 64 chunks per GPU** (so the partition can
balance), and cells of 1–64 MB per field. For the 10 µm coronary tree
(1.1 × 10⁹ stored cells, fill 0.7) at 32³ bricks, each chunk holds up to
3.3 × 10⁴ bricks. The occupied chunks number in the low thousands, and a
cell of one float32 field is typically 5–30 MB.

## Shard: `output.shard_shape`

Shards pack `shard_shape³` cells into one file. In our benchmark,
`shard_shape = 2` cut a snapshot from 7,353 files to 1,281 (5.7×) at a
cost of ~20 % in write rate. Use shards when:

- file counts approach an inode quota (BRIDGE hit one at 933 k files), or
- the store will be served from object storage.

With shards, workers must own whole shards. `BrickDomain.partition(n,
chunk_bricks · shard_shape)` gives that. Checkpoints written by one
process do not need it.

## Precision of stored fields: `output.dtype`

`float16` halves snapshot size and write time. Velocities in lattice units
are ≲ 0.1, and half precision keeps about three significant digits, which
is enough for viewing and most post-processing. Keep `float32` for fields
that will be differenced (pressure drops across a stenosis, for example)
or used to restart.
