# Architecture

This page describes the whole design, including the parts not yet built.
Each section says which it is.

---

## Layers

```text
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Interfaces       zvcfd CLI (probe, plan, mesh-info, run, info)           │ built
 │                  Python: zvcfd.*, RunConfig (YAML/JSON, strict)           │ built
 ├──────────────────────────────────────────────────────────────────────────┤
 │ Orchestration    coordinator + one worker process per GPU                 │ designed
 │                  NCCL / CUDA peer copies for halos; async writer per GPU  │ designed
 ├──────────────────────────────────────────────────────────────────────────┤
 │ Compute          SparseLBM / DenseLBM (D3Q19, fp32/fp16)                  │ built (1 GPU)
 │                  lubrication solve + AMG                                  │ built (CPU AMG)
 │                  boundary conditions, TRT/cumulant collision              │ roadmap
 ├──────────────────────────────────────────────────────────────────────────┤
 │ Domain           BrickDomain: bricks, neighbours, chunks, partition, halo │ built
 ├──────────────────────────────────────────────────────────────────────────┤
 │ Storage          brick stores via zarr_vectors.building                   │ built
 │                  OME-Zarr inputs via zarr; Fluent .msh reader              │ built
 │                  run collection (RFC-8), atomic publication                │ built
 └──────────────────────────────────────────────────────────────────────────┘
```

The rule between layers: **the solver never works through the store.**
Zarr Vectors is where a run starts, where it publishes results, and where
it restarts from. During the time loop everything lives in GPU memory,
because HBM is 60× faster than the PCIe link and orders of magnitude faster
than storage.

---

## Data flow of a run

### 1. Ingest: image to domain (built for one process; chunked version designed)

The fluid mask comes from one level of an OME-Zarr pyramid (a label value
or a threshold). The level can also come from a Fluent wall surface,
voxelised. For volumes that do not fit in host memory, the ingest works per
store chunk:

1. Read chunk *c* of the chosen level, plus a one-voxel halo, with `zarr`.
2. Threshold it into flags, and find the chunk's active bricks.
3. Write the chunk's cell of the **domain store** (`vertex_attributes/flags`).
4. The coordinator sums active-brick counts per chunk into global brick
   ids (a prefix sum), and no global dense array ever exists.

Chunks are independent, so ingest runs on as many CPU processes or GPUs as
there are. It uses the same three-phase write as everything else.

### 2. Partition (built)

`BrickDomain.partition(n_gpus, chunk_bricks)` orders occupied chunks along
a Morton curve and cuts it into runs of equal fluid work. Each GPU owns
whole chunks, or whole shards when the store is sharded. That makes the
write contract hold by construction, and keeps each GPU's region compact,
so its halo is small. It is the analogue of Fluent's Cartesian and
principal-axis bisection. It keeps the owned unit aligned to the storage
unit, which a graph partitioner (METIS) would not.

### 3. Solve (built for one GPU; multi-GPU designed)

Each worker holds its bricks plus a one-brick **ghost layer** of its
neighbours' bricks (`BrickDomain.halo`). A step is:

```text
  stream A:  collide+stream interior bricks (no ghost dependency)
  stream B:  pack boundary populations -> peer copy over NVLink -> unpack ghosts
  stream A:  wait(B); collide+stream boundary bricks
```

Only populations that cross a face need to travel: 5 of 19 per face
direction. For the 10 µm coronary run (6.3 × 10⁸ fluid voxels, eight GPUs),
the halo is a few percent of each GPU's bricks. At 450 GB/s per direction
over NVSwitch, the exchange takes well under a millisecond against a ~12 ms
step, and it overlaps with the interior kernel. The transport is
`cupy.cuda.nccl` send/recv, the pattern BRIDGE-Simulation already has
written (`NcclComm`), or direct `cudaMemcpyPeerAsync` within one node.

### 4. Output (built synchronously; asynchronous designed)

Every `output.every` steps:

1. `fields()` computes ρ and **u** into device arrays (one kernel).
2. A copy stream moves them into a **pinned host ring buffer** while the
   next steps run.
3. A writer thread pool per GPU encodes and writes that GPU's chunk cells
   (`write_brick_chunks`, presence deferred, no lock).
4. When all workers report, the coordinator runs `finalize_brick_store`
   (one presence rebuild) and adds the snapshot node to the run collection
   in one atomic replace.

Overlap is what matters, not raw write speed. A 10 µm coronary snapshot is
~17 GB (four float32 fields over 1.05 × 10⁹ stored cells). At the measured
0.4–0.6 GB/s per node it takes ~30–40 s. That is free if snapshots come
more than ~3,000 steps apart, and costly if more often (see
[I/O](../benchmarks/io.md)).

### 5. Checkpoint and restart (designed)

A checkpoint is a brick store of the 19 populations (fp32 or fp16),
written exactly like a snapshot and published the same way. Restart reads
it with `read_cells(..., device="cuda")`, each GPU reading only the chunks it
owns. Keep the last two. Repartitioning on restart, with a different GPU
count, needs nothing more, because ownership is computed from the domain.

---

## Scaling out

| Regime | Fluid voxels (fill 0.6) | Route |
|---|---|---|
| One GPU | ≲ 3 × 10⁸ fp32 | `zvcfd run` today |
| One node | ≲ 2 × 10⁹ fp32, ≲ 4 × 10⁹ fp16 | multi-GPU driver |
| Beyond one node, one region | 10¹⁰ | multi-node (NCCL over InfiniBand), same code path |
| Whole organ | 10¹¹–10¹² | network model on the centreline graph (a ZV `graph` store) plus 3-D sub-volumes whose boundary pressures come from it |

**Streaming bricks from host or disk each step is not viable.** A step
touches every population once, and PCIe moves ~55 GB/s against HBM's
3.35 TB/s. Temporal blocking (many steps per load) recovers some of that
at great complexity. Published out-of-core stencil work reaches 1.1–2.8×
over naive streaming, still far from in-core speed.

---

## Where each Zarr Vectors feature is used

| zarr-vectors feature | Used for |
|---|---|
| `building.create_store(shard_shape=…, compressor=…)` | brick stores; shards when file count matters (Lustre inode quotas) |
| `defer_presence` / `rebuild_presence` | lock-free parallel writes by GPU workers |
| `read_cells(device="cuda")`, `read_neighbourhood(halo=1)` | restart, sub-volume boundary data, post-processing on GPU |
| `runtime_capabilities()` | `zvcfd probe` gates |
| user metadata namespaces (`ds.metadata["zvcfd"]`) | brick size, voxel size, field dtypes on each store |
| OME `collection` root node | brick stores open as collection members |
| `mesh`, `polyline`, `graph` types | wall surfaces, streamlines, centreline networks (roadmap) |
| multiscale pyramids, `build_pyramid` | coarse previews of fields for viewing (roadmap) |
