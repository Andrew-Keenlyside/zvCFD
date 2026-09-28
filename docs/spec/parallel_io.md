# Parallel I/O contract

## Terms

**Owner**
: The one worker allowed to write a given store cell (and, with sharding,
  a given shard) during a phase.

**Coordinator**
: The process that creates stores, finalises them and writes the run
  collection. Exactly one per run.

**Presence**
: zarr-vectors' `nonempty_chunks` manifest per array. It is shared by every
  cell of an array, so writing it from several processes races.

---

## Introduction

zvCFD adopts zarr-vectors' three-phase parallel-write pattern unchanged.
First, a coordinator allocates. Then workers write disjoint cells with
presence deferred. Finally, the coordinator rebuilds presence once. The
only thing zvCFD adds is how ownership is decided: by
`BrickDomain.partition`, which assigns whole chunks (or whole shards) along
a Morton curve. Ownership is disjoint by construction, so no lock is
needed anywhere in the write path.

---

## Technical reference

### The three phases

| Phase | Who | Calls | Shared state touched |
|---|---|---|---|
| 1. allocate | coordinator | `create_brick_store` → `create_store`, `create_resolution_level`, `create_*_array`, `defer_presence` | all metadata |
| 2. write | every worker | `write_brick_chunks(..., bricks=own)` → `write_chunk_vertices` / `write_chunk_attributes` with `record_presence=False` | only the worker's own cells |
| 3. finalise | coordinator | `finalize_brick_store` → `rebuild_presence`, `refresh_arrays_present`; then `collection.write` | presence, level metadata, the run document |

### Rules

1. Workers MUST NOT create arrays. Re-creating an array inside a write
   session drops the cells already on disk (zarr-vectors' HPC guide).
2. A worker MUST write only cells it owns. With `shard_shape` set, it MUST
   own every cell of each shard it touches: a shard is written as one unit,
   and two writers of one shard is a lost update. `BrickDomain.partition(n,
   chunk_bricks · shard_shape)` gives shard-aligned ownership.
3. Cells are never partitioned by coordinate ranges with closed upper
   bounds. Ownership comes from chunk coordinates, so this cannot happen.
4. Phase 3 MUST NOT start until every worker has returned. Readers that go
   through zarr-vectors see a deferred level's cells before the rebuild. Other
   readers (Neuroglancer's datasource) should wait for publication.
5. The run collection is written only by the coordinator, and only after
   phase 3 (see [Run collections](run_collection.md)).

### Reads

Any number of processes may read concurrently. `read_brick_chunks`
issues one `read_cells` call per request, fetching every requested cell of
every requested array in one pooled pass. With `device="cuda"`:

| Store cells | `decode="auto"` | `decode="device"` |
|---|---|---|
| uncompressed | decoded on the GPU | decoded on the GPU |
| zstd | decoded on the host, one upload | nvCOMP on the GPU (trusted stores only) |

Local files are read into pinned memory and copied up once, or with kvikio
straight into device memory when GPUDirect Storage is enabled
(`KVIKIO_COMPAT_MODE=OFF`, local NVMe or a GDS-capable parallel
filesystem).

### Asynchronous writing

The contract says nothing about *when* a worker writes. The designed
output path overlaps writing with compute: device fields go to a pinned
host ring buffer on a copy stream, then a per-GPU writer thread pool writes
them. It needs no transactional store and no async API beyond what
zarr-python already does internally. Icechunk's asynchronous sessions and
fork/merge add atomic multi-array commits and versioning, not throughput
([When to use Icechunk](../how_to/icechunk.md)).

### Lustre and GPFS

Follow zarr-vectors' HPC guide: stripe the store directory
(`lfs setstripe -c 8`); keep cells ≥ 1–4 MB (the default 32³-brick chunks
give that for vessel trees); use sharding when file counts approach inode
quotas. BRIDGE hit a 1 M inode quota at 933 k files, and sharding cut file
counts 5.7× in our benchmark.
