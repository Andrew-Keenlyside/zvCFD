# I/O

**Questions.** How fast can eight workers (one per GPU) write a snapshot
and read it back? Do Zarr Vectors brick stores beat dense OME-Zarr-style
arrays for sparse domains? Does Icechunk help? How much does GPU-side
decode buy?

**Method.** `benchmarks/bench_io.py` builds a sparse domain: 1536³ voxels,
289,075 active 8³ bricks (4.1 % of the box), 1,224 store chunks of 16³
bricks. Eight processes (`forkserver`) each write the chunks they own
(`BrickDomain.partition`) of one snapshot, four float32 fields and 2.37 GB
of payload. Timing starts at a barrier after the data is prepared. Dense
blocks for the dense variants are built before the barrier: in a real run
that densification would happen on the GPU. Then one `sync()` flushes to
disk (timed), the store's pages are dropped from the cache, and eight
processes read their chunks back cold. Local NVMe (ext4), 32 threads.

```{figure} ../_static/figures/io_throughput.png
:width: 100%
:figclass: zv-figure

**Parallel snapshot write and read.** Eight processes; GB/s of payload
(2.37 GB). Write includes the coordinator's finalise step.
```

## Results

| Variant | Write GB/s (incl. finalise) | of which finalise | To disk GB/s | Cold read GB/s | Files | On disk |
|---|---:|---:|---:|---:|---:|---:|
| ZV bricks, flat, raw | **0.57** | 2.2 s | 0.47 | 2.12 | 7,353 | 2.37 GB |
| ZV bricks, flat, zstd | 0.38 | 3.8 s | 0.33 | 1.95 | 7,353 | 1.89 GB |
| ZV bricks, 2³-cell shards, zstd | 0.30 | 5.1 s | 0.28 | 2.05 | **1,281** | 1.89 GB |
| dense Zarr (64³ chunks, 128³ shards), raw | 0.17 | — | 0.12 | 4.67 | 4,901 | **21.0 GB** |
| dense Zarr, zstd | 0.13 | — | 0.13 | **6.13** | 4,901 | 1.95 GB |
| Icechunk, dense, zstd (fork → merge → commit) | 0.20 | 0.01 s | 0.19 | 5.80 | 4,933 | 1.95 GB |
| *one writer:* ZV bricks, flat, zstd | 0.28 | 4.2 s | 0.26 | 0.44 | | |
| *one writer:* dense Zarr, zstd | 0.03 | — | 0.03 | 1.80 | | |

## Reading the results

**Sparse storage wins on writes and on disk.** At 4 % brick occupancy,
dense 64³ chunks are mostly partial, so the dense arrays hold 9× the
payload in raw bytes (21 GB). Writing that is 2–4× slower. With zstd the
zeros compress away (1.95 GB, like the bricks), but the bytes still have to
be encoded.

**Dense Zarr reads faster per payload byte.** zarr-python's sharded read
pipeline decodes 64³ chunks at 4.7–6.1 GB/s. The zarr-vectors host path
reads brick cells at ~2 GB/s. The cost is in variable-length cell handling
on the host, and it is upstream request 2 in [Risks](../feasibility/risks.md#upstream-requests).

**Finalising dominates the brick writes.** The eight workers finish their
cells in 1.9–2.9 s (0.8–1.2 GB/s of payload). The coordinator's
`rebuild_presence`, which lists every array's cells to rebuild
`nonempty_chunks`, then takes 2.2–5.1 s, longer on the sharded store. The
workers already know which cells they wrote. Applying presence from their
lists instead of a directory listing would roughly halve snapshot time.
That is upstream request 5.

**Parallelism helps less than it should.** One zarr-vectors writer
already flushes a write session through a 32-thread pool, so eight
processes reach only 1.4× one (dense Zarr: 4.3×). The ceiling is
per-cell Python and codec work on the host, not the disk. Syncing to the
NVMe took under 1 s for most variants.

**Sharding cuts files 5.7× for ~20 % of write rate.** Use it when inode
quotas bind (BRIDGE hit one at 933 k files) or for object storage.

**Icechunk matches plain Zarr.** It was slightly faster on writes (Rust
I/O), the same on reads, and its fork/merge commit was free at this size.
It is not needed for throughput ([When to use Icechunk](../how_to/icechunk.md)).

## GPU reads

`benchmarks/bench_gpu_read.py` reads the whole snapshot (1,224 chunks,
four fields, 2.37 GB) back onto the GPU with one `read_cells` call from one
process.

| Store | Decode | Cold | Warm (page cache) |
|---|---|---:|---:|
| flat, raw | host, then one upload | 0.65 GB/s | 1.70 GB/s |
| flat, raw | **device** (kernels unframe cells on the GPU) | 0.72 GB/s | **3.36 GB/s** |
| 2³ shards, zstd | host | 0.69 GB/s | 0.73 GB/s |
| 2³ shards, zstd | device + nvCOMP | **fails: out of GPU memory** | fails |

Device decode doubles warm reads. Cold, one process is bound by reading
7,353 files, and the decode path doesn't matter. nvCOMP decode of zstd
cells ran out of memory on this 12 GB GPU: zarr-vectors decodes all 2.4 GB
of frames in one nvCOMP batch. It needs bounded batches (upstream
request 1). On an 80 GB H100 this read would fit, but a restart read of
the populations (19 values per cell, 10× these fields) would not.

## What it means for a run

A 10 µm coronary snapshot is ~17 GB (four float32 fields, 1.05 × 10⁹ stored
cells). At the measured 0.4–0.6 GB/s it takes 30–40 s. At the worker-only
rate, with presence applied from worker lists, it takes ~15–20 s. At ~12 ms
per step (8 × H100), overlapped output keeps up if snapshots are more than
~1,500–3,000 steps apart. A cardiac cycle is 5 × 10⁵ steps at 10 µm, so
100–300 snapshots per cycle are affordable. fp16 fields halve every figure.
