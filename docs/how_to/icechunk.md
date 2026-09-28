# When to use Icechunk

**Short answer: not by default.** Icechunk adds transactions and
versioning. It does not add throughput, and parallel writes do not need it.
Its local-filesystem backend is officially unsafe for concurrent commits,
so for zvCFD on a POSIX or Lustre filesystem it adds risk. It is worth it
for versioned, branchable run stores on object storage.

---

## What we measured

Same snapshot, same dense layout, eight writer processes, local NVMe,
[I/O benchmark](../benchmarks/io.md):

| Store | Write (GB/s of payload) | Cold read (GB/s) | Commit |
|---|---:|---:|---:|
| plain Zarr, dense, zstd | 0.13 | 6.1 | — |
| **Icechunk**, dense, zstd (fork → write → merge → commit) | **0.20** | **5.8** | 0.01 s |
| Zarr Vectors bricks, zstd | 0.38 | 2.0 | — |

Icechunk's Rust I/O layer made writes somewhat faster than zarr-python's
for the same arrays, and reads were the same. Its fork/merge commit cost
nothing measurable at this size (4,901 chunks). The bigger differences come
from the layout, not the store: sparse bricks against dense arrays.

## Why async I/O is not the question

The expensive part of writing a snapshot is not waiting on the disk. It is
the host-side codec pipeline (0.1–0.6 GB/s from eight processes, against
several GB/s of NVMe). Overlap solves the rest:
copy fields to pinned memory, keep stepping, write from a background thread
([Parallel I/O](../spec/parallel_io.md#asynchronous-writing)). zarr-python
3 is already asynchronous internally. Icechunk's `*_async` session methods
give the same overlap with transactions on top. Nothing in zvCFD's write
path needs a transaction, because atomic publication comes from the run
collection ([Snapshots](../spec/snapshots.md)).

## What Icechunk costs on a cluster filesystem

From the Icechunk documentation and issue tracker (September 2026, v2.2):

- The local filesystem storage is "not recommended for production" and
  "not safe in the presence of concurrent commits". It warned about exactly
  this on every commit in our benchmark.
- All chunks live in one flat `chunks/` directory. One user hit an NFS
  per-directory limit of 2²¹ files.
- Every rewrite of a chunk creates a new file, and old ones stay until
  snapshots are expired and garbage-collected. Checkpoints that overwrite
  would grow storage without bound.
- Data always passes through host memory: there is no GPUDirect Storage
  path.
- Arrays of ~10⁶ chunks or more need manifest splitting configured.

## When it is worth it

- **Parameter sweeps from one geometry.** Branch a run store per parameter
  set instead of copying the domain.
- **Stores on object storage** (S3, GCS), where Icechunk's commits are safe
  and its request handling is good (> 230 k chunk reads/s reported).
- **Auditable history.** Every snapshot and checkpoint becomes a commit
  you can check out.

## How to use it when you do

zarr-vectors already supports Icechunk as a backend (`backend="icechunk"`
on `create_store`/`open_store`, with `commit`, branches and merges). It has
no fork/merge support for distributed writers yet. With plain Icechunk
arrays the pattern is:

```python
import icechunk, zarr

repo = icechunk.Repository.create(icechunk.s3_storage(bucket="…", prefix="run-001"))
s = repo.writable_session("main")
# ... create groups and arrays ...
s.commit("allocate")                      # fork() refuses a session with changes

session = repo.writable_session("main")
forks = [session.fork() for _ in range(8)]    # picklable; one per worker
# workers: zarr.open_group(fork.store, mode="r+") ... write own chunks ...; return fork
session.merge(*returned_forks)
session.commit("snapshot step 5000")
```

Use the `forkserver` or `spawn` start method: Icechunk recommends it on
Linux, and CUDA requires it.
