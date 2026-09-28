# Feasibility study

**Question.** Can the current zarr-vectors-py, with its GPU module, carry a
computational fluid dynamics package that runs on the 8 × H100 node, takes
very large inputs including OME-Zarr volumes, organises its outputs as
OME-NGFF RFC-8 collections, and scales to nearly any size by working in
chunked Zarr Vectors stores? It should be a clean-room Python package with
Python and command-line interfaces that feel like the existing packages,
and it should be fast on large volumes.

**Answer: yes, for image-derived geometries, with three conditions.** The
solver must be a sparse, voxel-native lattice-Boltzmann method (not a
meshed finite-volume solver). Zarr Vectors must be the storage and exchange
layer, not the working memory of the solve. And "nearly any size" means
about 2–4 × 10⁹ fluid voxels per node. Beyond that it means sub-volumes,
fp16 storage or more nodes — not streaming from disk every step.

Everything on this page was measured on an RTX A2000 workstation or taken
from the sources in [References](../references.md). **No H100 run has been
made**; H100 figures are roofline extrapolations from the A2000
measurements and should be confirmed first (see [Roadmap](roadmap.md)).

---

## Answers, one question at a time

| Question | Answer | Evidence |
|---|---|---|
| Can a GPU solver work in chunked ZV stores? | Yes. One ZV vertex per 8³ brick, fields as 512-wide vertex attributes; round-trips exactly, including GPU-side reads | [Brick store](../spec/brick_store.md), tests |
| Is the zarr-vectors GPU module enough? | For reads, mostly: device decode is 2.2× faster warm. Gaps: nvCOMP decode runs out of memory on large reads (no batching); host write path is 0.3–0.6 GB/s | [I/O benchmarks](../benchmarks/io.md) |
| How fast is the kernel? | 96 % of copy bandwidth dense, 78 % sparse (open), 35 % sparse (thin vessels) on the A2000 → **7–16 G fluid-cell updates/s per H100** (planning basis) | [Kernels](../benchmarks/kernels.md) |
| How large a domain fits on 8 × H100? | ~2 × 10⁹ fluid voxels fp32, ~4 × 10⁹ fp16 (fill 0.6) | [Plan a run](../how_to/plan_a_run.md) |
| Do parallel reads and writes work? | Yes: whole chunks (or shards) per worker, presence deferred, one rebuild. 8 writers need no lock | [Parallel I/O](../spec/parallel_io.md) |
| Is Icechunk async I/O needed? | **No.** Same throughput as plain Zarr (0.20 vs 0.13–0.17 GB/s writes, 5.8 vs 4.7–6.1 GB/s reads); its local-filesystem backend is officially unsafe for concurrent commits. Optional for versioned runs on object storage | [Icechunk](../how_to/icechunk.md) |
| Do Ansys-style coarsening and scale give initial speedups? | **Partly.** Coarse *previews* are 7–35× cheaper (25–80 % flux error in thin vessels). Coarse-to-fine *initialisation* of LBM gave 0.6–1.6× — not worth it. *Operator* coarsening (AMG) works: iterations flat at 12–14 as the domain grows 4× | [Multiresolution](../benchmarks/multiresolution.md) |
| RFC-8 collections? | Yes, following BRIDGE's layout. RFC-8 is still a draft under review; expect a version change | [Run collection](../spec/run_collection.md) |
| Speed against common packages? | Real HiP-CT coronary case: at 10 µm (6.3 × 10⁸ fluid voxels) **~1.8 h per cardiac cycle on 8 × H100**, against **~1.2 days on 1,024 CPU cores** for a meshed solver at the same cell count. At Fluent's own mesh size (1.5 × 10⁷ cells) Fluent GPU is competitive | [Comparison](../benchmarks/comparison.md) |
| Python + CLI, consistent API? | Built: `zvcfd` package and CLI in zarr-vectors / bridge-sim style (`probe`, `plan`, `mesh-info`, `run`, `info`) | [API](../api/index.rst) |

## What was built to find out

The investigation produced a working, tested single-GPU core rather than
paper estimates alone:

- `zvcfd.domain.BrickDomain` — sparse bricks, neighbour tables, Morton
  ordering, whole-chunk partitioning across GPUs, halo sets.
- `zvcfd.lbm` — D3Q19 BGK lattice-Boltzmann, dense and sparse-brick,
  fp32/fp16 storage. The sparse kernel is bit-identical to the dense one;
  plane Poiseuille flow matches the analytic profile within 1 %.
- `zvcfd.io.fields` — brick stores on `zarr_vectors.building`, three-phase
  parallel writes, GPU reads through `read_cells`.
- `zvcfd.collection` — RFC-8 run documents, atomic publication of snapshots.
- `zvcfd.solvers.lubrication` — lubrication pressure solve with
  smoothed-aggregation AMG.
- `zvcfd.io.fluent_msh` — Ansys Fluent `.msh` boundary reader. It reads the
  1.9 GB HiP-CT coronary mesh in 15 s.
- `zvcfd.perfmodel` and `zvcfd plan` — the roofline model behind every
  estimate here, calibrated on the measured kernels.
- 38 tests; benchmarks under `benchmarks/` with results in
  `benchmarks/results/`.

## Where it works and where it does not

**Works well:** sparse vascular and airway trees, and any geometry whose
fluid fills a small fraction of its bounding box. The HiP-CT coronary tree
is 0.28 % of its box, so sparse storage is the difference between 4.4 TB
and 22 GB at 20 µm. Also porous media up to about 1536³ per node (dense
layout), and transient flows, where time-marching LBM is the natural
method.

**Works, with care:** steady low-Reynolds flows. LBM reaches steady state
by time-marching, which takes 10⁴–10⁵ steps. Coarse-level initialisation
did not shorten that in our tests. The fast path for steady flow is an
elliptic solve with AMG (built for the lubrication model; a Stokes solver
would be new work).

**Does not work without new design:** porous media larger than ~1536³ at
porosity 0.2 (bricks are nearly all active, so sparsity buys nothing; a
fluid-cell list layout or two nodes is needed). Also whole organs at
cellular resolution (10¹¹–10¹² voxels), where the practical route is a
network model on the centreline graph with 3-D sub-volumes.

## Recommendation

Build it, in the order in [Roadmap](roadmap.md). The first milestone is the
one that retires the largest uncertainty: run `benchmarks/bench_lbm.py` and
`benchmarks/bench_io.py` on the H100 node and replace the extrapolated
figures on these pages with measured ones. The next is the multi-GPU driver
with NVLink halo exchange, then a robust collision model (TRT or cumulant)
and inlet/outlet boundary conditions from Fluent zones or label images, so
the coronary case can run end to end.

Four upstream requests to zarr-vectors-py came out of this study. They are
listed in [Risks](risks.md#upstream-requests).
