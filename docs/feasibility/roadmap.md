# Roadmap

Milestones in the order that retires the most uncertainty first. Each is
useful on its own. Effort is in working weeks for one developer familiar
with the codebase.

---

## Built (this study)

- [x] Sparse-brick domain, partitioning and halos (`zvcfd.domain`)
- [x] D3Q19 BGK solver, dense and sparse, fp32/fp16, single GPU (`zvcfd.lbm`)
- [x] Brick stores on `zarr_vectors.building`, GPU reads (`zvcfd.io.fields`)
- [x] RFC-8 run collection, atomic snapshot publication (`zvcfd.collection`)
- [x] Lubrication pressure solve with AMG (`zvcfd.solvers.lubrication`)
- [x] Fluent `.msh` boundary reader (`zvcfd.io.fluent_msh`)
- [x] Roofline model and `zvcfd plan`; `probe`, `mesh-info`, `run`, `info`
- [x] Benchmarks and results for kernels, I/O and multiresolution

## Milestone 0 — measure on the H100 node (≈ 1 day)

- [ ] `zvcfd probe --require cupy,cuda_device,nccl,peer_access,zv_read_cells`
  on `seymour2` inside the Apptainer image
- [ ] `benchmarks/bench_lbm.py` → replace the extrapolated kernel figures
- [ ] `benchmarks/bench_io.py --root $TMPDIR` and on `/SAN` → real node I/O,
  and the filesystem type (GDS eligibility)

**Exit:** every H100 number in these pages is measured, not scaled.

## Milestone 1 — multi-GPU driver (≈ 3–4 weeks)

- [ ] Coordinator plus one spawned worker per GPU (the `run_nccl_ranks`
  pattern from BRIDGE-Simulation)
- [ ] Ghost-brick exchange: pack only crossing populations; NCCL send/recv
  or `cudaMemcpyPeerAsync`; interior/boundary split on two streams
- [ ] Bit-identity test: an N-GPU run equals the 1-GPU run, as
  BRIDGE-Simulation's halo plan guarantees for its graphs
- [ ] Asynchronous output: pinned ring buffer, writer thread pool per GPU,
  coordinator publication

**Exit:** weak scaling ≥ 80 % from 1 to 8 H100s at 5 × 10⁸ cells per GPU.

## Milestone 2 — physics for hemodynamics (≈ 4–6 weeks)

- [ ] TRT collision, then cumulant; Womersley validation
- [ ] Inlet/outlet boundaries: velocity (Zou–He / regularised) and pressure,
  on arbitrary oriented patches; zone mapping from Fluent `.msh` and from
  label images
- [ ] Interpolated bounce-back from sub-voxel wall distance
- [ ] Windkessel (RCR) outlets for the 77 coronary outlets
- [ ] Wall shear stress on a ZV `mesh` store of the lumen surface

**Exit:** the HiP-CT coronary case runs at 20 µm and 10 µm, with outlet
flow splits within 5 % of the Fluent solution on the same geometry.

## Milestone 3 — scale-out and steady flows (≈ 4–6 weeks)

- [ ] Chunked ingest from OME-Zarr (no global dense array)
- [ ] Checkpoint/restart with re-partitioning
- [ ] Fluid-cell-list layout for dense porous media (> 1536³)
- [ ] GPU AMG for the lubrication / network pressure solve (AmgX or a
  smoothed-aggregation implementation in cupy)
- [ ] Network flow on centreline graphs (ZV `graph` stores from BRIDGE or
  VesselVio-style skeletons), coupled to 3-D sub-volumes

**Exit:** a 5 µm coronary run in fp16 on one node, and a whole-organ network
solve feeding sub-volume boundary conditions.

## Milestone 4 — outputs and viewing (≈ 2 weeks)

- [ ] Streamlines and pathlines as ZV `polyline` stores
- [ ] Field pyramids for viewing (brick averaging), Neuroglancer via
  zv-ngtools
- [ ] Dense OME-Zarr export of a region, for tools that want images

## Not planned

- Unstructured finite volumes. That is Fluent's and OpenFOAM's ground.
- Streaming bricks from host or disk every step. PCIe is ~60× slower than
  HBM (see [Architecture](architecture.md#scaling-out)).
- Coarse-to-fine initialisation of time-marching LBM. It was measured as
  neutral to harmful.
