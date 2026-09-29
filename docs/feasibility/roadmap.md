# Roadmap

Milestones in the order that retires the most uncertainty first. Each is
useful on its own. Effort is in working weeks for one developer familiar
with the codebase.

---

## Built

### Feasibility study

- [x] Sparse-brick domain, partitioning and halos (`zvcfd.domain`)
- [x] D3Q19 solver, dense and sparse, fp32/fp16, single GPU (`zvcfd.lbm`)
- [x] Brick stores on `zarr_vectors.building`, GPU reads (`zvcfd.io.fields`)
- [x] RFC-8 run collection, atomic snapshot publication (`zvcfd.collection`)
- [x] Lubrication pressure solve with AMG (`zvcfd.solvers.lubrication`)
- [x] Fluent `.msh` boundary reader (`zvcfd.io.fluent_msh`)
- [x] Roofline model and `zvcfd plan`; `probe`, `mesh-info`, `run`, `info`

### MVP (v0.1)

- [x] **TRT collision** (magic Λ = 3/16): half-way walls exact, steady
  solutions independent of τ, so τ is a convergence knob for steady flows
  ([Numerics](../spec/numerics.md) has the τ scan)
- [x] **Shifted populations** (`f − w` stored and computed): fp32 keeps the
  digits that carry a slow flow; steady fluxes no longer drift with τ
- [x] **Carreau–Yasuda rheology** by local relaxation time from the
  non-equilibrium stress (blood parameters from SI with `CarreauYasuda.from_si`)
- [x] **Inlet/outlet patches** on arbitrary geometry: pressure and velocity
  (Guo non-equilibrium extrapolation), plug or parabolic profiles, waveforms
- [x] **Exact patch flux** (populations crossing the patch), used for flow
  splits, mass balance and **flow-rate control** of inlets
- [x] **RCR (Windkessel) outlets**, integrated per check interval
- [x] **Multi-GPU driver** (`MultiLBM`): one partition per device, ghost-brick
  exchange by peer copies, **bit-identical** to one partition
- [x] **Sparse voxeliser** for closed zoned surfaces (`zvcfd.geometry`): the
  HiP-CT coronary Fluent mesh becomes 5.03 × 10⁶ fluid voxels and 78 patches
  at 50 µm in 16 s, within 0.001 % of the mesh volume
- [x] `zvcfd run` from meshes, OME-Zarr, arrays or phantoms, with patch rules,
  Mach-based time step, monitors (`monitors.json`, `patches.csv`); `zvcfd voxelize`
- [x] **Benchmarks against OpenFOAM** on identical voxel geometries and on
  the coronary case ([OpenFOAM comparison](../benchmarks/openfoam.md))

## Milestone 0 — measure on the H100 node (≈ 1 day)

Scripts ready: `envs/zvcfd_sge_job.sh` runs all of these in one job.

- [ ] `zvcfd probe --require cupy,cuda_device,nccl,peer_access,zv_read_cells`
- [ ] `benchmarks/bench_lbm.py` → replace the extrapolated kernel figures
- [ ] `benchmarks/bench_multi.py --devices 0,...,7` → real multi-GPU scaling
- [ ] `benchmarks/bench_io.py` on `$TMPDIR` and on `/SAN` → node I/O and the
  filesystem type
- [ ] the coronary case at 20 µm and 10 µm on 8 GPUs

**Exit:** every H100 number in these pages is measured, not scaled.

## Milestone 1 — multi-GPU performance (≈ 2 weeks)

- [ ] Exchange only the populations that cross each face (today whole ghost
  bricks are copied)
- [ ] Overlap exchange with the interior kernel (two streams per device)
- [ ] Parallel snapshot writes, one writer thread per partition, each owning
  its chunks
- [ ] Asynchronous output: pinned ring buffer, writer threads

**Exit:** weak scaling ≥ 80 % from 1 to 8 H100s at 5 × 10⁸ cells per GPU.

## Milestone 2 — hemodynamics fidelity (≈ 4 weeks)

- [ ] Interpolated bounce-back from sub-voxel wall distance (wall shear stress)
- [ ] Open-loop coronary outlets and closed-loop models through SimVascular's
  svZeroDSolver (BSD-3), coupled every *N* steps via a pybind11 shim
- [ ] Backflow stabilisation at outlets
- [ ] Wall shear stress on a ZV `mesh` store of the lumen surface
- [ ] Womersley validation; svMultiPhysics test problems (`pipe_RCR_3d`,
  `carreau_yasuda`, `casson`); Vascular Model Repository coronary cases
- [ ] Faster steady convergence at low τ (regularised or cumulant collision,
  or acoustic damping), so Navier–Stokes cases converge as fast as Stokes ones

**Exit:** the HiP-CT coronary case matches the CFX solution on the same
geometry (outlet flow splits within 5 %), with the CFX reference using High
Resolution advection, second-order backward Euler, 3–5 coefficient loops per
step and RMS residuals ≤ 10⁻⁵, per the CFX Modeling Guide.

## Milestone 3 — scale-out (≈ 4–6 weeks)

- [ ] Chunked ingest from OME-Zarr (no global dense array); chunked voxeliser
- [ ] Checkpoint/restart with re-partitioning
- [ ] Fluid-cell-list layout for dense porous media (> 1536³)
- [ ] GPU AMG for the lubrication / network pressure solve
- [ ] Network flow on centreline graphs (ZV `graph` stores from BRIDGE
  skeletons), as a 0-D/1-D model in the manner of SimVascular's
  `sv_rom_simulation` and svOneDSolver, coupled to 3-D sub-volumes
- [ ] Multi-node runs (NCCL over InfiniBand)

## Milestone 4 — outputs and viewing (≈ 2 weeks)

- [ ] Streamlines and pathlines as ZV `polyline` stores
- [ ] Field pyramids for viewing (brick averaging), Neuroglancer via zv-ngtools
- [ ] Dense OME-Zarr export of a region; VTK export (`.vtu` fields, `.vtp`
  surfaces and centrelines with SimVascular's array names)

## Not planned

- Unstructured finite volumes. That is Fluent's, CFX's and OpenFOAM's ground.
- Streaming bricks from host or disk every step. PCIe is ~60× slower than
  HBM (see [Architecture](architecture.md#scaling-out)).
- Coarse-to-fine initialisation of time-marching LBM. It was measured as
  neutral to harmful.
