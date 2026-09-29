# Risks

What could make the plan fail or slip, how likely it is, and what retires
it. Ordered by how much of the plan each one puts at stake.

---

## Technical risks

### 1. H100 performance differs from the extrapolation

**Likelihood: medium. Impact: high.** Every H100 figure is the A2000
measurement scaled by copy bandwidth (249 GB/s → 3.12 TB/s). The A2000 has
a small L2 cache and few SMs. The H100 has 50 MB of L2 that helps stencils,
but also needs far more parallelism in flight. The dense fp32 kernel reached
98 % of copy bandwidth on the A2000; FluidX3D reaches 80 % on an H100.
Planning numbers already take the lower of the two.

*Retire it:* run `benchmarks/bench_lbm.py` on the H100 node. It takes about
5 minutes (roadmap milestone 0).

### 2. Collision model and stability at blood-flow parameters

**Likelihood: high. Impact: medium.** The built kernel is BGK. Pulsatile
coronary flow at 20 µm needs τ ≈ 0.6 to keep the lattice Mach number below
0.1, and at Reynolds numbers of a few hundred BGK becomes unstable and
inaccurate there. The standard remedies are TRT, regularised or cumulant
collision. All are published, and all cost more arithmetic but few extra
bytes, so they stay memory-bound on an H100.

*Retire it:* TRT first (a small change, fixes the viscosity-dependent wall
position), then cumulant. Validate on Womersley flow in a tube.

### 3. Staircase walls and wall shear stress

**Likelihood: certain. Impact: medium.** Half-way bounce-back puts the wall
on the voxel faces. Flow rate converges at first-to-second order in voxel
size, but wall shear stress is noisy on a staircase. That matters for
coronary hemodynamics.

*Retire it:* interpolated bounce-back (Bouzidi), using sub-voxel wall
distances from the segmentation's signed distance or from the Fluent wall
surface. Report WSS from a smoothed surface (a ZV `mesh` store).

### 4. Large porous media do not fit

**Likelihood: certain for those inputs. Impact: medium.** At porosity 0.2
nearly every brick is active, so the sparse layout buys nothing. A 2048³
sample needs 169 GB per GPU in fp32 and 87 GB in fp16 on eight GPUs.

*Retire it:* a second layout that stores fluid cells only, with a
neighbour-index list (+76 B/cell). Two nodes also work. Or 4³ bricks, which
fill better at a cost in kernel efficiency.

### 5. Output throughput

**Likelihood: medium. Impact: medium.** The zarr-vectors host write path
measured 0.3–0.6 GB/s from eight processes. A 10 µm coronary snapshot
(~17 GB) then takes ~30–40 s, which is free only if snapshots are ≳ 3,000
steps apart.

*Retire it:* overlap output with compute (pinned ring buffer plus writer
threads), write fp16 fields, and encode on the GPU and write whole cells
with `Group.write_cells`. That last one is an upstream request, below.

### 6. RFC-8 changes

**Likelihood: high. Impact: low.** RFC-8 is under review. Its own examples
use `"version": "0.x"`, it replaces the multiscales and labels layouts, and
no library implements it in production yet.

*Retire it:* keep all collection writing in `zvcfd.collection`, as BRIDGE
keeps it in `bridge/collection/`. Follow BRIDGE when it moves.

### 7. The zarr-vectors branch is unreleased

**Likelihood: certain now. Impact: medium.** Brick stores use
`gpu-backend` features (format 0.9.4). `runtime_capabilities()` guards
every one, and `zvcfd probe` reports them.

---

## Upstream requests

These came out of this study. They are all for zarr-vectors-py.

1. **Batch nvCOMP device decode.** `read_cells(device="cuda",
   decode="device")` decoded a 2.4 GB zstd read in one nvCOMP batch and ran
   out of memory on a 12 GB GPU. It should decode in bounded batches.
2. **Faster host decode of wide attribute cells.** Reading brick fields
   ran at ~1.5–2 GB/s warm through `read_cells` on the host path, against
   4.7–6.1 GB/s cold for the same bytes as dense sharded Zarr arrays.
3. **A bulk encoded-cell writer on the building surface.** `write_cells`
   exists on `Group`. What is missing is a documented way to hand it cells
   already encoded on the GPU, so a solver can skip the per-cell Python
   path.
4. **A user-metadata accessor in `building`.** Namespaced metadata is only
   reachable through `zv.open(...).metadata`, so code on the building
   surface has to reopen the store through the facade to set it.
5. **Presence from known keys.** `rebuild_presence` lists every array to
   rebuild `nonempty_chunks`. After an 8-worker snapshot write it took
   2.2–5.1 s, longer than the writes themselves (1.9–2.9 s). The workers
   know exactly which cells they wrote. A coordinator call that applies
   presence from those key lists would roughly halve snapshot time.

---

## Programme risks

- **Validation data.** Hemodynamics needs reference solutions (Womersley,
  a stenosis benchmark, then the coronary case against Fluent on the same
  geometry). The Fluent mesh and its zones make the last one possible, but
  it needs Fluent runs to compare against.
- **Boundary conditions from images.** Inlets and outlets have to be
  located on a segmentation. The Fluent mesh names them (1 inlet, 77
  outlets). A label image of the same scan would too. Without either, they
  must be derived from the centreline graph.
