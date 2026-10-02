# Quickstart

zvCFD has two solvers. The finite-volume solver, the main one, solves on a
mesh's own cells in the manner of Ansys CFX. The lattice-Boltzmann solver
solves on voxels, straight from an image. This page runs both. It needs
the base install plus the `gpu` extra and a CUDA device, and the
finite-volume solver uses AmgX if it is built (`ZVCFD_AMGX_LIB`).

Outputs are from an RTX A2000 workstation.

---

## On a mesh: the finite-volume solver

### A mesh

Any Ansys Fluent `.msh`, VTK `.vtu`, SimVascular `mesh-complete` folder
or meshio format will do. Here zvCFD's own generator builds a pipe of 1 mm
radius and 6 mm length (three prism layers at the wall around a
tetrahedral core) and writes it as a Fluent mesh in millimetres:

```python
from zvcfd.mesh.fluent import write_fluent_mesh
from zvcfd.mesh.generate import tube

write_fluent_mesh(tube(1.0, 6.0, n_core=6, n_ring=6, n_axial=24, kind="mixed", layers=3,
                       growth=0.8), "pipe.msh")
```

### Run it

`examples/pipe_fv.yaml` drives it with a flow-rate inlet carrying a
cardiac-like waveform and an RCR outlet, for blood. Its `source` is the
mesh. `solver.method` picks the finite-volume solver for a mesh, so no
voxel size is needed:

```yaml
name: pipe-fv
source: {kind: mesh, path: "pipe.msh", unit: mm}
physics: {nu: 3.5e-6, rho: 1060.0}
boundaries:
  patches:
    - {match: inlet, kind: velocity, flow_rate: 2.0e-7, profile: parabolic,
       waveform: [[0.0, 1.0], [0.15, 2.2], [0.4, 1.0], [0.8, 0.8], [1.0, 1.0]]}
    - {match: outlet, kind: rcr, rcr: [1.0e8, 1.0e-10, 4.0e8], pressure: 1000.0}
solver: {method: fv}
fv: {linear: auto, iterations: 300, tolerance: 1.0e-8, dt: 0.01, periods: 1.0, loops: 5,
     loop_tolerance: 1.0e-6}
output: {path: runs, every: 25}
```

```bash
zvcfd run examples/pipe_fv.yaml --out runs/
```

```text
importing pipe.msh into runs/meshes/pipe-90a0309e15.zvmesh
read pipe.msh: 4,825 nodes, {'tet': 15552, 'wedge': 3456}, 3 zones (0.1 s)
wrote runs/meshes/pipe-90a0309e15.zvmesh (2.9 s)
read pipe-90a0309e15.zvmesh (Zarr Vectors, 1.5 mm chunks): 1.2 s
mesh 4,825 nodes, {'tet': 15552, 'wedge': 3456}; zones inlet=velocity, outlet=rcr, wall=wall  (4.2 s)
solver set up: linear auto, 1.2 s
  it    1  rms u 3.31e-14 p 6.76e-02  du 9.68e-01 dp 1.39e+00  lin 5 (4.3e-02)  imb -1.4e-02  0.23s
  ...
  steady: 29 outer iterations, converged True, 1.6 s
  wrote step-000000000.zarr
  ...
  transient: 100 steps of 0.01 s, 500 coefficient loops, 20.5 s
  TAWSS/OSI/RRT over 0.99 s -> boundary.zarrvectors
done: 27.9 s -> runs/pipe-fv-8b32d5eb1a2b.zvcfd
```

The run starts by importing the mesh into a Zarr Vectors mesh collection,
and the solver works from the mesh read back from it
([Meshes in Zarr Vectors](../tutorials/zarr_vectors_meshes.md)). A steady
start converges in 29 outer iterations, each one coupled linear solve of
velocity and pressure. Then one cycle runs in 100 BDF2 steps, with the
RCR outlet coupled implicitly. Every patch's flow and mean pressure at the
end are in `patches.csv`:

```text
patch,kind,faces,area_m2,flow_m3s,split,pressure_pa
inlet,velocity,288,3.105828541230249e-06,2.0000000000000105e-07,,1108.9898850813727
outlet,rcr,288,3.105828541230249e-06,-2.000003690378799e-07,1.0,1096.831621108008
```

```bash
zvcfd info runs/pipe-fv-*.zvcfd
```

```text
run pipe-fv (zvc-run-bf505000d976)
  zvcfd:mesh       mesh-collection ../meshes/pipe-90a0309e15.zvmesh
  zvcfd:fv-mesh    mesh         ../meshes/pipe-90a0309e15.zvmesh/volume.zarrvectors
  zvcfd:fv-boundary boundary     ./boundary.zarrvectors
  zvcfd:config     config       ./config.json
  collection       fields
  5 snapshots: step-000000000 .. step-000000100
```

The snapshots are node fields on the collection's volume store. TAWSS, OSI
and RRT over the cycle are vertex attributes of the run's boundary store.
[Solving on the mesh itself](../tutorials/fv_mesh.md) reads them back and
goes further.

---

## On voxels: the lattice-Boltzmann solver

The rest of the page covers the voxel solver's core loop: build a sparse
domain from a voxel geometry, size the run, solve it on the GPU, write the
result to a Zarr Vectors brick store, read it back onto the GPU, publish
it in a run collection, and get a seconds-scale pressure estimate from the
lubrication solver. It is one continuous session, so later blocks reuse
what earlier blocks create.

### Build a domain

A domain is the set of 8³ **bricks** that hold at least one fluid voxel.
Nothing else is stored, on the GPU or on disk. Flags are `0` fluid, `1`
solid, `2` fixed-density boundary, with axes in OME-Zarr order `(z, y, x)`.

```python
import numpy as np
import zvcfd
from zvcfd.phantoms import porous_spheres

flags = porous_spheres(128, porosity_target=0.45, radius=6)   # 0 fluid, 1 solid; (z, y, x)
domain = zvcfd.BrickDomain.from_flags(flags, periodic=True)
print(domain.summary())
```

```text
{'shape': (128, 128, 128), 'brick': 8, 'bricks': 4069, 'active_fraction': 0.993408203125, 'fluid_cells': 943237, 'stored_cells': 2083328, 'fill': 0.4527549190525928}
```

A porous medium touches almost every brick (`active_fraction` 0.99), so a
sparse layout saves little here; a vessel tree is the opposite case (see
[Concepts](concepts.md#bricks)).

### Size the run

`zvcfd.estimate` is the roofline model behind `zvcfd plan`: device memory,
and time from the memory bandwidth and a measured kernel efficiency.

```python
plan = zvcfd.estimate(domain.fluid_cells, fill=domain.fill, gpus=1, gpu="RTX-A2000",
                      geometry="porous", steps=2000)
print(f"{plan.memory_per_gpu_gb:.2f} GB, {plan.updates_per_s / 1e6:.0f} MLUPS, "
      f"{plan.seconds:.1f} s for 2000 steps")
```

```text
0.33 GB, 684 MLUPS, 2.8 s for 2000 steps
```

### Solve

`SparseLBM` is the D3Q19 lattice-Boltzmann solver over the active bricks:
one CUDA block per brick, one thread per voxel, half-way bounce-back walls.
Here a body force drives flow through the periodic sample.

```python
import time
import cupy as cp
from zvcfd.lbm import SparseLBM

sim = SparseLBM(domain, tau=1.0, force=(1e-6, 0.0, 0.0))    # force in (x, y, z), lattice units
t = time.time()
sim.step(2000)
cp.cuda.Device().synchronize()
f = sim.fields()                       # {"rho", "ux", "uy", "uz"}: (n_bricks, 512) on the GPU
fluid = sim.flag.reshape(domain.n_bricks, 512) == 0
print(f"{sim.steps} steps in {time.time() - t:.1f} s; "
      f"mean u_x = {float(f['ux'][fluid].mean()):.3e} (lattice)")
```

```text
2000 steps in 2.9 s; mean u_x = 7.886e-06 (lattice)
```

The plan's 2.8 s and the measured 2.9 s agree because the model's kernel
efficiency was measured on this GPU; on an H100 the model extrapolates by
bandwidth (see [Kernel benchmarks](../benchmarks/kernels.md)).

### Write a snapshot

A **brick store** is an ordinary Zarr Vectors store in which one vertex is
one brick: `vertices` holds brick centres and each field is a vertex
attribute 512 values wide. Writing follows zarr-vectors' three-phase
contract — the coordinator creates, workers write the chunks they own, the
coordinator finalises — which here is one process playing both parts.

```python
from zvcfd.io import fields as zf

path = "step-000002000.zarrvectors"
level = zf.create_brick_store(path, domain, voxel_size=2.5, chunk_bricks=8,
                              fields=["rho", "ux", "uy", "uz"], compressor="zstd")
n = zf.write_brick_chunks(level, domain, {k: cp.asnumpy(v) for k, v in f.items()},
                          voxel_size=2.5, chunk_bricks=8)
zf.finalize_brick_store(level)
print(f"wrote {n} chunk cells")
```

```text
wrote 8 chunk cells
```

### Read it back onto the GPU

`read_brick_chunks` is one `zarr_vectors.building.read_cells` call. With
`device="cuda"` the cells are decoded on the GPU where the store allows it
(uncompressed cells always; zstd cells through nvCOMP with
`decode="device"`), and the result is a CSR column over the chunks read.

```python
level = zf.open_brick_level(path)
chunks, _ = domain.chunks(8)
batch = zf.read_brick_chunks(level, chunks, ["ux"], device="cuda")
col = batch["vertex_attributes/ux"]
print(type(col.data).__module__, col.data.shape, col.offsets.shape)
```

```text
cupy (4069, 512) (9,)
```

### Publish it in a run collection

A run is an OME-NGFF RFC-8 collection: a Zarr group whose metadata lists
the input image, the domain and every snapshot by relative path. A snapshot
is visible once its node is in the document, and the document is only ever
replaced atomically, so readers never see half a snapshot.

```python
from zvcfd import collection

doc = collection.run_document("demo.zvcfd", name="demo")
collection.add_snapshot(doc, "demo.zvcfd", path, step=sim.steps, fields=["rho", "ux", "uy", "uz"])
collection.write("demo.zvcfd", doc)
print([n["id"] for n in collection.snapshots(collection.read("demo.zvcfd"))])
```

```text
['step-000002000']
```

### A seconds-scale pressure estimate

The lubrication solver treats every fluid voxel as part of a local
Poiseuille channel (conductance ¾d², *d* the distance to the wall) and
solves one elliptic equation with algebraic multigrid. It is a coarse
estimator — 17 % low on this 6-voxel-radius tube — but it scales to masks
far larger than any 3-D flow solve, and supplies boundary pressures for
sub-volume runs.

```python
from zvcfd.solvers import lubrication

z, y = np.mgrid[:32, :32]
tube = np.repeat(((z - 15.5) ** 2 + (y - 15.5) ** 2 <= 36)[:, :, None], 64, axis=2)
fixed = np.zeros_like(tube)
fixed[:, :, [0, -1]] = True
p = np.zeros(tube.shape)
p[:, :, 0] = 1.0
r = lubrication.solve(tube, fixed & tube, p)
print(f"{r.method}: {r.iterations} iterations, flux {r.flux(axis=2, index=32):.2f} "
      f"(Poiseuille {np.pi * (tube[:, :, 0].sum() / np.pi) ** 2 / 8 / 63:.2f})")
```

```text
amg: 7 iterations, flux 6.56 (Poiseuille 7.92)
```

---

### The same from the command line

```bash
zvcfd plan --fluid-cells 9.4e5 --fill 0.45 --gpus 1 --gpu RTX-A2000 --geometry porous --steps 2000
zvcfd run examples/network_demo.yaml --out runs/
zvcfd info runs/network-demo-*.zvcfd
```

`zvcfd run` does everything above from one configuration file, and writes
the domain, the configuration and each snapshot into the run collection.
See [Your first simulation](../tutorials/first_simulation.md).
