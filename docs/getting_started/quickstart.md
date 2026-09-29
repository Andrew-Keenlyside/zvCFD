# Quickstart

This page covers the core loop of `zvcfd`: build a sparse domain from a
voxel geometry, size the run, solve it on the GPU, write the result to a
Zarr Vectors brick store, read it back onto the GPU, publish it in a run
collection, and get a seconds-scale pressure estimate from the lubrication
solver. It needs the base install plus the `gpu` extra, and a CUDA device.

The page is one continuous session — later blocks reuse what earlier blocks
create. Outputs are from an RTX A2000 workstation.

---

## Build a domain

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

## Size the run

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

## Solve

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

## Write a snapshot

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

## Read it back onto the GPU

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

## Publish it in a run collection

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

## A seconds-scale pressure estimate

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

## The same from the command line

```bash
zvcfd plan --fluid-cells 9.4e5 --fill 0.45 --gpus 1 --gpu RTX-A2000 --geometry porous --steps 2000
zvcfd run examples/network_demo.yaml --out runs/
zvcfd info runs/network-demo-*.zvcfd
```

`zvcfd run` does everything above from one configuration file, and writes
the domain, the configuration and each snapshot into the run collection.
See [Your first simulation](../tutorials/first_simulation.md).
