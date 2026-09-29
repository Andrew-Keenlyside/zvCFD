# Reading results

This tutorial opens a finished run, finds its snapshots through the run
collection, reads fields back — on the CPU and on the GPU — converts them
to physical units, and makes a dense image of a region for tools that want
one. It continues from [Your first simulation](first_simulation.md).

---

## Find the snapshots

```python
from zvcfd import collection

run = "runs/network-demo-d78037f7f963.zvcfd"
doc = collection.read(run)
snaps = collection.snapshots(doc)
print([(s["id"], s["attributes"]["zvcfd:time_s"]) for s in snaps])
print(doc["attributes"]["zvcfd:run"])
```

```text
[('step-000005000', 0.00476190476190476), ('step-000008000', 0.0076190476190476164)]
{'config_hash': 'd78037f7f963', 'voxel_size_um': 10.0, 'dt_s': 9.52380952380952e-07, 'tau': 0.6, 'solver': 'lbm'}
```

Paths in the document are relative to the run directory.

## Rebuild the domain

The domain store holds each brick's flags; `read_domain` rebuilds the
`BrickDomain` from it without ever making a dense volume.

```python
from zvcfd.io import fields as zf

domain = zf.read_domain(f"{run}/domain.zarrvectors")
meta = zf.store_metadata(f"{run}/domain.zarrvectors")
print(domain.n_bricks, domain.fluid_cells, meta["voxel_size"], meta["unit"])
```

```text
4989 1192679 10.0 micrometer
```

## Read a snapshot

`read_snapshot` reads every brick of a snapshot in one batched read and
returns each field as `(n_bricks, 512)` rows in the domain's order.

```python
import numpy as np

snap = f"{run}/{snaps[-1]['path']['path']}"
f = zf.read_snapshot(snap, domain)
print(sorted(f), f["ux"].shape, float(np.abs(f["ux"]).max()))
```

```text
['rho', 'ux', 'uy', 'uz'] (4989, 512) 0.00030803700792603195
```

With `device="cuda"` the same call returns cupy arrays, decoded on the GPU
where the store's codec allows it:

```python
g = zf.read_snapshot(snap, domain, ["ux"], device="cuda")
print(type(g["ux"]))
```

```text
<class 'cupy.ndarray'>
```

## Physical units

```python
from zvcfd import Lattice

lat = Lattice(dx=meta["voxel_size"] * 1e-6, nu=3.5e-6, tau=0.6, rho=1060.0)
print(f"peak velocity {lat.velocity(float(np.abs(f['ux']).max())) * 1e3:.2f} mm/s")
```

```text
peak velocity 3.23 mm/s
```

Density maps to pressure relative to the reference density:
`lat.pressure(rho - 1)` in pascal.

## A dense image of a region

```python
vol = domain.to_dense(f["ux"])                          # (z, y, x), zero outside the fluid
print(vol.shape, vol.dtype)
```

```text
(128, 128, 512) float32
```

From here, `zarr` writes it as an OME-Zarr image for any viewer. For a
large run, crop to a region first: select the bricks whose coordinates fall
inside it, and scatter only those.
