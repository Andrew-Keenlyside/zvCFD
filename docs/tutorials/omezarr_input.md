# Solving on an OME-Zarr volume

This tutorial takes a segmentation stored as an OME-Zarr multiscale image,
picks a pyramid level, and runs on it. To be self-contained it first writes
a small OME-Zarr 0.5 label pyramid from a phantom; with real data, start at
[Pick a level](#pick-a-level).

zvCFD reads OME-Zarr with `zarr` directly, as BRIDGE does. It accepts the
flat `multiscales` block of NGFF 0.4 and the nested `ome` block of 0.5 and
0.6, and always resolves `datasets[level].path`.

---

## Write a test pyramid

```python
import numpy as np
import zarr
from zvcfd.phantoms import coarsen_mask, vessel_network

fluid = vessel_network((128, 128, 512))                 # (z, y, x) booleans
levels = [fluid, coarsen_mask(fluid, "any"), coarsen_mask(coarsen_mask(fluid, "any"), "any")]

root = zarr.open_group("vessels.ome.zarr", mode="w")
datasets = []
for k, m in enumerate(levels):
    a = root.create_array(str(k), shape=m.shape, dtype="uint8", chunks=(64, 64, 64),
                          dimension_names=("z", "y", "x"))
    a[:] = m.astype("uint8")                            # label 1 = lumen
    s = 5.0 * 2 ** k                                    # micrometre
    datasets.append({"path": str(k), "coordinateTransformations": [
        {"type": "scale", "scale": [s, s, s]}]})
root.attrs["ome"] = {"version": "0.5", "multiscales": [{
    "axes": [{"name": n, "type": "space", "unit": "micrometer"} for n in "zyx"],
    "datasets": datasets}]}
print([root[str(k)].shape for k in range(3)])
```

```text
[(128, 128, 512), (64, 64, 256), (32, 32, 128)]
```

## Pick a level

```python
from zvcfd.io.omezarr import open_multiscale

ms = open_multiscale("vessels.ome.zarr")
for k in range(len(ms.datasets)):
    print(k, ms.level_path(k), ms.voxel_size(k), ms.unit())
print("nearest to 10 um:", ms.nearest_level(10.0))
```

```text
0 0 (5.0, 5.0, 5.0) micrometer
1 1 (10.0, 10.0, 10.0) micrometer
2 2 (20.0, 20.0, 20.0) micrometer
nearest to 10 um: 1
```

Choose the finest level whose domain fits (`zvcfd plan`), and which
resolves the narrowest channel you care about by at least ~6 voxels. A
coarser level is a *preview*: 8–43× cheaper in our tests, but tens of
percent off in thin vessels ([Multiresolution](../spec/multiresolution.md)).

## Run on it

`vessels.yaml`:

```yaml
name: vessels-level1
source: {kind: omezarr, path: vessels.ome.zarr, level: 1, label: 1}
physics: {nu: 3.5e-6, rho: 1060.0, pressure_drop: 20.0}
solver: {method: lbm, tau: 0.6, steps: 20000, check_every: 1000, tolerance: 1.0e-4}
domain: {chunk_bricks: 16}
output: {path: runs, every: 0}
```

```bash
zvcfd run vessels.yaml
```

```text
domain (64, 64, 256): 989 bricks (48.29% active), 173,373 fluid cells, fill 0.34; 2 patches; tau = 0.6000, dt = 9.52e-07 s  (0.7 s)
  step     1000  inflow 3.6218e-11 m3/s  imbalance +1.89e-01  change inf
  ...
  step     5000  inflow 3.2901e-11 m3/s  imbalance -3.20e-05  change 1.83e-05
  converged: change <= 0.0001, |imbalance| <= 0.001
  wrote step-000005000.zarrvectors
done: 5000 steps, 564 MLUPS incl. checks/output, 3.2 s total -> runs/vessels-level1-135c6fd6440b.zvcfd
```

The voxel size comes from the level's `scale` transform (10 µm), unless
`source.voxel_size` overrides it. `source.region` crops a sub-volume
(`[[z0, z1], [y0, y1], [x0, x1]]` in the level's voxels). The run
collection's `image` node points back at `vessels.ome.zarr` by relative
path; the image is never copied.

## Large volumes

`zvcfd run` currently reads the chosen level (or region) into host memory
in one piece. For volumes larger than host memory, the chunked ingest
([Architecture](../feasibility/architecture.md#1-ingest-image-to-domain-built-for-one-process-chunked-version-designed))
reads one store chunk plus a one-voxel halo at a time, and never holds the
whole mask. It is on the roadmap (milestone 3).
