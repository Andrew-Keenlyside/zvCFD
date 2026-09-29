<p align="center">
  <img src="assets/zvCFD.png" alt="zvCFD: GPU-based computational fluid dynamics backed by Zarr Vectors" width="520">
</p>

# zvCFD

**GPU-based computational fluid dynamics, backed by Zarr Vectors.**

zvCFD solves flow in image-derived geometries, such as vessel lumens and
porous media, directly on the voxels of large OME-Zarr volumes, on one or
several GPUs. There is no meshing step. Only the bricks that contain fluid
are stored, on the GPU and in chunked
[Zarr Vectors](https://github.com/AllenInstitute/zarr-vectors-py) stores, and
each run is an OME-NGFF RFC-8 collection.

📖 **Documentation: [zvcfd.readthedocs.io](https://zvcfd.readthedocs.io/en/latest/index.html)**

> [!NOTE]
> **Status: MVP (v0.1).** Validated against exact solutions and standard
> benchmarks ([validation](https://zvcfd.readthedocs.io/en/latest/validation/index.html)).
> On the HiP-CT coronary tree, one RTX A2000 settles the 77 outlet flow
> splits 10× sooner than OpenFOAM on 16 cores. Multi-GPU runs are verified
> but not yet timed on 8 × H100.

## Install

```bash
pip install -e ./zarr-vectors-py      # its gpu-backend branch
pip install -e ".[gpu,all]"           # in conda, install cupy from conda-forge instead of [gpu]
zvcfd probe
```

See [Installation](https://zvcfd.readthedocs.io/en/latest/getting_started/installation.html).

## Quick start

```bash
zvcfd run examples/network_demo.yaml --out runs/
zvcfd info runs/network-demo-*.zvcfd
```

Next: the [tutorials](https://zvcfd.readthedocs.io/en/latest/tutorials/index.html),
including [running a Fluent mesh](https://zvcfd.readthedocs.io/en/latest/tutorials/ansys_mesh.html),
and the [benchmarks](https://zvcfd.readthedocs.io/en/latest/benchmarks/index.html).
