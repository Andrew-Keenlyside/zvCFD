> [!NOTE]
> This package is an MVP (v0.1): steady and pulsatile flow in voxelised vessels and porous media, on one or several GPUs, benchmarked against OpenFOAM. Multi-GPU runs have been verified for correctness but not yet timed on the H100 node. See the [feasibility study](docs/feasibility/index.md) and [roadmap](docs/feasibility/roadmap.md).

<img src="assets/zvCFD.png" alt="zvCFD" width="60%" />

**GPU-based computational fluid dynamics backed by Zarr Vectors**

`zvcfd` solves flow in image-derived geometries — vessel lumens, porous media, tissue — directly on the voxels of large OME-Zarr volumes, on GPUs, with every input and output held in chunked [Zarr Vectors](https://github.com/AllenInstitute/zarr-vectors-py) stores. There is no meshing step: the segmentation is the domain. Only bricks of the volume that contain fluid are stored, on the GPU and on disk, so memory follows the fluid volume rather than the bounding box. Runs are organised as OME-NGFF RFC-8 collections, and are written in parallel with no locks, following zarr-vectors' three-phase contract.

The target is one node with 8 × H100 GPUs, and domains of 10⁸–10⁹ fluid voxels per node. The HiP-CT coronary tree at 10 µm is estimated at ~1.7 h per cardiac cycle, where a meshed solver at the same cell count needs 7 h to 1.2 days on 1,024 CPU cores ([comparison](docs/benchmarks/comparison.md)).

Measured so far, on one workstation: the solver is second-order accurate against exact solutions (first order on voxel curved walls) and reruns are byte-identical ([validation](docs/validation/index.md)). On the HiP-CT coronary tree at the Ansys mesh's own cell count, one RTX A2000 settles the 77 outlet flow splits 10× sooner than OpenFOAM on 16 cores, at the same inlet pressure within 2.5 % ([OpenFOAM comparison](docs/benchmarks/openfoam.md)).

## What exists

| Module | What it does | Status |
|---|---|---|
| `zvcfd.domain` | sparse 8³-brick domains, neighbour tables, whole-chunk partitioning across GPUs, halo sets | built, tested |
| `zvcfd.lbm` | D3Q19 lattice-Boltzmann, BGK or TRT, Carreau–Yasuda rheology, dense and sparse, fp32/fp16 (cupy RawKernels) | built; 98 % of copy bandwidth dense, 77 % sparse on an RTX A2000 |
| `zvcfd.lbm.MultiLBM` | one partition per GPU, ghost-brick exchange by peer copies | built; bit-identical to one partition |
| `zvcfd.boundary` | pressure, velocity (flow-rate controlled) and RCR patches; exact patch flux | built, tested |
| `zvcfd.geometry` | sparse voxeliser for zoned surface meshes (inlets and outlets become patches) | built, tested |
| `zvcfd.io.fields` | brick stores on `zarr_vectors.building`; parallel writes; GPU reads | built, tested |
| `zvcfd.collection` | RFC-8 run collections, atomic snapshot publication | built, tested |
| `zvcfd.solvers.lubrication` | local-Poiseuille pressure solve, smoothed-aggregation AMG | built, tested |
| `zvcfd.io.fluent_msh` | ASCII Fluent `.msh` boundary reader (a 1.9 GB mesh in 15 s) | built, tested |
| `zvcfd.perfmodel` | roofline memory and time model behind `zvcfd plan` | built |
| `zvcfd.run`, `zvcfd` CLI | runs from YAML: mesh, OME-Zarr, array or phantom → RFC-8 run collection with monitors and per-patch flows | built, tested |

## Install

```bash
pip install -e ./zarr-vectors-py          # the gpu-backend branch; see docs/getting_started/installation.md
pip install -e ".[gpu,all]"                # in conda: install cupy from conda-forge instead of [gpu]
zvcfd probe
```

## Quick start

```bash
zvcfd plan --fluid-cells 6.3e8 --fill 0.7 --gpus 8 --steps 500000
zvcfd mesh-info coronary.msh --voxel-size 0.02,0.01,0.005
zvcfd run examples/network_demo.yaml --out runs/
zvcfd info runs/network-demo-*.zvcfd
```

```python
import zvcfd
from zvcfd.lbm import SparseLBM
from zvcfd.io import fields as zf

domain = zvcfd.BrickDomain.from_flags(flags)            # (z, y, x) uint8: 0 fluid, 1 solid
sim = SparseLBM(domain, tau=1.0, force=(1e-6, 0, 0))
sim.step(2000)
level = zf.create_brick_store("step-2000.zarrvectors", domain, voxel_size=2.5)
zf.write_brick_chunks(level, domain, {k: v.get() for k, v in sim.fields().items()},
                      voxel_size=2.5, chunk_bricks=32)
zf.finalize_brick_store(level)
```

## Documentation

Built with Sphinx (Furo + MyST), in the same structure as zarr-vectors-py:

```bash
pip install -r docs/requirements-docs.txt
sphinx-build -b html docs docs/_build/html
```

- [Feasibility study](docs/feasibility/index.md): the verdict, the measurements behind it, [architecture](docs/feasibility/architecture.md), [Ansys context](docs/feasibility/ansys_context.md), [risks](docs/feasibility/risks.md), [roadmap](docs/feasibility/roadmap.md)
- [Getting started](docs/getting_started/quickstart.md), [concepts](docs/getting_started/concepts.md)
- [Specification](docs/spec/index.md): brick stores, run collections, snapshots, parallel I/O
- [Tutorials](docs/tutorials/first_simulation.md), [how-to guides](docs/how_to/plan_a_run.md)
- [Validation](docs/validation/index.md): exact solutions, standard benchmarks and OpenFOAM, with orders of accuracy and reproducibility
- [Benchmarks](docs/benchmarks/index.md), the measured [OpenFOAM comparison](docs/benchmarks/openfoam.md) and the extrapolated [comparison with other packages](docs/benchmarks/comparison.md)

## Layout

```text
zvcfd/          the package
tests/          pytest; GPU and store tests skip without cupy / zarr-vectors gpu-backend
benchmarks/     kernel, I/O, multiresolution, OpenFOAM and estimate scripts; results/ holds the measurements
examples/       run configurations
docs/           Sphinx sources
```

## Relationship to other packages

zvCFD writes stores only through `zarr_vectors.building`. It reads OME-Zarr with `zarr` directly, as BRIDGE does, and follows BRIDGE's RFC-8 collection conventions. Its command line follows `zvtools` and `bridge-sim`. The solver is clean-room: written from published equations, not from other solvers' source ([policy](docs/how_to/cleanroom.md)).
