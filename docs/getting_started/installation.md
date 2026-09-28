# Installation

`zvcfd` requires Python 3.11 or later. Its mandatory dependencies are
`numpy`, `scipy`, `zarr>=3.0` and `zarr-vectors`; everything that touches
a GPU is an optional extra, imported only when used, so `import zvcfd` works
on a machine with no GPU.

```{admonition} zarr-vectors branch
:class: warning

Brick stores use `read_cells`, `defer_presence`, `rebuild_presence` and
device decode, which are on the zarr-vectors-py **`gpu-backend`** branch
(format 0.9.4) and in no release yet. Install that branch editable until it
is released. `zvcfd probe` reports exactly which zarr-vectors capabilities
are present (the `zv_*` keys).
```

## Standard install

```bash
git clone https://github.com/AllenInstitute/zarr-vectors-py
git -C zarr-vectors-py checkout gpu-backend
pip install -e ./zarr-vectors-py
pip install -e "./zvCFD[all]"
```

This gives the domain builder, the Fluent mesh reader, the lubrication
solver (with algebraic multigrid through `pyamg`), the run collection and
the `zvcfd` command line. The lattice-Boltzmann solver needs the GPU extra.

## Optional extras

### GPU solvers

```bash
pip install "zvcfd[gpu]"          # cupy-cuda12x
```

In a conda environment install cupy from conda-forge instead, so there is
one cupy, exactly as zarr-vectors advises:

```bash
conda install -c conda-forge cupy
```

### GPU I/O and codecs

These are zarr-vectors' extras, re-exported under the same names:

```bash
pip install "zvcfd[gpu-codecs]"   # nvCOMP: zstd cells decoded on the device
pip install "zvcfd[gpu-io]"       # kvikio: GPUDirect Storage reads (pip environments)
```

In conda, install kvikio from the `rapidsai` channel at the version
matching your RAPIDS stack.

### Icechunk

```bash
pip install "zvcfd[icechunk]"     # Python 3.12+
```

Optional, and not needed for parallel writes or throughput; see
[When to use Icechunk](../how_to/icechunk.md).

### Development and documentation

```bash
pip install -e ".[dev]"
pip install -r docs/requirements-docs.txt
sphinx-build -b html docs docs/_build/html
```

## Checking an install

```bash
zvcfd probe
```

```text
cupy                     yes
kvikio                   yes
nvcomp                   yes
pyamg                    yes
icechunk                 no
zarr_vectors             yes
nccl                     yes
cuda_device              yes
peer_access              no
zv_read_cells            yes
zv_device_decode         yes
zv_gpu_codecs            yes
zv_gpu_io                yes
zv_defer_presence        yes
zv_append_safe_sharding  yes
zv_dense_manifests       yes
device_count             1
```

(A single-GPU workstation, RTX A2000, with the `bridge-gpu-zv3` environment.
On the 8 × H100 node `peer_access` should be `yes` and `device_count` 8.)

`--require` turns the probe into a gate for job scripts: it exits non-zero
and names what is missing.

```bash
zvcfd probe --require cupy,cuda_device,zv_read_cells,zv_defer_presence
```

`probe` initialises CUDA. A process that forks workers should probe inside
the workers, not before forking.

## Cluster images

The UCL CS cluster runs Grid Engine with Apptainer images built from
`rapidsai/base` (see BRIDGE's `envs/bridge-gpu-h100.def`). zvCFD installs
into that image with `--no-deps` on top of BRIDGE's environment, the same
way BRIDGE-Simulation does; [Running on the cluster](../how_to/hpc_sge.md)
has the job script.
