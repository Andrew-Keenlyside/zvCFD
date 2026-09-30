# Command line

`zvcfd` follows the conventions of `zvtools` and `bridge-sim`: argparse
subcommands, kebab-case flags, comma-separated shapes (`--shape Z,Y,X`),
exit status 0 on success. `python -m zvcfd` is the same command.

| Command | Does |
|---|---|
| `zvcfd probe [--require a,b] [--json]` | report capabilities; with `--require`, exit 1 naming what is missing |
| `zvcfd plan …` | memory and time for a domain ([Plan a run](../how_to/plan_a_run.md)) |
| `zvcfd mesh-info <file.msh> [--voxel-size …] [--unit mm] [--volume]` | summarise an ASCII Fluent mesh and plan voxel runs of it; with `--volume`, the finite-volume view |
| `zvcfd voxelize <file.msh> --voxel-size UM [--unit mm] [--out domain.zarrvectors]` | voxelise a Fluent mesh into a sparse domain with inlet/outlet patches; report patch voxel counts |
| `zvcfd phantom list` / `zvcfd phantom build <name> --out mask.npy` | synthetic geometries |
| `zvcfd run <config> [--out DIR] [--steps N]` | run a configuration, on one GPU or several (`parallel.gpus`) |
| `zvcfd info <run.zvcfd \| store.zarrvectors>` | describe a run collection or brick store |

## `zvcfd plan`

| Flag | Default | Meaning |
|---|---|---|
| `--fluid-cells N` | | fluid voxels; or give `--shape` with `--fluid-fraction` |
| `--shape Z,Y,X` | | bounding box in voxels (required for `--layout dense`) |
| `--fluid-fraction F` | | fluid fraction of the box |
| `--fill F` | 0.6 | brick fill (sparse layout) |
| `--layout` | `sparse` | `sparse` or `dense` |
| `--geometry` | `vessel` | `open`, `porous` or `vessel` (kernel-efficiency class) |
| `--gpus N` | 8 | |
| `--gpu` | `H100-SXM` | `H100-SXM`, `H100-PCIe`, `A100-SXM-80`, `RTX-A2000` |
| `--method` | `lbm-fp32` | comma list of `lbm-fp32`, `lbm-fp16`, `lubrication-amg` |
| `--steps N` | | adds a total time |
| `--efficiency` | `planning` | `planning`, `measured` or `target` |
| `--json` | | machine-readable rows |

## `zvcfd mesh-info`

Takes `--fill`, `--gpus`, `--gpu`, `--method` as `plan` does, and
`--voxel-size` as a comma list in mesh units.

With `--volume` it reads the whole volume mesh instead (ASCII or binary)
and reports, for the finite-volume solver: element counts by type, the
dual-geometry checks (control volumes against element volumes, closure,
minimum orthogonality), blocks per row of the coupled system, flux
points, and the GPU memory the solve would need in double, mixed and
DILU variants. On the HiP-CT coronary mesh this takes about 4 minutes
and 11 GB of host memory.

## `zvcfd run`

The configuration file is YAML (needs `pyyaml`) or JSON. Its schema is
`zvcfd.config.RunConfig`, validated strictly
([Your first simulation](../tutorials/first_simulation.md#the-configuration)).

| Section | Keys |
|---|---|
| `source` | `kind` (`phantom`, `omezarr`, `npy`, `mesh`), `name`, `path`, `level`, `label`, `threshold`, `voxel_size` (µm), `unit` (mesh unit), `region` |
| `physics` | `nu` (m²/s), `rho` (kg/m³), `u_ref` (m/s, with `solver.mach` sets the time step), `pressure_drop` (Pa, shorthand for two x-face patches), `body_force`, `rheology` (`{model: carreau-yasuda, mu_0, mu_inf, lam, a, n}`) |
| `boundaries` | `faces: {xmin: {kind, ...}}`, `patches: [{match: substring, kind, pressure / velocity / flow_rate / profile / rcr / waveform}]` |
| `solver` | `collision` (`trt`), `precision`, `tau` or `mach`, `steps`, `check_every`, `tolerance` (flux change), `mass_tolerance` (imbalance), `flow_control` |
| `domain` | `brick` (8), `chunk_bricks`, `periodic` |
| `parallel` | `gpus`, `partitions` |
| `output` | `path`, `every`, `fields`, `dtype`, `compressor`, `shard_shape` |

A run writes `monitors.json` (flux history, imbalance, convergence) and
`patches.csv` (per patch: kind, voxels, area, flow, split, mean pressure)
into the run collection.
