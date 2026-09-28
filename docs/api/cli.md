# Command line

`zvcfd` follows the conventions of `zvtools` and `bridge-sim`: argparse
subcommands, kebab-case flags, comma-separated shapes (`--shape Z,Y,X`),
exit status 0 on success. `python -m zvcfd` is the same command.

| Command | Does |
|---|---|
| `zvcfd probe [--require a,b] [--json]` | report capabilities; with `--require`, exit 1 naming what is missing |
| `zvcfd plan …` | memory and time for a domain ([Plan a run](../how_to/plan_a_run.md)) |
| `zvcfd mesh-info <file.msh> [--voxel-size …] [--unit mm]` | summarise an ASCII Fluent mesh and plan voxel runs of it |
| `zvcfd phantom list` / `zvcfd phantom build <name> --out mask.npy` | synthetic geometries |
| `zvcfd run <config> [--out DIR] [--steps N]` | run a configuration (one GPU today) |
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

## `zvcfd run`

The configuration file is YAML (needs `pyyaml`) or JSON. Its schema is
`zvcfd.config.RunConfig`, validated strictly
([Your first simulation](../tutorials/first_simulation.md#the-configuration)).
