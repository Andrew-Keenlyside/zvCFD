# Command line

`zvcfd` follows the conventions of `zvtools` and `bridge-sim`: argparse
subcommands, kebab-case flags, comma-separated shapes (`--shape Z,Y,X`),
exit status 0 on success. `python -m zvcfd` is the same command.

| Command | Does |
|---|---|
| `zvcfd probe [--require a,b] [--json]` | report capabilities; with `--require`, exit 1 naming what is missing |
| `zvcfd plan …` | memory and time for a domain ([Plan a run](../how_to/plan_a_run.md)) |
| `zvcfd mesh-info <file.msh> [--voxel-size …] [--unit mm] [--volume]` | summarise an ASCII Fluent mesh and plan voxel runs of it; with `--volume`, the finite-volume view |
| `zvcfd import-mesh <mesh> [--out X.zvmesh] [--unit mm] [--chunk M] [--surface-only]` | import any mesh into a Zarr Vectors mesh collection, the input both solvers run from |
| `zvcfd voxelize <X.zvmesh \| file.msh> --voxel-size UM [--unit mm] [--out domain.zarrvectors]` | voxelise a mesh collection (or a Fluent mesh) into a sparse domain with inlet/outlet patches; report patch voxel counts |
| `zvcfd phantom list` / `zvcfd phantom build <name> --out mask.npy` | synthetic geometries |
| `zvcfd run <config> [--out DIR] [--steps N]` | run a configuration, on one GPU or several (`parallel.gpus`) |
| `zvcfd info <run.zvcfd \| X.zvmesh \| store.zarrvectors>` | describe a run collection, mesh collection or store |

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

## `zvcfd import-mesh`

Reads a mesh and writes it as a mesh collection
([Mesh collections](../spec/mesh_store.md#mesh-collections)): a volume
store and a boundary store, in metres. Both solvers run from a
collection, so this is the first step for any mesh.

| Source | Read by |
|---|---|
| Ansys Fluent `.msh` (ASCII or binary) | `zvcfd.mesh.read_fluent_mesh`; with `--surface-only`, only the boundary zones |
| VTK `.vtu`, or a SimVascular `mesh-complete` folder | `zvcfd.mesh.vtk.read_vtu_mesh` (the folder's `.vtp` faces become zones) |
| Gmsh, Abaqus, Exodus, MED, Nastran, … | [meshio](https://github.com/nschloe/meshio), if installed: tagged boundary faces (Gmsh physical groups, cell sets) become zones, untagged ones zone `wall` |

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `<mesh>.zvmesh` beside the mesh | the collection |
| `--unit` | `mm` | the source's coordinate unit |
| `--chunk` | a quarter of the longest box edge | spatial chunk edge, metres |
| `--surface-only` | | boundary store only, enough for the voxel solver |

A zone whose name contains `inlet` is a velocity inlet and one containing
`outlet` a pressure outlet, unless the source says otherwise; `zvcfd run`
maps zones to conditions by the `boundaries.patches` rules either way.

```console
$ zvcfd import-mesh "mesh 1.msh" --out coronary.zvmesh --unit mm
read mesh 1.msh: 5,673,949 nodes, {'tet': 6546228, 'pyramid': 87, 'wedge': 8244327}, 82 zones (64.3 s)
wrote coronary.zvmesh (70.5 s)
  5,673,949 nodes, {'tet': 6546228, 'pyramid': 87, 'wedge': 8244327}; chunk 16.5526 mm; box [58.481, 58.765, 66.21] mm
  zone    3 wall                     4 faces  from-lumen_…-to-xmax-wall
  …
  zone   84 velocity-inlet       1,822 faces  from-lumen_…-to-cor_inlet_001_amira_amiranode_329-velocity-inlet
```

## `zvcfd run`

The configuration file is YAML (needs `pyyaml`) or JSON. Its schema is
`zvcfd.config.RunConfig`, validated strictly
([Your first simulation](../tutorials/first_simulation.md#the-configuration)).

| Section | Keys |
|---|---|
| `source` | `kind` (`phantom`, `omezarr`, `npy`, `mesh`), `name`, `path`, `level`, `label`, `threshold`, `voxel_size` (µm), `unit` (raw mesh unit), `region` |
| `physics` | `nu` (m²/s), `rho` (kg/m³), `u_ref` (m/s, with `solver.mach` sets the time step), `pressure_drop` (Pa, shorthand for two x-face patches), `body_force`, `rheology` (`{model: carreau-yasuda, mu_0, mu_inf, lam, a, n}`) |
| `boundaries` | `faces: {xmin: {kind, ...}}`, `patches: [{match: substring, kind, pressure / velocity / flow_rate / profile / rcr / coronary / pim / waveform}]`; with `solver.method: fv` also the kinds `wall`, `symmetry` and the pressure-zone keys `backflow_stabilisation` (β), `pressure_profile: average`, `opening: true` |
| `solver` | `method` (`auto`, the default: `fv` for mesh sources, `lbm` for voxel sources and for meshes given a `voxel_size`; or `fv`, `lbm` explicitly), `collision` (`trt`), `precision`, `tau` or `mach`, `steps`, `check_every`, `tolerance` (flux change), `mass_tolerance` (imbalance), `flow_control` |
| `fv` | finite-volume settings (`solver.method: fv`): `linear` (`auto`, `amgx`, `simple`, `acm`, `host-direct`), `linear_rtol` (0.1), `precision` (`double`, `mixed`: single-precision AmgX hierarchy), `advection` (`high-resolution`, `upwind`, or a blend 0–1), `false_dt`, `iterations`, `tolerance` (relative change of u and p), `residual_target` (largest RMS normalised residual; steady and per time step), `backflow_stabilisation` (default 0.2), `dt`, `steps` or `periods` with `period`, `loops`, `loop_tolerance`, `scheme` (`bdf1`, `bdf2`), `steady_start`, `wss`, `tawss_from`, `chunk` |
| `domain` | `brick` (8), `chunk_bricks`, `periodic` |
| `parallel` | `gpus`, `partitions` |
| `output` | `path`, `every`, `fields`, `dtype`, `compressor`, `shard_shape` |

With `source.kind: mesh`, `path` is a mesh collection (`.zvmesh`), or
any mesh `import-mesh` reads. A raw mesh is imported once into
`<output.path>/meshes/<name>-<hash>.zvmesh` and reused by later runs; the
key covers the file's path, size and modification time, `unit` and
`fv.chunk`. The voxel solver voxelises the collection's boundary store,
and the finite-volume solver solves on the mesh read back from its volume
store. The run collection links the mesh collection (node
`mesh-collection`).

A run writes `monitors.json` (flux history, imbalance, convergence) and
`patches.csv` (per patch: kind, voxels, area, flow, split, mean pressure)
into the run collection. A finite-volume run also writes node-field
snapshots (`fields/step-*.zarr`: velocity, pressure, wall shear at wall
nodes) and, after a transient, TAWSS, OSI and RRT as vertex attributes of
the boundary store; `monitors.json` records each outer iteration's
residuals, linear iterations and convergence, the linear solver used, and
any switches of the `auto` ladder. Settings: [Finite-volume numerics](../spec/fv_numerics.md);
examples: `examples/pipe_fv.yaml`, `examples/coronary_fv.yaml`.
