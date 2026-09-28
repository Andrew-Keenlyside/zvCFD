# Your first simulation

This tutorial runs a complete simulation from a configuration file:
pressure-driven blood flow through a synthetic vessel network, on one GPU.
It then inspects what the run wrote. It uses only the command line; the
[Quickstart](../getting_started/quickstart.md) does the same steps from
Python.

---

## The configuration

`examples/network_demo.yaml`:

```yaml
# Pressure-driven flow through the synthetic vessel network (128 x 128 x 512 voxels).
name: network-demo
source: {kind: phantom, name: network-128x128x512, voxel_size: 10.0}   # micrometre
physics: {nu: 3.5e-6, rho: 1060.0, pressure_drop: 20.0}                  # blood; Pa across x
solver: {method: lbm, precision: fp32, tau: 1.0, steps: 20000, check_every: 1000, tolerance: 1.0e-4}
domain: {brick: 8, chunk_bricks: 16}
output: {path: runs, every: 5000, fields: [rho, ux, uy, uz], compressor: zstd}
```

Every section is validated strictly: a misspelt key is an error, not a
silent default. The configuration's hash (the first 12 hex digits of the
SHA-256 of its sorted JSON) names the run, so two runs of the same
configuration land in the same directory.

| Section | What it sets |
|---|---|
| `source` | where the geometry comes from: `phantom`, `omezarr` (a pyramid level, threshold or label) or `npy` |
| `physics` | kinematic viscosity (m²/s), density (kg/m³), and the driving: a pressure drop between the x faces or a body force |
| `solver` | method, population precision, τ, step budget, convergence check |
| `domain` | brick size (8) and bricks per store chunk |
| `output` | where, how often, which fields, codec, shards |

## Size it first

```bash
zvcfd plan --fluid-cells 1.2e6 --fill 0.47 --gpus 1 --gpu RTX-A2000 --steps 20000
```

```text
1.2e+06 fluid cells, sparse layout, vessel geometry, 1 x RTX-A2000, planning kernel efficiency
method              stored   GB/GPU  fits   Gupd/s   ms/step       total
lbm-fp32          2.55e+06      0.4   yes      0.6      2.11     42.11 s
```

The phantom has ~1.2 × 10⁶ fluid voxels. `plan` gives memory and a time
budget before anything is allocated. The run below converges long before
the 20,000-step budget.

## Run it

```bash
zvcfd run examples/network_demo.yaml --out runs/
```

```text
domain (128, 128, 512): 4989 bricks (30.5% active), 1,192,679 fluid cells, fill 0.47; dt = 4.76e-06 s, drho = 0.0128
  step     1000  mean u_x 2.6418e-04 (lattice)
  step     2000  mean u_x 2.8336e-04 (lattice)
  step     3000  mean u_x 2.8674e-04 (lattice)
  step     4000  mean u_x 2.8736e-04 (lattice)
  step     5000  mean u_x 2.8750e-04 (lattice)
  wrote step-000005000.zarrvectors
  step     6000  mean u_x 2.8754e-04 (lattice)
  step     7000  mean u_x 2.8755e-04 (lattice)
  converged: relative change <= 0.0001
  wrote step-000007000.zarrvectors
done: 7000 steps, 721 MLUPS incl. checks/output, 14.0 s total -> runs/network-demo-78c04469a293.zvcfd
```

What happened:

1. The phantom's flags became a `BrickDomain`: 4,989 active bricks, 30 %
   of the box. Reservoir flags were added to the fluid voxels of both x
   faces.
2. The 20 Pa pressure drop was converted to a lattice density difference
   (0.0128) with `zvcfd.Lattice` (10 µm voxels, blood viscosity, τ = 1 →
   dt = 4.8 µs).
3. The domain store (flags), the configuration and an empty run collection
   were written.
4. The solver ran in blocks of 1,000 steps, checking the mean velocity. It
   wrote a snapshot at step 5,000, stopped when the relative change fell
   below 10⁻⁴, and wrote the final state.
5. Each snapshot was published by one atomic update of the collection.

## What is on disk

```bash
zvcfd info runs/network-demo-78c04469a293.zvcfd
```

```text
run network-demo (zvc-run-8e7437307dd4)
  zvcfd:domain     domain       ./domain.zarrvectors
  zvcfd:config     config       ./config.json
  collection       fields
  2 snapshots: step-000005000 .. step-000007000
```

```text
runs/network-demo-78c04469a293.zvcfd/
├── zarr.json                      # the RFC-8 collection document
├── config.json                    # the resolved configuration
├── domain.zarrvectors/            # flags, one row per brick
└── fields/
    ├── step-000005000.zarrvectors/
    └── step-000007000.zarrvectors/
```

Each `.zarrvectors` directory is an ordinary Zarr Vectors store. Any
zarr-vectors reader can open it, and so can zv-ngtools for viewing.
[Reading results](reading_results.md) continues from here.

## Things to try

- `precision: fp16` halves the population memory. On this GPU the fp16
  kernel is slower per step than fp32, because it is not yet vectorised.
- `output: {shard_shape: 2}` packs 8 cells per file; see
  [Choosing brick and chunk sizes](../how_to/choose_brick_and_chunk.md).
- Increase `pressure_drop` and watch the lattice velocity: keep it below
  ~0.1 (Mach number), or reduce τ and accept a smaller time step.
