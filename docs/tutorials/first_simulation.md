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
solver: {method: lbm, precision: fp32, tau: 0.6, steps: 20000, check_every: 1000, tolerance: 1.0e-4}
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
lbm-fp32          2.55e+06      0.4   yes      0.6      1.94     38.79 s
```

The phantom has ~1.2 × 10⁶ fluid voxels. `plan` gives memory and a time
budget before anything is allocated. The run below converges long before
the 20,000-step budget.

## Run it

```bash
zvcfd run examples/network_demo.yaml --out runs/
```

```text
domain (128, 128, 512): 4989 bricks (30.45% active), 1,192,679 fluid cells, fill 0.47; 2 patches; tau = 0.6000, dt = 9.52e-07 s  (2.6 s)
  step     1000  inflow 2.3829e-10 m3/s  imbalance +7.06e-01  change inf
  step     2000  inflow 1.6666e-10 m3/s  imbalance -3.68e-02  change 6.17e-01
  step     3000  inflow 1.7155e-10 m3/s  imbalance -8.26e-03  change 2.85e-02
  step     4000  inflow 1.7187e-10 m3/s  imbalance -5.22e-03  change 1.90e-03
  step     5000  inflow 1.7249e-10 m3/s  imbalance +1.98e-03  change 3.60e-03
  wrote step-000005000.zarrvectors
  step     6000  inflow 1.7229e-10 m3/s  imbalance -4.04e-04  change 1.21e-03
  step     7000  inflow 1.7233e-10 m3/s  imbalance +5.74e-05  change 2.37e-04
  step     8000  inflow 1.7231e-10 m3/s  imbalance -8.33e-05  change 7.07e-05
  converged: change <= 0.0001, |imbalance| <= 0.001
  wrote step-000008000.zarrvectors
done: 8000 steps, 384 MLUPS incl. checks/output, 29.3 s total -> runs/network-demo-d78037f7f963.zvcfd
```

What happened:

1. The phantom's flags became a `BrickDomain`: 4,989 active bricks, 30 %
   of the box.
2. `pressure_drop: 20.0` is shorthand for two pressure patches: the fluid
   voxels of the `xmin` face at 20 Pa and those of the `xmax` face at 0 Pa
   ([Boundary conditions](../spec/boundary_conditions.md)). The pressures
   became lattice densities with `zvcfd.Lattice` (10 µm voxels, blood
   viscosity, τ = 0.6 → dt = 0.95 µs).
3. The domain store (flags), the configuration and an empty run collection
   were written.
4. The solver ran in blocks of 1,000 steps. At each check it measured the
   exact flow through every patch. It stopped when the flow changed by
   less than 10⁻⁴ between checks *and* the inflow matched the outflow to
   10⁻³ (the "imbalance"). It wrote a snapshot at step 5,000 and the final
   state at 8,000.
5. Each snapshot was published by one atomic update of the collection,
   and `patches.csv` records each patch's flow and pressure: 0.172 nL/s
   (1.723 × 10⁻¹⁰ m³/s) through the network at 20 Pa.

τ = 0.6 is not arbitrary. In steady flow the answer does not depend on τ,
only the time to reach it, and for branched vessels small τ converges
fastest (at τ = 1 this run needs more than 20,000 steps; see
[Numerics](../spec/numerics.md)).

## What is on disk

```bash
zvcfd info runs/network-demo-d78037f7f963.zvcfd
```

```text
run network-demo (zvc-run-47c26ba0c69b)
  zvcfd:domain     domain       ./domain.zarrvectors
  zvcfd:config     config       ./config.json
  collection       fields
  2 snapshots: step-000005000 .. step-000008000
```

```text
runs/network-demo-d78037f7f963.zvcfd/
├── zarr.json                      # the RFC-8 collection document
├── config.json                    # the resolved configuration
├── domain.zarrvectors/            # flags, one row per brick
├── monitors.json                  # flows, imbalance and change at every check
├── patches.csv                    # per patch: voxels, area, flow, split, pressure
└── fields/
    ├── step-000005000.zarrvectors/
    └── step-000008000.zarrvectors/
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
