# Plan a run

`zvcfd plan` answers the two questions to settle before submitting a job:
does the domain fit, and how long will it take? It reads nothing and
allocates nothing. It is the roofline model in `zvcfd.perfmodel`.

---

## From a fluid-voxel count

```bash
zvcfd plan --fluid-cells 6.3e8 --fill 0.7 --gpus 8 --method lbm-fp32,lbm-fp16 --steps 500000
```

```text
6.3e+08 fluid cells, sparse layout, vessel geometry, 8 x H100-SXM, planning kernel efficiency
method              stored   GB/GPU  fits   Gupd/s   ms/step       total
lbm-fp32             9e+08     17.7   yes     48.5     13.00       1.8 h
lbm-fp16             9e+08      9.1   yes     30.3     20.82       2.9 h
```

## From a box and a fluid fraction

```bash
zvcfd plan --shape 2048,2048,2048 --fluid-fraction 0.2 --layout dense --geometry porous \
           --method lbm-fp32,lbm-fp16 --steps 30000
```

```text
1.72e+09 fluid cells, dense layout, porous geometry, 8 x H100-SXM, planning kernel efficiency
method              stored   GB/GPU  fits   Gupd/s   ms/step       total
lbm-fp32          8.59e+09    168.6    NO     73.4     23.41    11.7 min
lbm-fp16          8.59e+09     87.0    NO     74.3     23.13    11.6 min
```

A porous medium touches nearly every brick, so use `--layout dense` with
the box size. This one does not fit one node in either precision; the
times are what it would take if it did. fp16 is no faster here, because
the fp16 kernel is not yet vectorised and the planning basis counts only
what has been measured.

## From an Ansys mesh

```bash
zvcfd mesh-info coronary.msh --voxel-size 0.02,0.01,0.005
```

gives fluid voxels, fill of the box, the narrowest patch in voxels, and the
plan per GPU for each voxel size ([From an Ansys mesh](../tutorials/ansys_mesh.md)).

## What the numbers mean

`stored`
: Cells held on the GPU: fluid ÷ fill (sparse), or the box (dense).

`GB/GPU`
: Two population buffers plus flags and neighbour table: 157 B per stored
  cell in fp32, 81 B in fp16, divided by the GPU count. `fits` allows 90 %
  of device memory.

`Gupd/s`
: Aggregate fluid-cell updates per second: copy bandwidth × kernel
  efficiency ÷ bytes per update, × GPUs × 0.85 multi-GPU efficiency.

`--efficiency`
: `measured` uses the kernel efficiencies measured on the A2000. `target`
  uses published figures for tuned kernels (FluidX3D, waLBerla).
  `planning` (the default) uses the lower of the two.

## Choosing the brick fill

The fill is known exactly once the domain exists (`BrickDomain.fill`).
Before that, estimate it from the narrowest channels in voxels:

| Narrowest channel radius | Typical fill |
|---|---:|
| ≲ 5 voxels (thin capillaries, coarse levels) | 0.35–0.5 |
| 6–20 voxels (coronary tree at 10–20 µm) | 0.55–0.7 |
| ≳ 25 voxels | 0.8 and above |
| porous media, porosity φ | ≈ φ, with nearly every brick active (use `--layout dense`) |

## Steps

For **transient** runs, the step count follows from the time step: with
the lattice velocity held at ≤ 0.1, `dt = 0.1 · dx / u_peak`. For a 1 s
cardiac cycle at 0.5 m/s peak that is 2.5 × 10⁵ steps at 20 µm and
5 × 10⁵ at 10 µm. `zvcfd.Lattice` gives τ for that dt (0.605 at 20 µm,
0.71 at 10 µm).

For **steady** runs, budget 10⁴–10⁵ steps. Our vessel networks reached
0.1 % of the steady flux in 5,600 steps at 512 voxels long and 14,000
steps at 1,024 voxels long.
