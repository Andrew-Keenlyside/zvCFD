# Running on the 8 × H100 node

```{admonition} Designed, not built
:class: warning

The multi-GPU driver is milestone 1 of the [roadmap](../feasibility/roadmap.md).
This page describes how a run on the UCL `seymour2` node (8 × H100 80 GB,
NVLink) will work. It also shows what already runs there today: probing,
planning, partitioning, and single-GPU runs, one per device.
```

---

## Today: check the node and plan

Inside the Apptainer image (see [Running on the cluster](../how_to/hpc_sge.md)):

```bash
zvcfd probe --require cupy,cuda_device,nccl,peer_access,zv_read_cells,zv_defer_presence
zvcfd plan --fluid-cells 6.3e8 --fill 0.7 --gpus 8 --method lbm-fp32,lbm-fp16 --steps 500000
```

## Today: see how the domain would be split

```python
import numpy as np
from zvcfd import BrickDomain

domain = BrickDomain.from_flags(flags)              # your flag volume
parts = domain.partition(8, chunk_bricks=32)        # whole chunks per GPU, balanced by fluid
work = np.bincount(parts, weights=(domain.flags != 1).sum(1), minlength=8)
halo = [len(domain.halo(parts, p)) for p in range(8)]
print(work / work.mean(), halo)
```

`work / mean` near 1 means the fluid is balanced across GPUs. The halo
counts are the ghost bricks each GPU will refresh every step.

## Today: one independent run per GPU

Parameter sweeps and ensembles need no communication. Pin one run to each
device, as BRIDGE-Simulation does for its ensembles:

```bash
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g zvcfd run sweep/case_$g.yaml --out runs/ &
done
wait
```

## Planned: one run across all eight GPUs

```bash
zvcfd run coronary_10um.yaml          # with parallel: {gpus: 8}
```

What will happen:

1. **Coordinator** (the `zvcfd run` process) builds or opens the domain
   store, partitions whole chunks across 8 GPUs, creates each output store
   and writes the run collection.
2. **Eight workers** are spawned, one per GPU (`spawn` start method; CUDA
   and GPUDirect Storage do not survive `fork`). Each loads only the bricks
   it owns plus its ghost layer, reading them straight from the domain store
   with `read_cells(device="cuda")`.
3. **Each step** is split into an interior kernel and a boundary kernel. In
   between, the crossing populations of boundary bricks go to neighbouring
   GPUs, over NVLink via NCCL send/recv.
4. **Snapshots**: every worker copies its fields to a pinned host buffer
   and writes its own chunks from a writer thread, with no lock, while
   stepping on. The coordinator rebuilds presence once and publishes.
5. **Checkpoints** are the same, with populations instead of fields.

Expected for the coronary tree at 10 µm (6.3 × 10⁸ fluid voxels): 18 GB per
GPU, ~14 ms per step, ~1.8 h per cardiac cycle (planning basis;
[Comparison](../benchmarks/comparison.md)). The first measurement on the
node (milestone 0) will confirm or correct those figures.
