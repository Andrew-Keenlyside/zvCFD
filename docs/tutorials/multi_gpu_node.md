# Running on the 8 × H100 node

```{admonition} Built, not yet measured on the node
:class: note

The multi-GPU driver (`zvcfd.lbm.MultiLBM`) is built and tested: a run
split into partitions is bit-identical to a single-partition run. What has
not happened yet is a run on the H100 node itself (milestone 0 of the
[roadmap](../feasibility/roadmap.md)); `envs/zvcfd_sge_job.sh` does that in
one job.
```

---

## Check the node and plan

Inside the Apptainer image (see [Running on the cluster](../how_to/hpc_sge.md)):

```bash
zvcfd probe --require cupy,cuda_device,nccl,peer_access,zv_read_cells,zv_defer_presence
zvcfd plan --fluid-cells 6.3e8 --fill 0.7 --gpus 8 --method lbm-fp32,lbm-fp16 --steps 500000
```

## Run across all GPUs

Set `parallel.gpus` in the configuration; `examples/coronary_20um_8gpu.yaml`
and `examples/coronary_10um_8gpu.yaml` do:

```yaml
parallel: {gpus: 8}
```

```bash
zvcfd run examples/coronary_20um_8gpu.yaml --out runs/
```

From Python:

```python
from zvcfd.lbm import MultiLBM

sim = MultiLBM(domain, n_parts=8, chunk_bricks=16, devices=list(range(8)),
               boundary=boundary, tau=0.6, collision="trt")
sim.set_patch(0, u_zyx=(0.0, 0.0, 0.01))
sim.step(1000)
flows = sim.patch_flux()
```

What happens:

1. **Partitioning.** `BrickDomain.partition` orders the store chunks along a
   Morton curve and cuts them into eight runs of equal fluid work. Each GPU
   owns whole chunks, so it can later write its own store cells with no lock.
2. **Local domains.** Each partition holds its owned bricks, then a
   one-brick ghost layer (`BrickDomain.local`), with a neighbour table
   renumbered to local indices.
3. **A step.** Every GPU collides and streams its owned bricks; the kernels
   are launched back to back, so the devices run concurrently. Ghost bricks
   are then refreshed from their owners by device-to-device copies (peer
   access over NVLink is enabled where available). The boundary kernel
   writes patch voxels, and ghosts are refreshed once more.
4. **Monitoring.** Patch fluxes and states are computed per partition on
   its owned patch voxels and summed.

## Check the overhead on one GPU

With more partitions than devices, partitions share devices round-robin.
On a one-GPU machine that exercises the whole exchange path:

```bash
python benchmarks/bench_multi.py --devices 0 --parts 1,2,4,8
```

On the node, the same script with `--devices 0,1,2,3,4,5,6,7` measures
weak and strong scaling.

## What is not optimised yet

- The exchange copies whole ghost bricks (all 19 populations of 512
  voxels), not only the populations that cross each face.
- Exchange and compute do not overlap: every device synchronises before the
  copies.
- Snapshots are gathered to the host and written by one process.

Each is a milestone-1 item. Expected for the coronary tree at 10 µm
(6.3 × 10⁸ fluid voxels) once they are done: 18 GB per GPU, ~12 ms per step,
~1.7 h per cardiac cycle (planning basis; [Comparison](../benchmarks/comparison.md)).
