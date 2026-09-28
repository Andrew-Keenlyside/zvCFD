# Running on the cluster

The target node is UCL CS's `seymour2` (8 × H100 80 GB, driver 550.127.08,
CUDA 12.4), scheduled by **Grid Engine, not SLURM**. Directives are `#$`,
the job id is `$JOB_ID`, and the GPU prolog exports `CUDA_VISIBLE_DEVICES`
before the script runs. This page follows BRIDGE's
`envs/bridge_sge_job.sh` and BRIDGE-Simulation's `bridge_sim_sge_job.sh`.

---

## The image

zvCFD installs into BRIDGE's Apptainer image (from `rapidsai/base`, with
cupy, kvikio and nvCOMP) with `--no-deps`, as BRIDGE-Simulation does:

```text
Bootstrap: localimage
From: bridge-gpu-h100.sif

%post
    pip install --no-deps /opt/src/zarr-vectors-py      # gpu-backend branch
    pip install --no-deps "/opt/src/zvCFD[amg,yaml]"
    pip install pyamg pyyaml
```

## A job script

```bash
#!/bin/bash
#$ -N zvcfd
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -o /SAN/external/bridge_project_data/logs/$JOB_NAME.$JOB_ID.log
#$ -l gpu=true,gpu_type=h100
#$ -l h_rt=24:00:00
#$ -l tmem=32G,h_vmem=32G
#$ -pe gpu 8
# tmem is PER SLOT: Grid Engine multiplies it by $NSLOTS (8 x 32G = 256G).

set -euo pipefail
CONTAINER=${CONTAINER:-/SAN/external/bridge_project_data/containers/zvcfd-h100.sif}
RUN_ROOT=${RUN_ROOT:-/SAN/external/bridge_project_data/derivatives/zvCFD}

[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: not a GPU allocation" >&2; exit 1; }
export TMPDIR="${TMPDIR:-/tmp/zvcfd.$$}"; mkdir -p "$TMPDIR"
export CUPY_CACHE_DIR="$HOME/.cache/zvcfd-kernels"

RUNTIME=$(command -v apptainer || command -v singularity)
run() { "$RUNTIME" exec --nv --bind /SAN "$CONTAINER" "$@"; }

run zvcfd probe --require cupy,cuda_device,zv_read_cells,zv_defer_presence
run zvcfd run "$1" --out "$RUN_ROOT/runs"
```

```bash
qsub envs/zvcfd_sge_job.sh configs/coronary_20um.yaml
```

Whether `-pe gpu 8` (or `-l gpu=8`) is granted all eight cards together is
not yet confirmed on this cluster. BRIDGE-Simulation's 8-GPU benchmark
script requests `-l gpu=8` with `tmem=256G`. Check
`CUDA_VISIBLE_DEVICES` in the log of the first job.

## Where to write

| Data | Where | Why |
|---|---|---|
| Run collections, snapshots | `/SAN/...` project space | durable; the filesystem type is not yet known (milestone 0 measures it) |
| Checkpoints of a running job | node-local `$TMPDIR`, then copied | fast local NVMe; GPUDirect Storage is only possible on local NVMe/ext4 or a GDS-capable parallel filesystem |
| Kernel caches (`CUPY_CACHE_DIR`) | `$HOME/.cache` | reused across jobs |

Never put the cupy kernel cache or scratch on `/SAN` or an NFS home. Also
check file counts against the project's inode quota. BRIDGE recorded an
EDQUOT at 933 k files under a 1 M quota. Flat brick stores make one file
per chunk per field, so use `output.shard_shape` for long runs with many
snapshots ([Choosing brick and chunk sizes](choose_brick_and_chunk.md)).

## Measuring the node first

```bash
run python benchmarks/bench_lbm.py --json benchmarks/results/bench_lbm_h100.json
run python benchmarks/bench_io.py --root "$TMPDIR/io" --workers 8 --json benchmarks/results/bench_io_h100_local.json
run python benchmarks/bench_io.py --root "$RUN_ROOT/_io_bench" --workers 8 --json benchmarks/results/bench_io_h100_san.json
```

These three replace the extrapolated figures in the documentation with
measured ones. That is milestone 0 of the roadmap.
