#!/bin/bash
# ---------------------------------------------------------------------------
# zvCFD milestone 0 on the UCL CS Grid Engine cluster (seymour2: 8x H100 80GB).
#
#   qsub -v MESH=/SAN/.../mesh1.msh,CONTAINER=/SAN/.../bridge-gpu-h100-a7b48a0.sif \
#       envs/zvcfd_sge_job.sh
#
# One job, one folder: everything it produces lands in benchmarks/results/h100/
# (OUT), including a copy of this log, so that folder is the whole hand-back:
#
#   env.txt            node, GPUs, NVLink topology, image, repo commit
#   job.log            this job's full output, with peak host RSS per step
#   steps.tsv          step, exit code, seconds (a failed step does not stop the rest)
#   probe.json         zvcfd probe --json
#   pytest.xml         the test suite on an H100
#   validation/        exact solutions and benchmarks (benchmarks/validation/run.py)
#   bench_lbm.json     single-GPU kernels
#   bench_multi.json   weak and strong scaling over the granted GPUs
#   bench_io_{local,san}.json
#   coronary/          HiP-CT coronary tree at 20 and 10 um on every granted GPU:
#                      monitors.json, patches.csv, config.json per run, plus the
#                      final fields (*.zarrvectors, ~10 GB at 10 um; skip on rsync)
#
# Runs in BRIDGE's image, with no zvCFD image to build: zvCFD is pure Python
# plus runtime-compiled cupy kernels, so the repo goes on PYTHONPATH. The image
# must carry zarr-vectors 06a3bc8 (gpu-backend) or later: the 06e3af0, e59e1a8
# and a7b48a0 builds do, the Sep-2 bridge-gpu-h100.sif does not (the probe
# fails fast on it). Pass CONTAINER=... when the default path is the old build.
#
# Grid Engine, NOT SLURM: directives are #$, the job id is $JOB_ID, and the GPU
# prolog exports CUDA_VISIBLE_DEVICES. Follows BRIDGE's envs/bridge_sge_job.sh.
# ---------------------------------------------------------------------------
#$ -N zvcfd-m0
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -o /SAN/external/bridge_project_data/logs/$JOB_NAME.$JOB_ID.log
#$ -l gpu=true,gpu_type=h100
#$ -l h_rt=12:00:00
#$ -l tmem=32G,h_vmem=32G
#$ -pe gpu 8
# tmem is PER SLOT: 8 x 32G = 256G. Whether -pe gpu 8 (or -l gpu=8) grants all
# eight cards is unconfirmed on this cluster; env.txt records what was granted.

set -euo pipefail
CONTAINER=${CONTAINER:-/SAN/external/bridge_project_data/containers/bridge-gpu-h100.sif}
REPO=${REPO:-$PWD}
MESH=${MESH:?set MESH to the coronary .msh path}
OUT=${OUT:-$REPO/benchmarks/results/h100}

[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: not a GPU allocation" >&2; exit 1; }
[ -f "$MESH" ] || { echo "FATAL: mesh not found: $MESH" >&2; exit 1; }
[ -f "$CONTAINER" ] || { echo "FATAL: image not found: $CONTAINER" >&2; exit 1; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1

NGPU=$(printf '%s\n' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
DEVICES=$(seq -s, 0 $((NGPU - 1)))
echo "job ${JOB_ID:-local} on $(hostname): $NGPU GPU(s) [$CUDA_VISIBLE_DEVICES]"
export TMPDIR="${TMPDIR:-/tmp/zvcfd.$$}"; mkdir -p "$TMPDIR"
export CUPY_CACHE_DIR="$HOME/.cache/zvcfd-kernels"
RUNTIME=$(command -v apptainer || command -v singularity)
# /usr/bin/time -v puts each step's peak host RSS in the log (10 um: ~200 GB of
# the 256 GB requested, extrapolated from 50/35/25 um on one GPU).
TIMEV=$([ -x /usr/bin/time ] && echo "/usr/bin/time -v" || true)
run() { $TIMEV "$RUNTIME" exec --nv --bind /SAN --bind "$TMPDIR" \
          --env PYTHONPATH="$REPO",PYTHONNOUSERSITE=1 "$CONTAINER" "$@"; }
step() {  # step <name> <cmd...>: run it, record exit code and seconds, carry on
  local name=$1 t0=$SECONDS rc=0; shift
  echo; echo "=== $name  ($(date '+%F %T'))"
  "$@" || rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - t0))" >> "$OUT/steps.tsv"
  echo "=== $name: exit $rc after $((SECONDS - t0)) s"
}

cd "$REPO"
printf 'step\texit\tseconds\n' > "$OUT/steps.tsv"
{
  echo "job ${JOB_ID:-local} $(date -Is) on $(hostname)"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (NGPU=$NGPU)"
  echo "image: $CONTAINER"
  echo "repo:  $(git -C "$REPO" describe --always --dirty 2>/dev/null || echo unknown)"
  echo "mesh:  $MESH"
  echo; nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max --format=csv
  echo; nvidia-smi topo -m
  echo; free -g; nproc
  echo; stat -f -c 'SAN filesystem type: %T' /SAN/external/bridge_project_data
} > "$OUT/env.txt" 2>&1 || true
cat "$OUT/env.txt"

run python -m zvcfd probe --json > "$OUT/probe.json"
run python -m zvcfd probe --require cupy,cuda_device,zv_read_cells,zv_defer_presence

step tests       run python -m pytest -q -p no:cacheprovider tests --junitxml="$OUT/pytest.xml"
step validation  run python benchmarks/validation/run.py --out "$OUT/validation"
step bench_lbm   run python benchmarks/bench_lbm.py --json "$OUT/bench_lbm.json"
step bench_multi run python benchmarks/bench_multi.py --devices "$DEVICES" \
                     --bricks-per-gpu 400000 --chunk-bricks 16 --json "$OUT/bench_multi.json"
step bench_io_local run python benchmarks/bench_io.py --root "$TMPDIR/io" --workers "$NGPU" \
                     --json "$OUT/bench_io_local.json"
step bench_io_san   run python benchmarks/bench_io.py --root "$REPO/.io_bench" --workers "$NGPU" \
                     --json "$OUT/bench_io_san.json"
rm -rf "$REPO/.io_bench" "$TMPDIR/io"

for um in 20 10; do
  sed -e "s|/home/andrew/Downloads/mesh 1.msh|$MESH|" \
      -e "s|parallel: {gpus: 8}|parallel: {gpus: $NGPU}|" \
      "examples/coronary_${um}um_8gpu.yaml" > "$TMPDIR/coronary_${um}um.yaml"
  step "coronary_${um}um" run python -m zvcfd run "$TMPDIR/coronary_${um}um.yaml" --out "$OUT/coronary"
done

echo; echo "done $(date -Is)"; column -t "$OUT/steps.tsv"
