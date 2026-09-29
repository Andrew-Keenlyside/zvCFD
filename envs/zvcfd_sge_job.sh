#!/bin/bash
# ---------------------------------------------------------------------------
# zvCFD milestone 0 on the UCL CS Grid Engine cluster (seymour2: 8x H100 80GB).
#
#   qsub -v MESH=/SAN/.../mesh1.msh envs/zvcfd_sge_job.sh
#
# Grid Engine, NOT SLURM: directives are #$, the job id is $JOB_ID, and the GPU
# prolog exports CUDA_VISIBLE_DEVICES. Follows BRIDGE's envs/bridge_sge_job.sh.
# Runs, in order: probe, kernel benchmark, multi-GPU scaling, I/O on local
# scratch and on /SAN, and the HiP-CT coronary case at 20 um and 10 um on
# every granted GPU. Results land in benchmarks/results/*_h100*.
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
# eight cards is unconfirmed on this cluster; the log prints what was granted.

set -euo pipefail
CONTAINER=${CONTAINER:-/SAN/external/bridge_project_data/containers/zvcfd-h100.sif}
REPO=${REPO:-$PWD}
MESH=${MESH:?set MESH to the coronary .msh path}
RESULTS=$REPO/benchmarks/results

[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: not a GPU allocation" >&2; exit 1; }
NGPU=$(printf '%s\n' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
DEVICES=$(seq -s, 0 $((NGPU - 1)))
echo "job ${JOB_ID:-local} on $(hostname): $NGPU GPU(s) [$CUDA_VISIBLE_DEVICES]"
export TMPDIR="${TMPDIR:-/tmp/zvcfd.$$}"; mkdir -p "$TMPDIR"
export CUPY_CACHE_DIR="$HOME/.cache/zvcfd-kernels"
RUNTIME=$(command -v apptainer || command -v singularity)
run() { "$RUNTIME" exec --nv --bind /SAN --bind "$TMPDIR" "$CONTAINER" "$@"; }

cd "$REPO"
run zvcfd probe --json | tee "$RESULTS/probe_h100.json"
run zvcfd probe --require cupy,cuda_device,zv_read_cells,zv_defer_presence

run python benchmarks/bench_lbm.py --json "$RESULTS/bench_lbm_h100.json"
run python benchmarks/bench_multi.py --devices "$DEVICES" --bricks-per-gpu 400000 \
    --chunk-bricks 16 --json "$RESULTS/bench_multi_h100.json"
run python benchmarks/bench_io.py --root "$TMPDIR/io" --workers "$NGPU" \
    --json "$RESULTS/bench_io_h100_local.json"
run python benchmarks/bench_io.py --root "$REPO/.io_bench" --workers "$NGPU" \
    --json "$RESULTS/bench_io_h100_san.json"
rm -rf "$REPO/.io_bench"
stat -f -c 'SAN filesystem type: %T' /SAN/external/bridge_project_data || true

for um in 20 10; do
  sed -e "s|/home/andrew/Downloads/mesh 1.msh|$MESH|" \
      -e "s|parallel: {gpus: 8}|parallel: {gpus: $NGPU}|" \
      "examples/coronary_${um}um_8gpu.yaml" > "$TMPDIR/coronary_${um}um.yaml"
  run zvcfd run "$TMPDIR/coronary_${um}um.yaml" --out "$RESULTS/coronary_h100"
done
