#!/usr/bin/env bash
# The SimVascular comparison end to end (docs/validation/simvascular.md).
#
#   VMR_DIR=/hdd/data/zvcfd_vmr bash benchmarks/simvascular/run_all.sh
#
# Needs ~10 GB in $VMR_DIR, one GPU with >= 8 GB, and the `simvascular`
# extra (pyvista, vtk, scipy, matplotlib). About 4 h of GPU time on an
# RTX A2000 (1.7 h of it the 60 um run), plus the 8.9 GB download.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${VMR_DIR:=/hdd/data/zvcfd_vmr}"
export VMR_DIR
PY=${PY:-python}
S=https://stacks.stanford.edu/file/druid:dh173bw0673
mkdir -p "$VMR_DIR"

# 1. SimVascular's project and its rigid-wall results (Stanford Digital Repository)
[ -d "$VMR_DIR/0066_H_CORO_H" ] || {
  curl -fL -o "$VMR_DIR/0066_H_CORO_H.zip" "$S/svprojects/0066_H_CORO_H.zip"
  (cd "$VMR_DIR" && unzip -q 0066_H_CORO_H.zip)
}
for f in 0066_H_CORO_H_3D_RIGID.vtp 0066_H_CORO_H_3D_RIGID.vtu; do
  [ -f "$VMR_DIR/$f" ] || curl -fL -o "$VMR_DIR/$f" "$S/svresults/0066_H_CORO_H/$f"
done

# 2. SimVascular's answers: cap flows and pressures, probe points, volume samples
$PY benchmarks/simvascular/svref.py surface
$PY benchmarks/simvascular/svref.py probes
$PY benchmarks/simvascular/svref.py volume
$PY benchmarks/simvascular/svref.py wall

# 3. zvCFD: three resolutions, plus the Mach-number and inlet-profile checks
#    (and the straight-pipe outlet test, before and after the fix)
$PY benchmarks/simvascular/oblique_outlet.py
for args in "--voxel 100" "--voxel 80" "--voxel 60" "--voxel 100 --u-lat 0.07" \
            "--voxel 100 --inlet parabolic"; do
  $PY benchmarks/simvascular/run_zvcfd.py $args --cycles 2 --snapshots 160,265,530,800
done

# 3b. The finite-volume solver on SimVascular's own mesh (cropped, through a Zarr
#     Vectors mesh collection): one cycle, ~1.4 h on an RTX A2000; needs AmgX
#     (ZVCFD_AMGX_LIB) and ~8.5 GB of GPU memory
PYTHONPATH=. $PY benchmarks/simvascular/run_fv.py --linear-rtol 0.01

# 4. Compare, draw, render
$PY benchmarks/simvascular/compare.py
$PY benchmarks/simvascular/figures.py --best vmr0066-60um-sv
$PY benchmarks/simvascular/renders.py --best vmr0066-60um-sv --coarse vmr0066-100um-sv
