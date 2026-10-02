# Benchmarks

Every number on these pages was measured with the scripts in
`benchmarks/`, and its raw output is in `benchmarks/results/`. The one
exception is the [comparison](comparison.md): it combines our measurements
with published figures through the roofline model, and says so line by
line. The [OpenFOAM comparison](openfoam.md) is its measured counterpart:
both codes run on the same problems on the same workstation.

**Hardware measured so far:** one workstation, AMD Threadripper PRO 5955WX
(16 cores / 32 threads), 251 GB RAM, one NVIDIA RTX A2000 12 GB (251 GB/s
device copy bandwidth), local NVMe (ext4). **No H100 measurement has been
made yet.** H100 figures are extrapolations, and milestone 0 of the
[roadmap](../feasibility/roadmap.md) replaces them.

| Suite | Question | Script | Page |
|---|---|---|---|
| Kernels | How close to the memory roofline do the LBM kernels run, dense and sparse, fp32 and fp16? | `bench_lbm.py` | [Kernels](kernels.md) |
| I/O | How fast can 8 workers write and read a snapshot: ZV bricks vs dense Zarr vs Icechunk? GPU reads? | `bench_io.py`, `bench_gpu_read.py` | [I/O](io.md) |
| Multiresolution | Do coarse levels or a cheap pressure solve shorten the fine solve? Does AMG scale? | `multires_init.py`, `amg_check.py` | [Multiresolution](multiresolution.md) |
| OpenFOAM | Same problems, same workstation: do zvCFD and OpenFOAM v2506 agree, and which gets there first? | `openfoam/voxel_compare.py`, `openfoam/coronary.py`, `openfoam/tau_scan.py` | [OpenFOAM](openfoam.md) |
| FV solver speed | Where does the GPU finite-volume solver spend its time, what did the speed review gain, and how does it compare with OpenFOAM, the CPU reference and the LBM on one pipe? | `fv/profile_iteration.py`, `fv/amgx_smoothers.py`, `fv/compare_methods.py`, `fv/speed_simvascular.py` | [FV solver speed](fv_speed.md) |
| Comparison | How does that translate against Fluent, CFX, STAR-CCM+, OpenFOAM, SimVascular and GPU LBM codes? | `estimates.py` | [Comparison](comparison.md) |

## Reproducing

```bash
export PYTHONPATH=/path/to/zarr-vectors-py:.        # gpu-backend branch
python benchmarks/bench_lbm.py --json benchmarks/results/bench_lbm_<gpu>.json
python benchmarks/bench_io.py --root <scratch on the target filesystem> --workers 8 \
       --serial-baseline --keep zv-flat-none,zv-shard2-zstd --json benchmarks/results/bench_io_<host>.json
python benchmarks/bench_gpu_read.py --root <same scratch> --json benchmarks/results/bench_gpu_read_<gpu>.json
python benchmarks/multires_init.py --levels 2 --init full --json benchmarks/results/multires_L2_full_512.json
python benchmarks/multires_init.py --init darcy --json benchmarks/results/multires_darcy_512.json
python benchmarks/amg_check.py
python benchmarks/estimates.py > benchmarks/results/estimates.md
python benchmarks/figures.py
```

`bench_io.py` needs Icechunk for its last variant; everything else needs
the `gpu` extra and a CUDA device (`amg_check.py` and `estimates.py` run
on the CPU).

## Correctness before speed

The kernels being timed are verified first (`pytest`), and the solver is
validated against exact solutions, standard benchmarks and OpenFOAM in
[Validation](../validation/index.md):

- The sparse-brick kernel is **bit-identical** to the dense kernel on a
  periodic porous sample.
- Plane Poiseuille flow matches the analytic profile to **10⁻⁴** of its
  maximum at any τ (TRT; measured 2 × 10⁻⁶).
- Brick stores round-trip exactly, including through GPU-side decode.

```{toctree}
:hidden:

kernels
io
multiresolution
openfoam
fv_speed
comparison
```
