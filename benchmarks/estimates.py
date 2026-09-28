"""Scenario estimates for docs/benchmarks/comparison.md, from zvcfd.perfmodel.

Every number the comparison page quotes for zvCFD or for the reference
packages comes out of this script, so the assumptions sit in one place:

- zvCFD rates: roofline x kernel efficiency, ``planning`` basis (the lower
  of the A2000 measurement and the published figure for tuned kernels),
  85% multi-GPU efficiency over NVLink.
- Unstructured FV (Fluent/OpenFOAM class): 0.12 M cell-iterations/s per CPU
  core; 200 M per H100 for a GPU-native FV solver (vendor-reported range
  150-300 M); 1.8 GB per million cells on the GPU.
- Transient LBM: acoustic scaling, lattice Mach <= 0.1 at the peak velocity,
  so dt = 0.1 dx / u_peak (tau then follows from the viscosity).
- Transient FV: 1 ms steps, 20 inner iterations each (a typical pulsatile
  hemodynamics setting).

    python benchmarks/estimates.py > benchmarks/results/estimates.md
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zvcfd.perfmodel import cpu_fv_estimate, estimate, fv_memory_gb, gpu_fv_estimate  # noqa: E402
from zvcfd.units import Lattice  # noqa: E402

NU_BLOOD = 3.5e-6


def hms(s: float) -> str:
    for unit, n in (("d", 86400), ("h", 3600), ("min", 60)):
        if s >= n:
            return f"{s / n:.1f} {unit}"
    return f"{s:.0f} s"


def coronary():
    """HiP-CT coronary lumen (the Simpleware/Fluent mesh): 628.6 mm^3 of lumen."""
    print("## HiP-CT coronary tree: one cardiac cycle (1 s), pulsatile, u_peak = 0.5 m/s\n")
    print("| Solver | Resolution | Cells | Memory | Hardware | Time per cycle |")
    print("|---|---|---:|---:|---|---:|")
    cells = 14_790_642
    iters = 1000 * 20
    print(f"| Fluent-class FV, CPU | Simpleware mesh (~35 um equiv.) | {cells:.3g} | "
          f"{cells / 1e6 * 2.5:.0f} GB RAM | 128 cores | {hms(cpu_fv_estimate(cells, iters))} |")
    print(f"| Fluent-class FV, CPU | same | {cells:.3g} | | 1024 cores | "
          f"{hms(cpu_fv_estimate(cells, iters, cores=1024))} |")
    print(f"| Fluent-class FV, GPU | same | {cells:.3g} | {fv_memory_gb(cells):.0f} GB | 1 x H100 | "
          f"{hms(gpu_fv_estimate(cells, iters))} |")
    for voxel_mm, fill in ((0.02, 0.55), (0.01, 0.7), (0.005, 0.82)):
        fluid = 628.6 / voxel_mm ** 3
        lat_dt = 0.1 * voxel_mm * 1e-3 / 0.5
        steps = int(1.0 / lat_dt)
        tau = 0.5 + 3 * NU_BLOOD * lat_dt / (voxel_mm * 1e-3) ** 2
        for method in ("lbm-fp32", "lbm-fp16"):
            for gpus in (1, 8):
                e = estimate(fluid, fill=fill, gpus=gpus, method=method, geometry="vessel",
                             steps=steps)
                t = hms(e.seconds) if e.fits else "does not fit"
                print(f"| zvCFD {method} | {voxel_mm * 1000:g} um voxels (tau {tau:.3f}) | "
                      f"{fluid:.3g} fluid | {e.memory_per_gpu_gb:.0f} GB/GPU | {gpus} x H100 | "
                      f"{t} ({steps:.2g} steps) |")
        if voxel_mm == 0.01:
            fv = fluid
            print(f"| Fluent-class FV at the same cell count | 10 um-equivalent mesh | {fv:.3g} | "
                  f"{fv_memory_gb(fv) / 1e3:.1f} TB GPU | 1024 cores | "
                  f"{hms(cpu_fv_estimate(fv, iters, cores=1024))} |")
    print()


def rock():
    """Digital-rock permeability: steady Stokes, ~3e4 LBM steps / ~2000 SIMPLE iterations."""
    print("## Digital rock, steady permeability (porosity 0.2)\n")
    print("| Image | Solver | Memory | Hardware | Time to steady state |")
    print("|---|---|---:|---|---:|")
    for n in (1024, 1536, 2048):
        box = float(n) ** 3
        fluid = 0.2 * box
        for method in ("lbm-fp32", "lbm-fp16"):
            e = estimate(fluid, box_cells=box, layout="dense", geometry="porous", gpus=8,
                         method=method, steps=30_000)
            t = hms(e.seconds) if e.fits else "does not fit"
            print(f"| {n}^3 | zvCFD {method}, dense | {e.memory_per_gpu_gb:.0f} GB/GPU | "
                  f"8 x H100 | {t} |")
        cells = fluid
        print(f"| {n}^3 | FV (snappy/Fluent mesh of the pore space) | "
              f"{fv_memory_gb(cells, 2.5) / 1e3:.2f} TB RAM | 1024 cores | "
              f"{hms(cpu_fv_estimate(cells, 2000, cores=1024))} |")
    print("\nReference points: GeoChemFoam (OpenFOAM DBS) solved 8e9 voxels in 1230 s on "
          "81,920 ARCHER2 cores; Palabos (A100) solved Berea 400^3 in under 10 min.\n")


def per_gpu():
    print("## Per-GPU throughput (H100 SXM, fluid-cell updates per second)\n")
    print("| Kernel | Efficiency basis | GLUPS |")
    print("|---|---|---:|")
    for method in ("lbm-fp32", "lbm-fp16"):
        for layout, geom in (("dense", "open"), ("sparse", "open"), ("sparse", "vessel")):
            for eff in ("measured", "target", "planning"):
                e = estimate(1e9, box_cells=1e9, layout=layout, geometry=geom, gpus=1,
                             method=method, efficiency=eff)
                print(f"| zvCFD {method} {layout} {geom} | {eff} | "
                      f"{e.updates_per_s / 1e9:.1f} |")
    print("| FluidX3D FP32 / FP16S dense | published, H100 SXM | 17.6 / 29.6 |")
    print("| waLBerla sparse coronary (FP64, A100) | published | 1.37 kernel-only |")
    print("| XLB (JAX), A100 | published | 1.43 |")
    print()


def lattice_note():
    lat = Lattice(dx=10e-6, nu=NU_BLOOD, tau=0.6)
    print(f"(10 um, tau 0.6: dt = {lat.dt:.3g} s, 1 m/s is {lat.u_lattice(1.0):.3f} lattice)\n")


if __name__ == "__main__":
    per_gpu()
    coronary()
    rock()
    lattice_note()
