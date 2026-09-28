"""Throughput of the zvcfd LBM kernels against this GPU's measured copy bandwidth.

    python benchmarks/bench_lbm.py [--json results/bench_lbm_<gpu>.json]

Reports MLUPS per fluid cell, effective bandwidth (bytes a pull step must
move per fluid cell) and that as a fraction of a device-to-device copy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zvcfd.domain import BrickDomain  # noqa: E402
from zvcfd.lbm import DenseLBM, SparseLBM  # noqa: E402
from zvcfd.phantoms import porous_spheres, vessel_tree  # noqa: E402


def copy_bandwidth(nbytes=1 << 30, reps=20) -> float:
    a = cp.empty(nbytes // 4, cp.float32)
    b = cp.empty_like(a)
    b[...] = a
    s, e = cp.cuda.Event(), cp.cuda.Event()
    s.record()
    for _ in range(reps):
        b[...] = a
    e.record()
    e.synchronize()
    return 2 * nbytes * reps / (cp.cuda.get_elapsed_time(s, e) / 1e3)


def time_steps(sim, fluid_cells, steps=50, warm=5) -> dict:
    sim.step(warm)
    cp.cuda.Device().synchronize()
    s, e = cp.cuda.Event(), cp.cuda.Event()
    s.record()
    sim.step(steps)
    e.record()
    e.synchronize()
    sec = cp.cuda.get_elapsed_time(s, e) / 1e3
    mlups = fluid_cells * steps / sec / 1e6
    return {"mlups": mlups, "gbs": mlups * 1e6 * sim.bytes_per_update / 1e9,
            "ms_per_step": 1e3 * sec / steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    props = cp.cuda.runtime.getDeviceProperties(0)
    bw = copy_bandwidth()
    print(f"{props['name'].decode()}: device copy {bw / 1e9:.0f} GB/s")
    rows = []

    def run(label, sim, fluid, frac_active=None):
        r = time_steps(sim, fluid)
        r.update(case=label, fluid_cells=fluid, of_copy=r["gbs"] * 1e9 / bw,
                 active_brick_fraction=frac_active,
                 device_gb=getattr(sim, "device_bytes", None) and sim.device_bytes / 1e9)
        rows.append(r)
        print(f"{label:<42} {r['mlups']:8.0f} MLUPS  {r['gbs']:6.0f} GB/s "
              f"({100 * r['of_copy']:4.0f}% of copy)  {r['ms_per_step']:7.2f} ms/step"
              + (f"  bricks {100 * frac_active:.1f}%" if frac_active is not None else ""))
        del sim
        cp.get_default_memory_pool().free_all_blocks()

    n = args.n
    open_box = np.zeros((n, n, n), np.uint8)
    for half in (False, True):
        tag = "fp16" if half else "fp32"
        run(f"dense {tag} open {n}^3", DenseLBM(open_box, force=(1e-6, 0, 0), half=half), n ** 3)
        sp = SparseLBM(BrickDomain.from_flags(open_box, periodic=True), force=(1e-6, 0, 0),
                       half=half)
        run(f"sparse {tag} open {n}^3", sp, sp.fluid_cells, sp.domain.active_fraction)

    porous = porous_spheres(n, porosity_target=0.45, radius=6)
    fluid = int((porous == 0).sum())
    for half in (False, True):
        tag = "fp16" if half else "fp32"
        run(f"dense {tag} porous phi=0.45 {n}^3", DenseLBM(porous, force=(1e-6, 0, 0),
                                                              half=half), fluid)
        sp = SparseLBM(BrickDomain.from_flags(porous, periodic=True), force=(1e-6, 0, 0),
                       half=half)
        run(f"sparse {tag} porous phi=0.45 {n}^3", sp, sp.fluid_cells, sp.domain.active_fraction)

    t = time.time()
    shape = (512, 512, 512)
    vessels = vessel_tree(shape, n_vessels=60, r_min=3, r_max=9)
    print(f"vessel phantom {shape}: fluid {100 * (vessels == 0).mean():.2f}% "
          f"(built in {time.time() - t:.0f} s)")
    for half in (False, True):
        tag = "fp16" if half else "fp32"
        sp = SparseLBM(BrickDomain.from_flags(vessels, periodic=True), force=(1e-6, 0, 0),
                       half=half)
        run(f"sparse {tag} vessels 512^3", sp, sp.fluid_cells, sp.domain.active_fraction)
    dense_need = 512 ** 3 * (2 * 19 * 4 + 1) / 1e9
    print(f"(dense fp32 at 512^3 would need {dense_need:.1f} GB; this GPU has "
          f"{cp.cuda.Device().mem_info[1] / 1e9:.1f} GB)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"gpu": props["name"].decode(), "copy_gbs": bw / 1e9, "rows": rows}, fh,
                      indent=1)


if __name__ == "__main__":
    main()
