"""Multi-GPU scaling of :class:`zvcfd.lbm.MultiLBM`: weak and strong, with exchange cost.

On one GPU, ``--devices 0 --parts 1,2,4`` puts every partition on device 0:
that measures the *overhead* of partitioning and ghost exchange (it cannot
show speed-up). On the 8 x H100 node, ``--devices 0,1,2,3,4,5,6,7`` measures
real scaling over NVLink.

    python benchmarks/bench_multi.py --devices 0,1,2,3,4,5,6,7 --json results/bench_multi_h100.json

Domain: random tubes at brick resolution (every voxel of an active brick is
fluid), ``--bricks-per-gpu`` active bricks per partition for weak scaling,
the 1-partition size for strong scaling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tube_mask(grid: int, frac: float, seed: int = 0) -> np.ndarray:
    """Brick-level mask with roughly ``frac`` of bricks active, as random tubes."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[:grid, :grid, :grid].astype(np.float32)
    p = np.stack([z, y, x], -1)
    act = np.zeros((grid,) * 3, bool)
    while act.mean() < frac:
        a, b = rng.uniform(0, grid, 3), rng.uniform(0, grid, 3)
        ax = rng.integers(3)
        a[ax], b[ax] = 0, grid - 1
        d = b - a
        t = np.clip(((p - a) @ d) / (d @ d), 0, 1)
        act |= np.linalg.norm(p - (a + t[..., None] * d), axis=-1) <= rng.uniform(1.0, 2.5)
    return act


def domain_for(n_bricks_target: int, frac: float = 0.08):
    from zvcfd.domain import BrickDomain

    grid = int(np.ceil((n_bricks_target / frac) ** (1 / 3) / 4) * 4)
    return BrickDomain.from_brick_mask(tube_mask(grid, frac))


def measure(dom, parts, devices, steps, chunk_bricks):
    import cupy as cp

    from zvcfd.lbm import MultiLBM, SparseLBM

    if parts == 1 and len(devices) == 1:
        sim = SparseLBM(dom, tau=0.8, force=(1e-6, 0, 0), collision="trt")
        ex = 0.0
    else:
        sim = MultiLBM(dom, n_parts=parts, chunk_bricks=chunk_bricks, devices=devices,
                       tau=0.8, force=(1e-6, 0, 0), collision="trt")
        ex = sim.ghost_fraction
    sim.step(5)
    for d in set(devices):
        cp.cuda.Device(d).synchronize()
    t = time.time()
    sim.step(steps)
    for d in set(devices):
        cp.cuda.Device(d).synchronize()
    sec = time.time() - t
    cells = dom.fluid_cells
    return {"parts": parts, "devices": len(set(devices[:parts])), "fluid_cells": cells,
            "bricks": dom.n_bricks, "ghost_fraction": ex, "ms_per_step": 1e3 * sec / steps,
            "glups": cells * steps / sec / 1e9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="0")
    ap.add_argument("--parts", default=None, help="comma list; default 1,2,4,... up to #devices")
    ap.add_argument("--bricks-per-gpu", type=int, default=20000, dest="bpg")
    ap.add_argument("--chunk-bricks", type=int, default=8, dest="chunk_bricks")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--json")
    args = ap.parse_args()
    devices = [int(d) for d in args.devices.split(",")]
    if args.parts:
        parts_list = [int(p) for p in args.parts.split(",")]
    else:
        parts_list = [p for p in (1, 2, 4, 8, 16) if p <= len(devices)]
    rows = []
    base = domain_for(args.bpg)
    for mode in ("strong", "weak"):
        for p in parts_list:
            dom = base if mode == "strong" else (base if p == 1 else domain_for(args.bpg * p))
            r = measure(dom, p, devices, args.steps, args.chunk_bricks)
            r["mode"] = mode
            rows.append(r)
            print(f"{mode:<6} parts {p:>2} on {r['devices']} device(s): {r['fluid_cells']:>12,} "
                  f"cells  {r['ms_per_step']:8.2f} ms/step  {r['glups']:6.2f} GLUPS  "
                  f"ghosts {100 * r['ghost_fraction']:5.1f}%", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
