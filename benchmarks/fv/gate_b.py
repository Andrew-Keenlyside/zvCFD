"""Gate B of the finite-volume plan: AmgX against zvCFD's own ACM on a prism-layered tube.

    ZVCFD_AMGX_LIB=.../libamgxsh.so PYTHONPATH=. python benchmarks/fv/gate_b.py [nc ...]

A pipe mesh built like the coronary mesh (wedge inflation layers graded
towards the wall around a tetrahedral core), Re = 100 laminar flow
(parabolic inlet, pressure outlet, High Resolution). For each linear
solver:

1. one linear system, the linearisation after five outer iterations,
   solved to a relative residual of 10⁻⁶: iterations, reduction per
   iteration, setup and solve time, and peak device memory;
2. a whole steady solve to a relative change of 10⁻⁸, each outer
   iteration's system solved to 0.1 (CFX-style): outer iterations, linear
   iterations per outer iteration, time per outer iteration, total time.

Results: ``benchmarks/results/fv/gate_b.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from zvcfd.fv.linear import amgx_library
from zvcfd.fv.reference import Fluid
from zvcfd.fv.solver import GPUSolver
from zvcfd.mesh.generate import tube

ROOT = Path(__file__).resolve().parents[2]
R, L = 0.5, 4.0
MU = 0.01                    # U_mean 1 (parabolic peak 2), D = 1: Re = 100


def case(nc):
    m = tube(R, L, n_core=nc, n_ring=nc, n_axial=8 * nc, kind="mixed", layers=nc // 2,
             growth=0.85)
    zid = {z.name: k for k, z in m.zones.items()}

    def prof(x):
        r2 = (x[:, 0] ** 2 + x[:, 1] ** 2) / R ** 2
        return np.stack([0 * r2, 0 * r2, 2 * np.clip(1 - r2, 0, None)], 1)

    return m, {zid["inlet"]: {"kind": "velocity", "value": prof},
               zid["outlet"]: {"kind": "pressure", "value": 0.0}, zid["wall"]: {"kind": "wall"}}


def peak():
    import cupy as cp

    return cp.get_default_memory_pool().total_bytes() / 1e9


def run(nc, linear):
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    m, bcs = case(nc)
    t0 = time.time()
    name, opts = {"amgx-mixed": ("amgx", {"precision": "mixed"}),
                  "amgx-dilu": ("amgx", {"smoother": "dilu"})}.get(linear, (linear, None))
    g = GPUSolver(m, Fluid(1.0, MU), bcs, linear=name, linear_rtol=0.1, linear_options=opts)
    setup = time.time() - t0
    g.solve(max_iterations=5, tol=0.0)
    A, b = g.assemble()
    x0 = cp.concatenate([g.U, g.P[:, None]], 1).reshape(-1)
    x, info = g.linear.solve(A, b.reshape(-1), x0, rtol=1e-6)
    single = {"iterations": info.iterations, "residual": info.residual,
              "reduction_per_iteration": info.residual ** (1 / max(info.iterations, 1)),
              "setup_s": info.setup_seconds, "solve_s": info.seconds,
              "converged": info.converged, "device_pool_gb": peak()}
    if hasattr(g.linear, "stats"):
        single.update(g.linear.stats())
    if hasattr(g.linear, "release"):
        g.linear.release(1e-6)                 # its hierarchy would stay on the device
    del A, b, x, x0
    cp.get_default_memory_pool().free_all_blocks()
    t1 = time.time()
    rep = g.solve(max_iterations=300, tol=1e-8)
    total = time.time() - t1
    h = rep.history
    out = {"nc": nc, "nodes": m.n_nodes, "elements": m.counts(),
           "blocks": int(g.pattern[1].size), "setup_s": setup, "single": single,
           "steady": {"outer_iterations": rep.iterations, "converged": rep.converged,
                      "linear_per_outer": float(np.mean([r["linear_iterations"] for r in h])),
                      "linear_failures": int(sum(not r["linear_converged"] for r in h)),
                      "seconds_per_outer": float(np.mean([r["seconds"] for r in h])),
                      "assemble_s": float(np.mean([r["assemble_s"] for r in h])),
                      "linear_setup_s": float(np.mean([r["linear_setup_s"] for r in h])),
                      "linear_solve_s": float(np.mean([r["linear_solve_s"] for r in h])),
                      "total_s": total, "final_rms_u": h[-1]["rms_u"],
                      "final_rms_p": h[-1]["rms_p"], "imbalance": h[-1]["imbalance"]},
           "device_pool_gb": peak()}
    print(json.dumps({"linear": linear, **out}, default=float), flush=True)
    del g
    cp.get_default_memory_pool().free_all_blocks()
    return out


def main():
    levels = [int(a) for a in sys.argv[1:]] or [8, 12, 16, 20, 24]
    solvers = os.environ.get("GATE_B_SOLVERS", "").split(",") if os.environ.get("GATE_B_SOLVERS") \
        else ["acm"] + (["amgx", "amgx-mixed"] if amgx_library() else [])
    dest = ROOT / "benchmarks" / "results" / "fv" / os.environ.get("GATE_B_OUT", "gate_b.json")
    res = {s: [] for s in solvers}
    for nc in levels:                         # both solvers per size; written as it goes
        for s in solvers:
            try:
                res[s].append(run(nc, s))
            except Exception as exc:          # out of device memory at the largest sizes
                print(f"{s} nc={nc}: {type(exc).__name__}: {exc}", flush=True)
                res[s].append({"nc": nc, "error": f"{type(exc).__name__}: {exc}"})
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
            dest.write_text(json.dumps(res, indent=1, default=float))
    print("->", dest)


if __name__ == "__main__":
    main()
