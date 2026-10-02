"""Where one outer iteration's time goes, stage by stage, on a real or generated case.

    ZVCFD_AMGX_LIB=... PYTHONPATH=. python benchmarks/fv/profile_iteration.py [simvascular|tube NC]
        [--kernels fast|classic] [--linear amgx] [--iterations 4]

Every stage of ``GPUSolver`` (gradients, limiter, diagonal, assembly
kernel, boundary rows, residuals, row scaling, linear set-up and solve,
state update, mass flows, zone flows; with the fast kernels, the fused
boundary pass ``_assemble_fused``) is wrapped with a device-synchronised
timer; the table is the mean over the iterations after the first. The
wrappers add synchronisations of their own, so the per-outer time here is
an upper bound: ``speed_simvascular.py`` and ``compare_methods.py`` time
the unwrapped loop.
Results: ``benchmarks/results/fv/profile_<case>_<kernels>.json``.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def build(case, nc):
    from zvcfd.fv.reference import Fluid

    if case == "simvascular":
        sys.path.insert(0, str(ROOT / "benchmarks" / "simvascular"))
        import fvcase
        import vmrcase as vc

        c = fvcase.build(store=vc.VMR_DIR / "meshes" / "coronary.zvmesh", log=lambda *m: None)
        data = fvcase.SVBoundaryData(c)
        bcs = fvcase.boundary_conditions(c, data, backflow=0.2)
        x0 = data.at_nodes(0.0)
        return c.mesh, Fluid(rho=vc.RHO, mu=vc.MU), bcs, (x0[:, :3].astype(float),
                                                          x0[:, 3].astype(float)), 1e-3
    sys.path.insert(0, str(ROOT / "benchmarks" / "fv"))
    from gate_b import MU
    from gate_b import case as tube_case

    m, bcs = tube_case(nc)
    return m, Fluid(1.0, MU), bcs, None, None


def main():
    import cupy as cp

    from zvcfd.fv.solver import GPUSolver

    ap = argparse.ArgumentParser()
    ap.add_argument("case", default="tube", nargs="?")
    ap.add_argument("nc", type=int, default=16, nargs="?")
    ap.add_argument("--kernels", default="fast")
    ap.add_argument("--linear", default="amgx")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--options", default="{}")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    mesh, fluid, bcs, init, dt = build(a.case, a.nc)
    cp.cuda.Device().synchronize()
    t = time.time()
    s = GPUSolver(mesh, fluid, bcs, linear=a.linear, linear_rtol=0.1, kernels=a.kernels,
                  linear_options=json.loads(a.options))
    if init is not None:
        s.initialise(U=init[0], P=init[1])
    cp.cuda.Device().synchronize()
    setup = time.time() - t
    times = defaultdict(list)

    def wrap(obj, name, label=None):
        f = getattr(obj, name)

        @functools.wraps(f)
        def g(*args, **kw):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            out = f(*args, **kw)
            cp.cuda.Device().synchronize()
            times[label or name].append(time.perf_counter() - t0)
            return out

        setattr(obj, name, g)

    for n in ("gradient", "limiter", "diagonal", "assemble", "massflow"):
        wrap(s.asm, n, f"asm.{n}")
    for n in ("_assemble_stage1", "_assemble_stage2", "_dirichlet", "_lumped_pieces",
              "_finish_lumped", "_residuals", "_scale_rows", "update_mass_flows", "zone_flows",
              "_assemble_fused", "_zone_contrib"):
        if hasattr(s, n):
            wrap(s, n)
    wrap(s.linear, "solve", "linear.solve")
    if dt:
        s._time = ("bdf2", dt, [(s.U.copy(), {k: v.copy() for k, v in s.mdot.items()})], None)
    hist = []
    cp.cuda.Device().synchronize()
    t = time.time()
    s._iterate(a.iterations, 0.0, None, hist)
    wall = time.time() - t
    table = {k: float(np.mean(v[1:] if len(v) > 1 else v)) for k, v in times.items()}
    per = wall / a.iterations
    out = {"case": a.case, "nodes": s.N, "elements": mesh.counts(), "kernels": a.kernels,
           "setup_s": setup, "per_outer_s": per, "stages_s": table,
           "linear_iterations": [h["linear_iterations"] for h in hist],
           "linear_options": json.loads(a.options),
           "device_pool_gb": cp.get_default_memory_pool().total_bytes() / 1e9}
    tag = a.tag or f"{a.case}{'' if a.case == 'simvascular' else a.nc}_{a.kernels}"
    (ROOT / "benchmarks" / "results" / "fv" / f"profile_{tag}.json").write_text(
        json.dumps(out, indent=1))
    print(f"{a.case} {s.N:,} nodes, kernels {a.kernels}: setup {setup:.1f} s, "
          f"{per:.3f} s per outer iteration")
    for k, v in sorted(table.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {1e3 * v:9.1f} ms")


if __name__ == "__main__":
    main()
