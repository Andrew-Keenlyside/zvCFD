"""Speed of the GPU finite-volume solver on the SimVascular coronary case (398 k nodes, 2.1 M tets).

    ZVCFD_AMGX_LIB=... PYTHONPATH=. python benchmarks/fv/speed_simvascular.py \\
        [--steps 5] [--loops 5] [--linear amgx] [--preset robust] [--precision mixed] \\
        [--rtol 0.1] [--kernels fast] [--capture DIR] [--tag NAME]

The case is zvcfd-be's (``benchmarks/simvascular``: SimVascular's own mesh
from its Zarr Vectors collection, SimVascular's solution as the boundary
data, BDF2 at 1 ms). Runs a few time steps and records, per outer
iteration, the time in each stage (assembly, linear set-up, linear solve,
the rest), the linear iterations and whether each solve converged; with
``--capture`` it also saves the first few scaled linear systems (for the
offline solver study in ``benchmarks/fv/linear_study.py``).

Results: ``benchmarks/results/fv/speed_simvascular_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "simvascular"))
import fvcase  # noqa: E402
import vmrcase as vc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--loops", type=int, default=5)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--linear", default="amgx")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--rtol", type=float, default=0.1)
    ap.add_argument("--kernels", default="fast")
    ap.add_argument("--store-geometry", default="auto")
    ap.add_argument("--capture", default=None)
    ap.add_argument("--capture-n", type=int, default=3)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    import cupy as cp

    from zvcfd.fv.reference import Fluid
    from zvcfd.fv.solver import GPUSolver

    t0 = time.time()
    store = vc.VMR_DIR / "meshes" / "coronary.zvmesh"
    case = fvcase.build(store=store, log=lambda *m: None)
    data = fvcase.SVBoundaryData(case)
    bcs = fvcase.boundary_conditions(case, data, backflow=0.2)
    t_case = time.time() - t0
    opts = {}
    if a.linear in ("amgx", "auto"):
        opts["precision"] = a.precision
    if a.preset and a.linear == "amgx":
        opts["preset"] = a.preset
    sg = {"auto": "auto", "true": True, "tets": "tets", "false": False}[a.store_geometry.lower()]
    cp.cuda.Device().synchronize()
    t1 = time.time()
    sim = GPUSolver(case.mesh, Fluid(rho=vc.RHO, mu=vc.MU), bcs, linear=a.linear,
                    linear_rtol=a.rtol, linear_options=opts, kernels=a.kernels,
                    store_geometry=sg)
    x0 = data.at_nodes(0.0)
    sim.initialise(U=x0[:, :3].astype(float), P=x0[:, 3].astype(float))
    cp.cuda.Device().synchronize()
    t_setup = time.time() - t1
    captured = []
    if a.capture:
        cap = Path(a.capture)
        cap.mkdir(parents=True, exist_ok=True)
        orig = sim.linear.solve

        def solve(A, b, x0=None, rtol=None):
            if len(captured) < a.capture_n:
                i = len(captured)
                np.savez(cap / f"system_{i}.npz", indptr=cp.asnumpy(A.indptr),
                         indices=cp.asnumpy(A.indices), data=cp.asnumpy(A.data),
                         b=cp.asnumpy(b), x0=cp.asnumpy(x0))
                captured.append(str(cap / f"system_{i}.npz"))
            return orig(A, b, x0, rtol=rtol)

        sim.linear.solve = solve
    pool = cp.get_default_memory_pool()
    t2 = time.time()
    rep = sim.solve_transient(a.dt, a.steps, scheme="bdf2", loops=a.loops)
    cp.cuda.Device().synchronize()
    wall = time.time() - t2
    h = rep.history
    li = np.array([r["linear_iterations"] for r in h])
    out = {"nodes": sim.N, "elements": case.mesh.counts(), "blocks": int(sim.pattern[1].size),
           "kernels": a.kernels, "linear": a.linear, "preset": a.preset, "precision": a.precision,
           "rtol": a.rtol, "steps": a.steps, "loops": a.loops, "case_s": t_case,
           "setup_s": t_setup, "wall_s": wall, "s_per_step": wall / a.steps,
           "outer_iterations": len(h),
           "per_outer": {k: float(np.mean([r[k] for r in h])) for k in
                         ("seconds", "assemble_s", "linear_setup_s", "linear_solve_s")},
           "linear_iterations_mean": float(li.mean()), "linear_iterations_max": int(li.max()),
           "not_converged": int(sum(not r["linear_converged"] for r in h)),
           "final_rms": {k: h[-1][k] for k in ("rms_u", "rms_v", "rms_w", "rms_p")},
           "imbalance_last": h[-1]["imbalance"], "device_pool_gb": pool.total_bytes() / 1e9,
           "store_mode": getattr(sim.asm, "store_mode", None), "captured": captured}
    out["per_outer"]["other_s"] = out["per_outer"]["seconds"] - out["per_outer"]["assemble_s"] \
        - out["per_outer"]["linear_setup_s"] - out["per_outer"]["linear_solve_s"]
    tag = a.tag or f"{a.kernels}-{a.linear}-{a.preset or 'default'}-{a.precision}-rtol{a.rtol:g}"
    dest = ROOT / "benchmarks" / "results" / "fv" / f"speed_simvascular_{tag}.json"
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
