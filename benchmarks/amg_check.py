"""Operator coarsening vs geometry coarsening, on the Darcy pressure system.

Same lubrication system as ``zvcfd.solvers.lubrication``.  Compares
Jacobi-preconditioned CG (no hierarchy) with smoothed-aggregation AMG
(pyamg, CPU) used as a CG preconditioner -- the Fluent/CFX style of
coarsening the matrix rather than the geometry -- across domain lengths,
to see whether iterations stay flat as the domain grows.

    python benchmarks/amg_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyamg
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zvcfd.phantoms import vessel_network  # noqa: E402
from zvcfd.solvers.lubrication import conductance  # noqa: E402
from zvcfd.solvers.lubrication import system as lub_system


def system(fluid, rho_in=1.002, rho_out=0.998):
    """The lubrication system of zvcfd.solvers.lubrication, reservoirs on the x faces."""
    fixed = np.zeros_like(fluid)
    fixed[:, :, 0] = fluid[:, :, 0]
    fixed[:, :, -1] = fluid[:, :, -1]
    value = np.where(np.arange(fluid.shape[2])[None, None, :] < fluid.shape[2] // 2,
                     rho_in, rho_out) * np.ones(fluid.shape)
    A, b, _ = lub_system(fluid, fixed, value, conductance(fluid))
    return A, b



def main():
    out = []
    for nx in (256, 512, 1024):
        fluid = vessel_network((128, 128, nx))
        A, b = system(fluid)
        n = A.shape[0]
        x0 = np.full(n, 1.0)
        res = []
        t = time.time()
        Minv = sp.diags(1.0 / A.diagonal())
        _, _ = sp.linalg.cg(A, b, x0=x0, M=Minv, rtol=1e-8, maxiter=50000,
                            callback=lambda x: res.append(1))
        t_cg = time.time() - t
        t = time.time()
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=500)
        t_setup = time.time() - t
        r2 = []
        t = time.time()
        ml.solve(b, x0=x0, tol=1e-8, accel="cg", residuals=r2)
        t_amg = time.time() - t
        cx = ml.operator_complexity()
        row = {"nx": nx, "unknowns": n, "jacobi_cg_iters": len(res), "jacobi_cg_s": t_cg,
               "amg_levels": len(ml.levels), "amg_cg_iters": len(r2) - 1,
               "amg_setup_s": t_setup, "amg_solve_s": t_amg, "operator_complexity": cx}
        out.append(row)
        print(f"nx={nx:5d}  n={n:9d}  Jacobi-CG {len(res):6d} it ({t_cg:6.1f} s)   "
              f"SA-AMG-CG {len(r2) - 1:3d} it, {len(ml.levels)} levels, "
              f"setup {t_setup:5.1f} s + solve {t_amg:5.1f} s (CPU), op. complexity {cx:.2f}")
    with open(Path(__file__).parent / "results" / "amg_check.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
