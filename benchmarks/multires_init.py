"""Does a coarse-level (or cheap fine-level) solve shorten the fine LBM solve?

Pressure-driven flow through a synthetic vessel network, reservoirs on the
two x faces: the slow case for LBM, because the pressure field settles
along long, narrow channels.

``--init full|rho``: nested iteration down an image pyramid. Each level is
a 2x coarsening of the fine mask (as an OME-Zarr pyramid gives), solved
with diffusive scaling at fixed tau (a coarse lattice sees 4x the density
drop and 2x the lattice velocity) and prolonged as the next level's start.
``rho`` prolongs density only.

``--init darcy``: Fluent-style "hybrid initialisation", but geometry-aware:
the lubrication pressure solve (``zvcfd.solvers.lubrication``) on the fine
mask, its density and Darcy velocity as the start.

Metric: flux through the mid-plane against a fine reference run to steady
state from rest. Cost is in fine-step equivalents (a level-k step costs
8^-k of a fine one; the lubrication solve is converted by wall time).

    python benchmarks/multires_init.py [--levels 2] [--rule majority|any] [--init full|rho|darcy]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zvcfd.lbm import DenseLBM, equilibrium  # noqa: E402
from zvcfd.phantoms import coarsen_mask, reservoir_flags, vessel_network  # noqa: E402
from zvcfd.solvers import lubrication  # noqa: E402


def mid_flux(sim: DenseLBM) -> float:
    f = sim.fields()
    x = sim.nx // 2
    fluid = sim.flag.reshape(sim.shape)[:, :, x] == 0
    return float((f["rho"][:, :, x] * f["ux"][:, :, x] * fluid).sum())


def host_state(sim: DenseLBM):
    """(rho, ux, uy, uz) on the host, rho = 1 and u = 0 outside the fluid."""
    f = {k: cp.asnumpy(v) for k, v in sim.fields().items()}
    solid = cp.asnumpy(sim.flag.reshape(sim.shape)) == 1
    f["rho"][solid] = 1.0
    return f["rho"], f["ux"], f["uy"], f["uz"]


def run_until(sim, *, check=200, target=None, rel=None, stable=1e-5, max_steps=400_000):
    """Step until the flux is within ``rel`` of ``target`` (or, with no target,
    changes by less than ``stable`` relative over 5 checks). Returns history, seconds."""
    hist = [(sim.steps, mid_flux(sim))]
    t0 = time.time()
    if target is not None and abs(hist[0][1] - target) <= rel * abs(target):
        return hist, 0.0
    while sim.steps < max_steps:
        sim.step(check)
        q = mid_flux(sim)
        hist.append((sim.steps, q))
        if target is not None:
            if abs(q - target) <= rel * abs(target):
                break
        elif len(hist) > 6:
            q5 = hist[-6][1]
            if q != 0 and abs(q - q5) <= stable * abs(q):
                break
    cp.cuda.Device().synchronize()
    return hist, time.time() - t0


def prolong(rho_c, ux, uy, uz, fluid_c, fluid_f, velocity=True):
    """Coarse lattice fields -> fine initial populations (diffusive scaling)."""
    idx = ndimage.distance_transform_edt(~fluid_c, return_distances=False, return_indices=True)

    def up(a):
        return ndimage.zoom(a[tuple(idx)], 2, order=1, mode="nearest", grid_mode=True)

    rho_f = 1.0 + (up(rho_c) - 1.0) / 4.0
    u = [up(c) / 2.0 * velocity for c in (ux, uy, uz)]
    for c in u:
        c[~fluid_f] = 0.0
    rho_f[~fluid_f] = 1.0
    return equilibrium(rho_f.astype(np.float32), *(c.astype(np.float32) for c in u))


def darcy_start(fluid, rho_in, rho_out, tau):
    """Lubrication solve for density (as pressure) and Darcy velocity, as populations."""
    fixed = np.zeros_like(fluid)
    fixed[:, :, 0] = fluid[:, :, 0]
    fixed[:, :, -1] = fluid[:, :, -1]
    value = np.where(np.arange(fluid.shape[2])[None, None, :] < fluid.shape[2] // 2,
                     rho_in, rho_out) * np.ones(fluid.shape)
    nu = (tau - 0.5) / 3.0
    r = lubrication.solve(fluid, fixed, value, mu=3.0 * nu, method="cg", tol=1e-6)
    rho = np.where(fluid, np.nan_to_num(r.pressure, nan=1.0), 1.0).astype(np.float32)
    uz, uy, ux = (c.astype(np.float32) for c in r.velocity)
    print(f"lubrication start: {r.iterations} CG iterations, {r.seconds:.2f} s ({r.method})")
    return equilibrium(rho, ux, uy, uz), r.seconds, r.iterations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="128,128,512")
    ap.add_argument("--levels", type=int, default=2, help="number of coarse levels (1-3)")
    ap.add_argument("--rule", default="majority", choices=["majority", "any"])
    ap.add_argument("--init", default="full", choices=["full", "rho", "darcy"])
    ap.add_argument("--drho", type=float, default=0.004)
    ap.add_argument("--json")
    args = ap.parse_args()
    shape = tuple(int(s) for s in args.shape.split(","))
    tau = 1.0
    if args.init == "darcy":
        args.levels = 0

    fine = vessel_network(shape)
    masks = [fine]
    for _ in range(args.levels):
        masks.append(coarsen_mask(masks[-1], args.rule))
    for k, m in enumerate(masks):
        lab = ndimage.label(m)[0]
        perc = np.intersect1d(np.unique(lab[:, :, 0]), np.unique(lab[:, :, -1])).size > 1
        print(f"level {k}: {m.shape}, fluid {100 * m.mean():.2f}%, percolates={perc}")

    d0 = args.drho
    ref = DenseLBM(reservoir_flags(fine), tau=tau, rho_in=1 + d0 / 2, rho_out=1 - d0 / 2)
    hist_ref, t_ref = run_until(ref, stable=2e-6)
    q_ref = hist_ref[-1][1]
    print(f"reference: {ref.steps} steps to stability, {t_ref:.1f} s, Q={q_ref:.5g}")

    def steps_to(hist, tol):
        return next((s for s, q in hist if abs(q - q_ref) <= tol * abs(q_ref)), None)

    cold = {tol: steps_to(hist_ref, tol) for tol in (1e-2, 1e-3)}
    ms_step = t_ref / max(ref.steps, 1)
    print(f"from rest: {cold[1e-2]} steps to 1%, {cold[1e-3]} steps to 0.1% "
          f"({ms_step * 1e3:.2f} ms/step)")
    del ref
    cp.get_default_memory_pool().free_all_blocks()

    init = None
    coarse_equiv = 0.0
    wall = 0.0
    per_level = []
    for k in range(args.levels, 0, -1):
        m = masks[k]
        drho = d0 * 4 ** k
        sim = DenseLBM(reservoir_flags(m), tau=tau, rho_in=1 + drho / 2, rho_out=1 - drho / 2,
                       init=init)
        hist, t = run_until(sim, stable=1e-4)
        q_pred = hist[-1][1] * 2 ** k        # flux in fine lattice units
        coarse_equiv += sim.steps / 8 ** k
        wall += t
        per_level.append({"level": k, "shape": m.shape, "steps": sim.steps, "seconds": t,
                          "flux_error_vs_fine": q_pred / q_ref - 1})
        print(f"level {k}: {sim.steps} steps, {t:.1f} s, predicts Q within "
              f"{100 * (q_pred / q_ref - 1):+.1f}% of the fine answer")
        init = prolong(*host_state(sim), m, masks[k - 1], velocity=args.init == "full")
        del sim
        cp.get_default_memory_pool().free_all_blocks()

    if args.init == "darcy":
        init, t_cg, n_it = darcy_start(fine, 1 + d0 / 2, 1 - d0 / 2, tau)
        wall += t_cg
        coarse_equiv += t_cg / ms_step
        per_level.append({"lubrication_cg_iterations": n_it, "lubrication_cg_seconds": t_cg})

    warm = DenseLBM(reservoir_flags(fine), tau=tau, rho_in=1 + d0 / 2, rho_out=1 - d0 / 2,
                    init=init)
    q0 = mid_flux(warm)
    out = {}
    for tol in (1e-2, 1e-3):
        hist, t = run_until(warm, target=q_ref, rel=tol)
        wall += t
        total = warm.steps + coarse_equiv
        out[tol] = {"fine_steps": warm.steps, "fine_equiv_total": total,
                    "speedup_steps": cold[tol] / total if cold[tol] else None,
                    "wall_s": wall, "wall_cold_s": cold[tol] * ms_step if cold[tol] else None}
        print(f"to {100 * tol:g}%: warm start {warm.steps} fine steps + {coarse_equiv:.0f} "
              f"fine-equivalent start-up = {total:.0f} vs {cold[tol]} from rest "
              f"-> {cold[tol] / total:.1f}x")
    print(f"(initial warm-start flux error {100 * (q0 / q_ref - 1):+.1f}%)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"shape": shape, "rule": args.rule, "init": args.init,
                       "levels": args.levels, "fluid_fraction": float(fine.mean()), "cold": cold,
                       "ms_per_fine_step": ms_step * 1e3, "per_level": per_level,
                       "warm": {str(k): v for k, v in out.items()},
                       "initial_warm_flux_error": q0 / q_ref - 1}, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
