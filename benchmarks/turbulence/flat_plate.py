"""NASA TMR zero-pressure-gradient flat plate with the k-kL model (k-kL-MEAH2015m).

    PYTHONPATH=. python benchmarks/turbulence/flat_plate.py [35 69 137 273]

Re = 5 million per unit length; the TMR farfield turbulence for M = 0.2. The
grids are the Resource's, one cell deep (span four wall spacings). The flow's
linear systems are solved directly (SuperLU on the host): on these slabs the
iterative solvers stall once the eddy viscosity is on (docs/spec/turbulence.md).
The false time step grows from 0.05 to ``--false-dt``. Writes
benchmarks/results/turbulence/flat_plate_<ni>.npz (cf along the plate, the
x = 0.97 profile) and prints cf at x = 0.97 against the TMR codes (k-kL,
0.0027 on the finest grids, NASA/TM-2015-218968 Fig. 2) and Karman-Schoenherr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import meshes as M  # noqa: E402
from common import save_fields, solve_ramped  # noqa: E402

from zvcfd.fv.reference import Fluid  # noqa: E402
from zvcfd.fv.solver import GPUSolver  # noqa: E402
from zvcfd.fv.turbulence import KkLModel, tmr_freestream  # noqa: E402

RES = Path(__file__).parents[1] / "results" / "turbulence"
U, RHO, RE = 1.0, 1.0, 5.0e6
MU = RHO * U * 1.0 / RE


def karman_schoenherr(re_theta):
    """cf from 1/cf = 17.08 (log10 Re_theta)^2 + 25.11 log10 Re_theta + 6.012."""
    lg = np.log10(re_theta)
    return 1.0 / (17.08 * lg ** 2 + 25.11 * lg + 6.012)


def run(ni: int, *, false_dt: float = 1.0, iterations: int = 1000, tol: float = 1e-8,
        linear: str = "host-direct", log=print):
    x2, y2 = M.read_plot3d_2d(M.TMR / "FlatPlate" / "Grids" / M.FLAT[ni])
    h1 = float(np.diff(y2[:2, -1]).min())                      # first cell at the outflow
    m = M.flat_plate(ni, span=4 * h1)
    ids = {z.name: k for k, z in m.zones.items()}
    vin = lambda x, t=0.0: np.tile([U, 0.0, 0.0], (len(x), 1))  # noqa: E731
    bcs = {ids["inflow"]: {"kind": "velocity", "value": vin},
           ids["outflow"]: {"kind": "pressure", "value": 0.0},
           ids["top"]: {"kind": "pressure", "value": 0.0}, ids["ahead"]: {"kind": "symmetry"},
           ids["plate"]: {"kind": "wall"}, ids["zmin"]: {"kind": "symmetry"},
           ids["zmax"]: {"kind": "symmetry"}}
    opts = {"linear_options": {"preset": "robust"}} if linear == "amgx" else {}
    s = GPUSolver(m, Fluid(RHO, MU), bcs, linear=linear, linear_rtol=0.1, dt=false_dt, **opts)
    s.initialise(U=np.tile([U, 0.0, 0.0], (m.n_nodes, 1)), P=np.zeros(m.n_nodes))
    k_inf, phi_inf = tmr_freestream(RHO, MU, U, 0.2)
    tm = KkLModel(s, k_inf, phi_inf).attach()
    rep = solve_ramped(s, false_dt, max_iterations=iterations, tol=tol)
    wall_s = rep.seconds
    nodes, tau = s.wall_shear()
    xw = m.nodes[nodes]
    on = (np.abs(xw[:, 2]) < 1e-12) & (xw[:, 0] > 1e-6)
    order = np.argsort(xw[on, 0])
    xs, cf = xw[on, 0][order], (tau[on, 0] / (0.5 * RHO * U ** 2))[order]
    cf097 = float(np.interp(0.97, xs, cf))
    # profile at x = 0.97: momentum thickness and u+, y+
    X = m.nodes
    Uf = s.cp.asnumpy(s.U)
    x97 = X[np.argmin(np.abs(X[:, 0] - 0.97)), 0]
    col = np.flatnonzero(np.isclose(X[:, 0], x97) & (X[:, 2] < 1e-12))
    col = col[np.argsort(X[col, 1])]
    yv, uv = X[col, 1], Uf[col, 0]
    bl = yv <= 0.1                                    # the layer is ~0.03 thick at x = 0.97
    ue = float(np.interp(0.1, yv, uv))
    theta = np.trapezoid((uv / ue * (1 - uv / ue))[bl], yv[bl])
    ut = np.sqrt(cf097 / 2) * U
    out = {"ni": ni, "nodes": m.n_nodes, "iterations": rep.iterations,
           "converged": bool(rep.converged), "wall_s": wall_s, "cf097": cf097,
           "re_theta": RHO * U * theta / MU,
           "cf_ks": float(karman_schoenherr(RHO * U * theta / MU)),
           "turbulence_linear_failures": tm.linear_failures,
           "flow_linear_converged": bool(rep.history[-1]["linear_converged"])}
    RES.mkdir(parents=True, exist_ok=True)
    np.savez(RES / f"flat_plate_{ni}.npz", x=xs, cf=cf, y=yv, u=uv, yplus=yv * ut * RHO / MU,
             uplus=uv / ut, k=s.cp.asnumpy(tm.k)[col], kl=s.cp.asnumpy(tm.phi)[col],
             mut=s.cp.asnumpy(tm.mu_t)[col] / MU, **{k: v for k, v in out.items()})
    (RES / f"flat_plate_{ni}.json").write_text(json.dumps(out, indent=1))
    save_fields(RES / f"fields_flat_plate_{ni}.npz", s, tm, MU)
    log(f"{ni}x: {m.n_nodes} nodes, {rep.iterations} its (converged {rep.converged}), "
        f"turbulence linear failures {tm.linear_failures}, "
        f"{wall_s:.0f} s; cf(0.97) = {cf097:.6f}; Re_theta {out['re_theta']:.0f}, "
        f"Karman-Schoenherr {out['cf_ks']:.6f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("grids", type=int, nargs="*", default=[35, 69, 137])
    ap.add_argument("--false-dt", type=float, default=1.0)
    ap.add_argument("--linear", default="host-direct")
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()
    for g in args.grids:
        run(g, false_dt=args.false_dt, linear=args.linear, tol=args.tol)
