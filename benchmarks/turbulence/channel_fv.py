"""Fully developed channel with zvCFD's finite-volume solver and the k-kL model.

    PYTHONPATH=. python benchmarks/turbulence/channel_fv.py [--ny 64] [--re 2e4] [--cycles 10]

A plane channel of half height H = 1, 20 H long, one cell deep, 2·ny tanh-stretched
cells across; Re = U_b H / nu. The inflow profiles of u, k and kL are recycled from
three quarters of the way down the channel, ``--cycles`` times, until they stop
changing: the fully developed flow, for comparison with the 1-D solver of the same
equations (``channel_1d.py``, Re_tau = 942 at Re = 2 x 10^4). The flow is solved
directly (one-cell slab, see docs/spec/turbulence.md). Writes
benchmarks/results/turbulence/channel_fv.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zvcfd.fv.reference import Fluid
from zvcfd.fv.solver import GPUSolver
from zvcfd.fv.turbulence import KkLModel
from zvcfd.mesh.generate import box

RES = Path(__file__).parents[1] / "results" / "turbulence"


def run(ny: int = 64, re: float = 2.0e4, cycles: int = 10, length: float = 20.0, log=print):
    H, U, rho = 1.0, 1.0, 1.0
    mu = rho * U * H / re
    m = box((int(2 * length), ny, 1), (length, 2 * H, 0.2), kind="hex")
    yy = m.nodes[:, 1] / (2 * H)
    m.nodes[:, 1] = H * (1 + np.tanh(2.5 * (2 * yy - 1)) / np.tanh(2.5))
    ids = {z.name: k for k, z in m.zones.items()}
    x = m.nodes
    wall = np.isclose(x[:, 1], 0) | np.isclose(x[:, 1], 2 * H)
    yp = np.linspace(0, 2 * H, 401)
    up = np.ones_like(yp)
    kp = np.full_like(yp, 1e-3)
    pp = np.full_like(yp, 10 * mu * np.sqrt(1e-3) / 0.09 ** 0.25)
    for c in range(cycles):
        uin = up / np.trapezoid(up, yp) * 2 * H * U                 # keep the bulk velocity

        def vin(xi, t=0.0, u=uin):
            return np.c_[np.interp(xi[:, 1], yp, u), np.zeros(len(xi)), np.zeros(len(xi))]
        bcs = {ids["xmin"]: {"kind": "velocity", "value": vin},
               ids["xmax"]: {"kind": "pressure", "value": 0.0},
               ids["ymin"]: {"kind": "wall"}, ids["ymax"]: {"kind": "wall"},
               ids["zmin"]: {"kind": "symmetry"}, ids["zmax"]: {"kind": "symmetry"}}
        s = GPUSolver(m, Fluid(rho, mu), bcs, linear="host-direct", linear_rtol=0.1, dt=2.0)
        s.initialise(U=vin(x), P=np.zeros(m.n_nodes))

        def inflow(xi, k_=kp.copy(), p_=pp.copy()):
            return np.interp(xi[:, 1], yp, k_), np.interp(xi[:, 1], yp, p_)
        tm = KkLModel(s, 1e-3, float(pp.max()), inflow=inflow)
        kin, pin = inflow(x)
        tm.k = s.cp.asarray(np.where(wall, 0.0, kin))               # start from the inflow
        tm.phi = s.cp.asarray(np.where(wall, 0.0, pin))
        tm.mu_t = tm._eddy_viscosity()
        tm.attach()
        rep = s.solve(max_iterations=400, tol=1e-7)
        Uf, kk, ph = (s.cp.asnumpy(a) for a in (s.U, tm.k, tm.phi))
        col = np.flatnonzero(np.isclose(x[:, 0], 0.75 * length, atol=0.25) & (x[:, 2] < 1e-9))
        col = col[np.argsort(x[col, 1])]
        new_u = np.interp(yp, x[col, 1], Uf[col, 0])
        change = float(np.abs(new_u - up).max())
        up = new_u
        kp = np.maximum(np.interp(yp, x[col, 1], kk[col]), 0)
        pp = np.maximum(np.interp(yp, x[col, 1], ph[col]), 0)
        tau = mu * Uf[col[1], 0] / x[col[1], 1]                     # first node's gradient
        out = {"cycle": c, "iterations": rep.iterations, "converged": bool(rep.converged),
               "cf": 2 * tau / (rho * U ** 2), "uc_ub": float(np.interp(H, yp, up) / U),
               "re_tau": float(np.sqrt(tau / rho) * H * rho / mu), "profile_change": change}
        log(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}))
    re_b = 2 * re
    out |= {"re_b": re_b, "dean_cf": 0.073 * re_b ** -0.25, "dean_uc_ub": 1.28 * re_b ** -0.0116}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "channel_fv.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--re", type=float, default=2.0e4)
    ap.add_argument("--cycles", type=int, default=10)
    args = ap.parse_args()
    run(args.ny, args.re, args.cycles)
