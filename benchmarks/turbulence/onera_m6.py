"""ONERA M6 wing at low speed with the k-kL model: a three-dimensional demonstration.

    PYTHONPATH=. python benchmarks/turbulence/onera_m6.py [--re 1e6] [--alpha 3.06]

The M6 planform (Schmitt & Charpin 1979) with ONERA D sections, in a C-H grid
of hexahedra (``meshes.onera_m6``). The classic case is transonic
(M = 0.84, Re = 11.72 million on the mean aerodynamic chord); zvCFD is
incompressible, so this runs the same wing at the same incidence at low speed,
at Re = 1 million on the MAC, where the 113-section grid puts the first node at
y+ ~ 0.4. There is no low-speed experiment to compare with: lift and drag,
surface pressure at the experiment's span stations and the convergence are
reported as a demonstration of the model on a three-dimensional wing.
Writes benchmarks/results/turbulence/onera_m6.npz and .json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import meshes as M  # noqa: E402
from common import solve_ramped  # noqa: E402

from zvcfd.fv.reference import Fluid  # noqa: E402
from zvcfd.fv.solver import GPUSolver  # noqa: E402
from zvcfd.fv.turbulence import KkLModel, tmr_freestream  # noqa: E402

RES = Path(__file__).parents[1] / "results" / "turbulence"
U, RHO, MACH = 1.0, 1.0, 0.15
STATIONS = (0.20, 0.44, 0.65, 0.80, 0.90, 0.96)        # Schmitt & Charpin's span stations
S_REF = 0.5 * M.M6_ROOT * (1 + M.M6_TAPER) * M.M6_SPAN   # semispan planform area


def run(re: float = 1.0e6, alpha: float = 3.06, *, false_dt: float = 0.5,
        iterations: int = 600, tol: float = 1e-6, linear: str = "amgx",
        preset: str = "robust", log=print) -> dict:
    mu = RHO * U * M.M6_MAC / re
    m = M.onera_m6()
    ids = {z.name: k for k, z in m.zones.items()}
    a = np.radians(alpha)
    uinf = np.array([np.cos(a), np.sin(a), 0.0]) * U
    vin = lambda x, t=0.0: np.tile(uinf, (len(x), 1))  # noqa: E731
    bcs = {ids["farfield"]: {"kind": "velocity", "value": vin},
           ids["outflow"]: {"kind": "pressure", "value": 0.0},
           ids["wing"]: {"kind": "wall"}, ids["root"]: {"kind": "symmetry"},
           ids["side"]: {"kind": "symmetry"}}
    opts = {"linear_options": {"preset": preset}} if linear == "amgx" else {}
    s = GPUSolver(m, Fluid(RHO, mu), bcs, linear=linear, linear_rtol=0.1, dt=false_dt, **opts)
    s.initialise(U=np.tile(uinf, (m.n_nodes, 1)), P=np.zeros(m.n_nodes))
    tm = KkLModel(s, *tmr_freestream(RHO, mu, U, MACH)).attach()
    rep = solve_ramped(s, false_dt, max_iterations=iterations, tol=tol)
    q = 0.5 * RHO * U ** 2 * S_REF
    lift_dir, drag_dir = np.array([-np.sin(a), np.cos(a), 0.0]), uinf / U
    F = s.zone_forces()[ids["wing"]]
    P = s.cp.asnumpy(s.P)
    p_inf = float(P[np.unique(m.zones[ids["farfield"]].faces)].mean())
    z = m.zones[ids["wing"]]
    w = np.unique(z.faces[z.faces >= 0])
    X = m.nodes[w]
    eta = X[:, 2] / M.M6_SPAN
    cp_ = (P[w] - p_inf) / (0.5 * RHO * U ** 2)
    sections = {}
    for st in STATIONS:
        k = np.unique(np.round(eta, 9))
        e = k[np.argmin(np.abs(k - st))]                 # the nearest grid station
        sel = np.abs(eta - e) < 1e-7
        c = M.M6_ROOT * (1 - (1 - M.M6_TAPER) * e)
        xle = e * M.M6_SPAN * np.tan(M.M6_SWEEP)
        sections[f"{st:.2f}"] = np.stack([(X[sel, 0] - xle) / c, X[sel, 1] / c, cp_[sel],
                                          np.full(sel.sum(), e)], 1)
    hist = [h.get("turbulence", np.nan) for h in rep.history]
    out = {"re_mac": re, "alpha": alpha, "nodes": m.n_nodes, "iterations": rep.iterations,
           "converged": bool(rep.converged), "wall_s": rep.seconds,
           "CL": float(F @ lift_dir / q), "CD": float(F @ drag_dir / q), "S_ref": S_REF,
           "turbulence_linear_failures": tm.linear_failures,
           "flow_linear_converged": bool(rep.history[-1]["linear_converged"]),
           "mu_t_max_over_mu": float(tm.mu_t.max()) / mu}
    RES.mkdir(parents=True, exist_ok=True)
    np.savez(RES / "onera_m6.npz", **{f"cp_{k}": v for k, v in sections.items()},
             history=np.array(hist), du=np.array([h["du"] for h in rep.history]),
             rms_u=np.array([h["rms_u"] for h in rep.history]),
             lin_ok=np.array([h["linear_converged"] for h in rep.history]))
    (RES / "onera_m6.json").write_text(json.dumps(out, indent=1))
    log(json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=1.0e6)
    ap.add_argument("--alpha", type=float, default=3.06)
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--false-dt", type=float, default=0.5)
    ap.add_argument("--linear", default="amgx")
    ap.add_argument("--preset", default="robust")
    args = ap.parse_args()
    run(args.re, args.alpha, false_dt=args.false_dt, iterations=args.iterations,
        linear=args.linear, preset=args.preset)
