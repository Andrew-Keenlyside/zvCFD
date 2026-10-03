"""NASA TMR 2-D NACA 0012 validation case with the k-kL model (k-kL-MEAH2015m).

    PYTHONPATH=. python benchmarks/turbulence/naca0012.py --grid 225 --alpha 0 10 15

Re = 6 million on the chord, the TMR farfield turbulence for M = 0.15, the
Resource's C-grids one cell deep (farfield about 500 chords). The flow's linear
systems are solved directly (SuperLU on the host), as for the flat plate; the
false time step grows from 0.05 to ``--false-dt`` (20: the farfield is 500 chords
away). The runs stop when the relative changes of u, p, k, Φ and μ_t fall below
``--tol`` (10⁻⁶; the turbulence's changes level off at a few 10⁻⁷). Lift and drag
come two ways: consistent reactions on the airfoil zone (``zone_forces``),
and the integral of surface pressure plus wall shear stress (the CFX-style
wall gradient). Surface Cp and Cf are saved for the comparison with Ladson
(1988) and Gregory & O'Reilly (1970): benchmarks/results/turbulence/
naca0012_<grid>_a<alpha>.npz.
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
U, RHO, RE, MACH = 1.0, 1.0, 6.0e6, 0.15
MU = RHO * U * 1.0 / RE


def run(ni: int, alpha: float, *, false_dt: float = 20.0, iterations: int = 1000,
        tol: float = 1e-6, linear: str = "host-direct", log=print) -> dict:
    x2, y2 = M.read_plot3d_2d(M.TMR / "NACA0012_grids" / M.NACA[ni])
    h1 = float(np.hypot(x2[1] - x2[0], y2[1] - y2[0])[(x2[0] >= 0) & (x2[0] <= 1)].min())
    span = 4 * h1
    m = M.naca0012(ni, span=span)
    ids = {z.name: k for k, z in m.zones.items()}
    a = np.radians(alpha)
    uinf = np.array([np.cos(a), np.sin(a), 0.0]) * U
    vin = lambda x, t=0.0: np.tile(uinf, (len(x), 1))  # noqa: E731
    bcs = {ids["farfield"]: {"kind": "velocity", "value": vin},
           ids["outflow"]: {"kind": "pressure", "value": 0.0},
           ids["airfoil"]: {"kind": "wall"}, ids["zmin"]: {"kind": "symmetry"},
           ids["zmax"]: {"kind": "symmetry"}}
    opts = {"linear_options": {"preset": "robust"}} if linear == "amgx" else {}
    s = GPUSolver(m, Fluid(RHO, MU), bcs, linear=linear, linear_rtol=0.1, dt=false_dt, **opts)
    s.initialise(U=np.tile(uinf, (m.n_nodes, 1)), P=np.zeros(m.n_nodes))
    tm = KkLModel(s, *tmr_freestream(RHO, MU, U, MACH)).attach()
    q = 0.5 * RHO * U ** 2 * 1.0 * span                     # dynamic pressure x chord x span
    lift_dir, drag_dir = np.array([-np.sin(a), np.cos(a), 0.0]), uinf / U
    forces = []                                             # (iteration, CL, CD) every 5
    update = tm.update

    def update_and_record(solver):
        change = update(solver)
        if len(tm.history) % 5 == 0:
            Fi = solver.zone_forces()[ids["airfoil"]]
            forces.append((len(tm.history), Fi @ lift_dir / q, Fi @ drag_dir / q))
        return change
    tm.update = update_and_record
    rep = solve_ramped(s, false_dt, max_iterations=iterations, tol=tol)
    wall_s = rep.seconds
    F = s.zone_forces()[ids["airfoil"]]
    # surface integral: pressure (relative to the farfield) and wall shear on the airfoil
    z = m.zones[ids["airfoil"]]
    f, S = s.sub[ids["airfoil"]]
    P = s.cp.asnumpy(s.P)
    p_inf = float(P[np.unique(m.zones[ids["farfield"]].faces)].mean())
    ok = f >= 0
    Fp = np.zeros(3)
    for k in range(3):
        Fp[k] = ((P[np.where(ok, f, 0)] - p_inf) * S[..., k] * ok).sum()
    nodes, tau = s.wall_shear()
    area = np.zeros(m.n_nodes)
    for kk in range(4):
        sel = ok[:, kk]
        np.add.at(area, f[sel, kk], np.linalg.norm(S[sel, kk], axis=1))
    on = np.isin(nodes, np.unique(z.faces[z.faces >= 0]))
    Fv = (tau[on] * area[nodes[on], None]).sum(0)
    Fs = Fp + Fv
    out = {"grid": ni, "alpha": alpha, "nodes": m.n_nodes, "span": span,
           "iterations": rep.iterations, "converged": bool(rep.converged), "wall_s": wall_s,
           "turbulence_linear_failures": tm.linear_failures,
           "flow_linear_converged": bool(rep.history[-1]["linear_converged"]),
           "CL": float(F @ lift_dir / q), "CD": float(F @ drag_dir / q),
           "CL_surface": float(Fs @ lift_dir / q), "CD_surface": float(Fs @ drag_dir / q),
           "CD_pressure": float(Fp @ drag_dir / q), "CD_viscous": float(Fv @ drag_dir / q)}
    last = np.array([f for f in forces if f[0] > len(tm.history) - 100])
    if len(last):                         # the band over the last 100 iterations
        out |= {"CL_range_last100": [float(last[:, 1].min()), float(last[:, 1].max())],
                "CD_range_last100": [float(last[:, 2].min()), float(last[:, 2].max())]}
    # surface distributions on the z = 0 plane
    w = np.unique(z.faces[z.faces >= 0])
    w = w[np.abs(m.nodes[w, 2]) < 1e-12]
    xs, ys = m.nodes[w, 0], m.nodes[w, 1]
    cp_ = (P[w] - p_inf) / (0.5 * RHO * U ** 2)
    tmap = dict(zip(nodes.tolist(), range(len(nodes))))
    tw = np.array([tau[tmap[n]] for n in w])
    # skin friction signed along the local surface tangent (x-ward on both sides)
    cf = np.sign(tw[:, 0]) * np.linalg.norm(tw, axis=1) / (0.5 * RHO * U ** 2)
    RES.mkdir(parents=True, exist_ok=True)
    np.savez(RES / f"naca0012_{ni}_a{alpha:g}.npz", x=xs, y=ys, cp=cp_, cf=cf,
             forces=np.array(forces),
             history=np.array([h.get("turbulence", np.nan) if isinstance(h, dict) else np.nan
                               for h in rep.history]), **{k: v for k, v in out.items()})
    (RES / f"naca0012_{ni}_a{alpha:g}.json").write_text(json.dumps(out, indent=1))
    save_fields(RES / f"fields_naca0012_{ni}_a{alpha:g}.npz", s, tm, MU)
    log(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs="+", default=[225])
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.0, 10.0, 15.0])
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--false-dt", type=float, default=20.0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--linear", default="host-direct")
    args = ap.parse_args()
    for g in args.grid:
        for al in args.alpha:
            run(g, al, iterations=args.iterations, false_dt=args.false_dt, tol=args.tol,
                linear=args.linear)
