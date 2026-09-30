"""Validation cases for the finite-volume reference solver (zvcfd.fv.reference).

    PYTHONPATH=. python benchmarks/validation/fv_cases.py [case ...]

Cases:

- ``exact``: hydrostatic and Couette solutions the scheme must reproduce to
  round-off;
- ``mms_stokes``, ``mms_ns``: the slip-wall manufactured solution under
  refinement, all element types;
- ``mms_noslip``: a manufactured solution with no-slip walls, including
  unstructured (Delaunay) tetrahedra;
- ``mms_outlet``: flow crossing a pressure outlet with prescribed traction,
  Newtonian and Carreau–Yasuda, on smoothly warped meshes;
- ``pipe``: Hagen–Poiseuille in a body-fitted tube (hex, tet, and the
  coronary mesh's mix of wall wedges and core tetrahedra);
- ``womersley``, ``ethier_steinman``: transient exact solutions, orders in
  space and in time (BDF1, BDF2);
- ``kovasznay``: exact steady Navier–Stokes flow, quasi-2-D;
- ``dfg``: DFG benchmark 2D-1, cylinder in a channel (Schäfer & Turek);
- ``cavity``: lid-driven cavity against Ghia, Ghia & Shin (1982);
- ``invariance``: rotation, reflection, renumbering, similarity, and the
  steady state's independence of the time step;
- ``gci_refine``: solution verification on a fixed mesh by uniform
  refinement and the grid convergence index.

Results go to
``benchmarks/results/validation/fv/<case>.json``. The exact solutions are
in ``exact.py``, independent of the solver.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exact import (  # noqa: E402
    DFG_2D1,
    ethier_steinman,
    kovasznay,
    manufactured,
    manufactured_gn,
    manufactured_noslip,
    womersley_pipe,
)

from zvcfd.fv.reference import Fluid, ReferenceSolver  # noqa: E402
from zvcfd.mesh.generate import (  # noqa: E402
    box,
    cylinder_channel,
    delaunay_box,
    delaunay_tube,
    tube,
)
from zvcfd.rheology import CarreauYasuda  # noqa: E402

DATA = Path(__file__).parent / "data"

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "results" / "validation" / "fv"
KINDS = ("tet", "pyramid", "wedge", "hex")


def _orders(errs):
    e = np.asarray(errs, float)
    return [None] + [float(v) for v in np.log2(e[:-1] / e[1:])]


def _interior(m):
    inner = np.ones(m.n_nodes, bool)
    for z in m.zones.values():
        inner[z.faces[z.faces >= 0]] = False
    return inner


def case_exact():
    rows = []
    g = np.array([0.3, -1.1, 0.7])
    def lin(x):
        return np.stack([0.5 + 2.0 * x[:, 2] - x[:, 1], 0 * x[:, 0], 0 * x[:, 0]], 1)

    for kind in KINDS:
        for perturb in (0.0, 0.2):
            m = box((5, 5, 5), kind=kind, perturb=perturb)
            p = m.nodes @ g
            s = ReferenceSolver(m, Fluid(), {z: {"kind": "wall"} for z in m.zones}, stokes=True,
                                source=lambda x: np.broadcast_to(g, (len(x), 3)),
                                reference_pressure=(0, float(p[0])))
            s.solve(max_iterations=3)
            hyd = (float(np.abs(s.U).max()), float(np.abs(s.P - p).max() / np.ptp(p)))
            s = ReferenceSolver(m, Fluid(1.0, 0.7), {z: {"kind": "velocity", "value": lin}
                                                     for z in m.zones}, stokes=True,
                                reference_pressure=(0, 0.0))
            s.solve(max_iterations=3)
            cou = float(np.abs(s.U - lin(m.nodes)).max() / 2.5)
            rows.append({"kind": kind, "perturb": perturb, "hydrostatic_u": hyd[0],
                         "hydrostatic_p": hyd[1], "couette_u": cou})
            print(rows[-1], flush=True)
    return {"rows": rows}


def _mms(kind, n, stokes, adv="high-resolution", mu=1.0, perturb=0.0):
    m = box((n, n, n), kind=kind, perturb=perturb)
    def ex(x):
        return manufactured(x, rho=1.0, mu=mu, stokes=stokes)

    u, p, _ = ex(m.nodes)
    node = int(np.argmin(np.linalg.norm(m.nodes - 0.5, axis=1)))
    t = time.time()
    s = ReferenceSolver(m, Fluid(1.0, mu), {z: {"kind": "velocity", "value": lambda x: ex(x)[0]}
                                            for z in m.zones}, source=lambda x: ex(x)[2],
                        stokes=stokes, advection=adv, reference_pressure=(node, float(p[node])))
    rep = s.solve(max_iterations=60, tol=1e-11)
    V = s.V
    eu = float(np.sqrt((V[:, None] * (s.U - u) ** 2).sum() / (V[:, None] * u ** 2).sum()))
    pe = s.P - p
    pe -= (V * pe).sum() / V.sum()
    pc = p - (V * p).sum() / V.sum()
    ep = float(np.sqrt((V * pe ** 2).sum() / (V * pc ** 2).sum()))
    inner = _interior(m)
    ep_in = float(np.sqrt((V[inner] * (pe[inner] - pe[inner].mean()) ** 2).sum()
                          / (V[inner] * pc[inner] ** 2).sum()))
    return {"n": n, "nodes": m.n_nodes, "eu": eu, "ep": ep, "ep_interior": ep_in,
            "iterations": rep.iterations, "converged": rep.converged,
            "seconds": time.time() - t}


def case_mms_stokes():
    out = {}
    for kind in KINDS:
        rows = [_mms(kind, n, True) for n in (4, 8, 16)]
        for key in ("eu", "ep", "ep_interior"):
            for r, o in zip(rows, _orders([r[key] for r in rows])):
                r[f"order_{key}"] = o
        out[kind] = rows
        print(kind, rows, flush=True)
    return out


def case_mms_ns():
    out = {}
    for kind, adv in (("hex", "high-resolution"), ("hex", "upwind"), ("tet", "high-resolution"),
                      ("wedge", "high-resolution")):
        rows = [_mms(kind, n, False, adv=adv, mu=0.1) for n in (4, 8, 16)]
        for key in ("eu", "ep", "ep_interior"):
            for r, o in zip(rows, _orders([r[key] for r in rows])):
                r[f"order_{key}"] = o
        out[f"{kind}-{adv}"] = rows
        print(kind, adv, rows, flush=True)
    return out


def _pipe(kind, nc, R=0.5, L=3.0):
    m = tube(R, L, n_core=nc, n_ring=nc, n_axial=6 * nc, kind=kind, layers=max(1, nc // 2))
    def prof(x):
        r2 = (x[:, 0] ** 2 + x[:, 1] ** 2) / R ** 2
        return np.stack([0 * r2, 0 * r2, 2 * np.clip(1 - r2, 0, None)], 1)

    zid = {z.name: k for k, z in m.zones.items()}
    t = time.time()
    s = ReferenceSolver(m, Fluid(), {zid["inlet"]: {"kind": "velocity", "value": prof},
                                     zid["outlet"]: {"kind": "pressure", "value": 0.0},
                                     zid["wall"]: {"kind": "wall"}}, stokes=True)
    rep = s.solve(max_iterations=40, tol=1e-10)
    q = s.zone_flows()
    x = m.nodes
    r2 = (x[:, 0] ** 2 + x[:, 1] ** 2) / R ** 2
    mid = np.isclose(x[:, 2], L / 2)
    eu = float(np.sqrt(((s.U[mid, 2] - 2 * (1 - r2[mid])) ** 2).mean()) / 2)
    G = 8.0 / R ** 2                                   # dp/dz for centreline 2 (mu = 1)
    p1, p2 = s.P[np.isclose(x[:, 2], 0.25 * L)].mean(), s.P[np.isclose(x[:, 2], 0.75 * L)].mean()
    p_in = s.P[np.isclose(x[:, 2], 0.0) & (r2 < 0.999)].mean()
    return {"nc": nc, "nodes": m.n_nodes, "elements": m.counts(), "u_mid_l2": eu,
            "dpdz_err": float((p1 - p2) / (0.5 * L) / G - 1),
            "inlet_p_err": float(p_in / (G * L) - 1),
            "mass_imbalance": float((q[zid["outlet"]] + q[zid["inlet"]]) / -q[zid["inlet"]]),
            "wall_polygon_area_ratio": float((4 * nc / (2 * np.pi)) * np.sin(2 * np.pi / (4 * nc))),
            "iterations": rep.iterations, "seconds": time.time() - t}


def case_pipe():
    out = {}
    for kind in ("hex", "tet", "mixed"):
        rows = [_pipe(kind, nc) for nc in (2, 4, 8)]
        for key in ("u_mid_l2", "dpdz_err", "inlet_p_err"):
            for r, o in zip(rows, _orders([abs(r[key]) for r in rows])):
                r[f"order_{key}"] = o
        out[kind] = rows
        print(kind, rows, flush=True)
    return out


def _l2(V, a, b, centre=False):
    """Relative L2 norm of ``a − b`` weighted by control volume (means removed if ``centre``)."""
    d, r = a - b, b
    if centre:
        w = V / V.sum()
        d = d - (w[:, None] * d).sum(0) if d.ndim == 2 else d - (w * d).sum()
        r = r - (w[:, None] * r).sum(0) if r.ndim == 2 else r - (w * r).sum()
    W = V[:, None] if d.ndim == 2 else V
    return float(np.sqrt((W * d ** 2).sum() / (W * r ** 2).sum()))


def _with_orders(rows, keys):
    for key in keys:
        for r, o in zip(rows, _orders([abs(r[key]) for r in rows])):
            r[f"order_{key}"] = o
    return rows


def _mesh(kind, n, **kw):
    return delaunay_box((n, n, n)) if kind == "delaunay" else box((n, n, n), kind=kind, **kw)


def case_mms_noslip():
    out = {}
    for kind in ("hex", "wedge", "tet", "pyramid", "delaunay"):
        rows = []
        for n in (4, 8, 16):
            m = _mesh(kind, n)

            def ex(x):
                return manufactured_noslip(x, stokes=True)

            u, p, _ = ex(m.nodes)
            node = int(np.argmin(np.linalg.norm(m.nodes - 0.5, axis=1)))
            t = time.time()
            s = ReferenceSolver(m, Fluid(), {z: {"kind": "wall"} for z in m.zones},
                                source=lambda x: ex(x)[2], stokes=True,
                                reference_pressure=(node, float(p[node])))
            s.solve(max_iterations=3)
            bnd = ~_interior(m)
            pe = s.P - p
            pe -= (s.V * pe).sum() / s.V.sum()
            rows.append({"n": n, "nodes": m.n_nodes, "eu": _l2(s.V, s.U, u),
                         "ep": _l2(s.V, s.P, p, centre=True),
                         "ep_boundary_max": float(np.abs(pe[bnd]).max() / np.ptp(p)),
                         "seconds": time.time() - t})
        out[kind] = _with_orders(rows, ("eu", "ep", "ep_boundary_max"))
        print(kind, out[kind], flush=True)
    return out


def case_mms_outlet():
    shift, mean = (0.2, 0.1, 0.3), (1.5, 0.0, 0.0)
    cy = CarreauYasuda(mu_0=0.5, mu_inf=0.05, lam=2.0)
    out = {}
    for kind in ("hex", "tet"):
        for fluid in ("newtonian", "carreau-yasuda"):
            gn = fluid != "newtonian"
            mu2 = cy.of_gamma_squared if gn else (lambda g2: 0 * g2 + 0.1)
            rows = []
            for n in (4, 8, 16):
                m = box((n, n, n), kind=kind, warp=0.03)

                def ex(x):
                    return manufactured_gn(x, mu2, shift=shift, mean=mean)

                z = {b.name: k for k, b in m.zones.items()}
                bcs = {k: {"kind": "velocity", "value": lambda x: ex(x)[0]} for k in m.zones}
                bcs[z["xmax"]] = {"kind": "pressure", "value": lambda x: ex(x)[1],
                                  "grad": lambda x: ex(x)[3]}
                fl = Fluid(1.0, 0.1, viscosity=(lambda g: cy.of_gamma_squared(g * g))
                           if gn else None)
                t = time.time()
                s = ReferenceSolver(m, fl, bcs, source=lambda x: ex(x)[2], transpose=True)
                rep = s.solve(max_iterations=80, tol=1e-11)
                u, p, _, _ = ex(m.nodes)
                rows.append({"n": n, "nodes": m.n_nodes, "eu": _l2(s.V, s.U, u),
                             "ep": _l2(s.V, s.P, p), "iterations": rep.iterations,
                             "converged": rep.converged, "mass": s.balances()["mass"],
                             "seconds": time.time() - t})
            out[f"{kind}-{fluid}"] = _with_orders(rows, ("eu", "ep"))
            print(kind, fluid, rows, flush=True)
    return out


def _womersley(nc, dt, steps, scheme, stokes, R=0.5, L=0.5, alpha=4.0):
    w = 2 * np.pi
    nu = R * R * w / alpha ** 2
    m = tube(R, L, n_core=nc, n_ring=nc, n_axial=2, kind="hex")
    zid = {z.name: k for k, z in m.zones.items()}
    bcs = {zid["inlet"]: {"kind": "pressure",
                          "value": lambda x, t: L * np.cos(w * t) * np.ones(len(x))},
           zid["outlet"]: {"kind": "pressure", "value": 0.0}, zid["wall"]: {"kind": "wall"}}
    s = ReferenceSolver(m, Fluid(1.0, nu), bcs, advection=1.0, stokes=stokes)
    r = np.hypot(m.nodes[:, 0], m.nodes[:, 1])

    def ex(t):
        return np.stack([0 * r, 0 * r, womersley_pipe(r, R, 1.0, w, nu, t)], 1)

    s.initialise(U=ex(0.0), P=lambda x: L - x[:, 2])
    rep = s.solve_transient(dt, steps, scheme=scheme, loops=6, tol=1e-12)
    return m, s, ex(dt * steps), rep


def case_womersley():
    out = {"space": [], "time": {}}
    for nc in (2, 4, 8):
        t = time.time()
        m, s, u, rep = _womersley(nc, 1 / 200, 200, "bdf2", stokes=False)
        b = s.balances()
        out["space"].append({"nc": nc, "nodes": m.n_nodes, "eu": _l2(s.V, s.U, u),
                             "loops_per_step": rep.iterations / 200, "mass": b["mass"],
                             "momentum": float(np.abs(b["momentum"]).max()),
                             "seconds": time.time() - t})
        print("womersley space", out["space"][-1], flush=True)
    _with_orders(out["space"], ("eu",))
    for scheme in ("bdf1", "bdf2"):
        _, ref, _, _ = _womersley(4, 1 / 800, 800, scheme, stokes=True)
        rows = []
        for k in (25, 50, 100):
            _, s, _, _ = _womersley(4, 1 / k, k, scheme, stokes=True)
            rows.append({"steps_per_period": k, "et": _l2(s.V, s.U, ref.U)})
        out["time"][scheme] = _with_orders(rows, ("et",))
        print("womersley time", scheme, rows, flush=True)
    return out


def _ethier(kind, n, dt, steps, scheme, nu=1.0):
    m = box((n, n, n), (2.0, 2.0, 2.0), origin=(-1, -1, -1), kind=kind)
    node = int(np.argmin(np.linalg.norm(m.nodes, axis=1)))

    def vel(x, t):
        return ethier_steinman(x, t, nu=nu)[0]

    def pin(x, t):
        return ethier_steinman(x, t, nu=nu)[1]

    s = ReferenceSolver(m, Fluid(1.0, nu), {z: {"kind": "velocity", "value": vel}
                                            for z in m.zones}, advection="high-resolution",
                        reference_pressure=(node, pin))
    s.initialise(U=lambda x: vel(x, 0.0), P=lambda x: pin(x, 0.0))
    rep = s.solve_transient(dt, steps, scheme=scheme, loops=6, tol=1e-11)
    return m, s, rep


def case_ethier_steinman():
    T = 0.1
    out = {"space": {}, "time": {}}
    for kind in ("hex", "tet"):
        rows = []
        for n in (4, 8, 16):
            t = time.time()
            m, s, rep = _ethier(kind, n, T / 40, 40, "bdf2")
            u, p = ethier_steinman(m.nodes, T)
            rows.append({"n": n, "nodes": m.n_nodes, "eu": _l2(s.V, s.U, u),
                         "ep": _l2(s.V, s.P, p, centre=True),
                         "loops_per_step": rep.iterations / 40, "seconds": time.time() - t})
            print("ethier space", kind, rows[-1], flush=True)
        out["space"][kind] = _with_orders(rows, ("eu", "ep"))
    for scheme in ("bdf1", "bdf2"):
        _, ref, _ = _ethier("hex", 8, T / 160, 160, scheme)
        rows = []
        for k in (5, 10, 20):
            _, s, _ = _ethier("hex", 8, T / k, k, scheme)
            rows.append({"steps": k, "et": _l2(s.V, s.U, ref.U)})
        out["time"][scheme] = _with_orders(rows, ("et",))
        print("ethier time", scheme, rows, flush=True)
    return out


def case_kovasznay(re=40.0):
    out = {}
    for kind in ("hex", "wedge", "tet"):
        rows = []
        for n in (8, 16, 32):
            m = box((int(1.5 * n), 2 * n, 1), (1.5, 2.0, 1.0 / n), origin=(-0.5, -0.5, 0.0),
                    kind=kind)
            z = {b.name: k for k, b in m.zones.items()}
            bcs = {k: {"kind": "velocity", "value": lambda x: kovasznay(x, re)[0]}
                   for k in m.zones}
            bcs[z["zmin"]] = bcs[z["zmax"]] = {"kind": "symmetry"}
            u, p = kovasznay(m.nodes, re)
            node = int(np.argmin(np.linalg.norm(m.nodes[:, :2] - [0.25, 0.5], axis=1)))
            t = time.time()
            s = ReferenceSolver(m, Fluid(1.0, 1.0 / re), bcs,
                                reference_pressure=(node, float(p[node])))
            rep = s.solve(max_iterations=80, tol=1e-11)
            rows.append({"n": n, "nodes": m.n_nodes, "eu": _l2(s.V, s.U, u),
                         "ep": _l2(s.V, s.P, p, centre=True), "iterations": rep.iterations,
                         "seconds": time.time() - t})
        out[kind] = _with_orders(rows, ("eu", "ep"))
        print("kovasznay", kind, rows, flush=True)
    return out


def case_dfg(levels=(8, 16, 32)):
    H, UMAX, NU, D = 0.41, 0.3, 1e-3, 0.1
    rows = []
    for mres in levels:
        msh = cylinder_channel(mres)
        z = {b.name: k for k, b in msh.zones.items()}

        def inflow(x):
            return np.stack([4 * UMAX * x[:, 1] * (H - x[:, 1]) / H ** 2, 0 * x[:, 0],
                             0 * x[:, 0]], 1)

        bcs = {z["inlet"]: {"kind": "velocity", "value": inflow},
               z["outlet"]: {"kind": "pressure", "value": 0.0}, z["walls"]: {"kind": "wall"},
               z["cylinder"]: {"kind": "wall"}, z["front"]: {"kind": "symmetry"},
               z["back"]: {"kind": "symmetry"}}
        t = time.time()
        s = ReferenceSolver(msh, Fluid(1.0, NU), bcs)
        rep = s.solve(max_iterations=100, tol=1e-10)
        dz = msh.nodes[:, 2].max()
        F = s.zone_forces()[z["cylinder"]] / dz
        um = 2 * UMAX / 3
        x = msh.nodes

        def p_at(px, py):
            return s.P[np.argmin(np.hypot(x[:, 0] - px, x[:, 1] - py) + 10 * x[:, 2])]

        r = {"m": mres, "nodes": msh.n_nodes, "cd": 2 * F[0] / (um ** 2 * D),
             "cl": 2 * F[1] / (um ** 2 * D), "dp": p_at(0.15, 0.2) - p_at(0.25, 0.2),
             "iterations": rep.iterations, "mass": s.balances()["mass"],
             "seconds": time.time() - t}
        r.update({f"{k}_err": (r[k] - DFG_2D1[k]) / DFG_2D1[k] for k in ("cd", "cl", "dp")})
        rows.append(r)
        print("dfg", r, flush=True)
    return {"rows": _with_orders(rows, ("cd_err", "dp_err")), "reference": DFG_2D1}


def case_cavity():
    gu = np.loadtxt(DATA / "ghia1982_u.txt")
    gv = np.loadtxt(DATA / "ghia1982_v.txt")
    out = {}
    for col, re in ((1, 100.0), (2, 400.0)):
        rows = []
        for n in (16, 32, 64):
            m = box((n, n, 1), (1.0, 1.0, 1.0 / n), kind="hex")
            z = {b.name: k for k, b in m.zones.items()}
            bcs = {k: {"kind": "wall"} for k in m.zones}
            bcs[z["ymax"]] = {"kind": "velocity", "value": (1.0, 0.0, 0.0)}
            bcs[z["zmin"]] = bcs[z["zmax"]] = {"kind": "symmetry"}
            t = time.time()
            s = ReferenceSolver(m, Fluid(1.0, 1.0 / re), bcs)
            rep = s.solve(max_iterations=150, tol=1e-10)
            x = m.nodes
            front = np.isclose(x[:, 2], 0.0)
            vl = front & np.isclose(x[:, 0], 0.5)
            hl = front & np.isclose(x[:, 1], 0.5)
            ov = np.argsort(x[vl, 1])
            oh = np.argsort(x[hl, 0])
            u_line = np.interp(gu[1:-1, 0], x[vl, 1][ov], s.U[vl, 0][ov])
            v_line = np.interp(gv[1:-1, 0], x[hl, 0][oh], s.U[hl, 1][oh])
            # Ghia's Re = 400 v at x = 0.9063 is probably a typo (see the data file): report
            # the deviations without it too
            ok = ~((re == 400.0) & np.isclose(gv[1:-1, 0], 0.9063))
            dv = v_line - gv[1:-1, col]
            rows.append({"n": n, "nodes": m.n_nodes, "iterations": rep.iterations,
                         "converged": rep.converged,
                         "u_rms_dev": float(np.sqrt(np.mean((u_line - gu[1:-1, col]) ** 2))),
                         "v_rms_dev": float(np.sqrt(np.mean(dv ** 2))),
                         "v_rms_dev_clean": float(np.sqrt(np.mean(dv[ok] ** 2))),
                         "u_max_dev": float(np.abs(u_line - gu[1:-1, col]).max()),
                         "v_max_dev": float(np.abs(dv).max()),
                         "v_max_dev_clean": float(np.abs(dv[ok]).max()),
                         "u_min": float(s.U[vl, 0].min()), "seconds": time.time() - t})
            print("cavity", re, rows[-1], flush=True)
        out[f"Re{int(re)}"] = _with_orders(rows, ("u_rms_dev", "v_rms_dev_clean"))
    return out


def _pipe_bcs(mesh, axis, centre, R, scale=1.0, U=1.0):
    axis, centre = np.asarray(axis, float), np.asarray(centre, float)

    def inflow(x):
        d = x - centre
        r2 = (d * d).sum(1) - (d @ axis) ** 2
        return 2 * U * np.clip(1 - r2 / (R * scale) ** 2, 0, None)[:, None] * axis

    zid = {z.name: k for k, z in mesh.zones.items()}
    return {zid["inlet"]: {"kind": "velocity", "value": inflow},
            zid["outlet"]: {"kind": "pressure", "value": 0.0}, zid["wall"]: {"kind": "wall"}}


def case_invariance():
    R, L = 0.5, 1.5
    pipe = delaunay_tube(R, L, h=0.16, seed=4)
    out = {"mesh": {"nodes": pipe.n_nodes, "elements": pipe.counts()}}

    def solve(mesh, axis, centre, adv, mu=0.05, U=1.0, scale=1.0, dt=None):
        s = ReferenceSolver(mesh, Fluid(1.0, mu), _pipe_bcs(mesh, axis, centre, R, scale, U),
                            advection=adv, dt=dt)
        s.solve(max_iterations=400, tol=1e-13)
        return s

    q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    rot = q if np.linalg.det(q) > 0 else q @ np.diag([-1, 1, 1])
    ref = rot @ np.diag([-1, 1, 1])
    t = np.array([0.3, -1.2, 2.5])
    for adv in (1.0, "high-resolution"):
        base = solve(pipe, (0, 0, 1), (0, 0, 0), adv)
        us, ps = np.abs(base.U).max(), np.ptp(base.P)
        for name, Q in (("rotation", rot), ("reflection", ref)):
            s = solve(pipe.transformed(Q, t), Q @ [0, 0, 1], t, adv)
            out[f"{name}-{adv}"] = {"du": float(np.abs(s.U - base.U @ Q.T).max() / us),
                                    "dp": float(np.abs(s.P - base.P).max() / ps)}
    base = solve(pipe, (0, 0, 1), (0, 0, 0), 1.0)
    order = np.random.default_rng(11).permutation(pipe.n_nodes)
    s = solve(pipe.renumbered(order), (0, 0, 1), (0, 0, 0), 1.0)
    out["renumbering"] = {"du": float(np.abs(s.U - base.U[order]).max()),
                          "dp": float(np.abs(s.P - base.P[order]).max() / np.ptp(base.P))}
    big = pipe.transformed(np.eye(3))
    big.nodes = pipe.nodes * 3.0
    s = solve(big, (0, 0, 1), (0, 0, 0), 1.0, mu=0.05 * 3.0 * 0.2, U=0.2, scale=3.0)
    out["similarity"] = {"du": float(np.abs(s.U / 0.2 - base.U).max()),
                         "dp": float(np.abs(s.P / 0.04 - base.P).max() / np.ptp(base.P))}
    for dt in (0.05, 0.5):
        s = solve(pipe, (0, 0, 1), (0, 0, 0), 1.0, dt=dt)
        out[f"false_dt_{dt}"] = {"du": float(np.abs(s.U - base.U).max()),
                                 "dp": float(np.abs(s.P - base.P).max() / np.ptp(base.P))}
    b = base.balances()
    out["balances"] = {"mass": b["mass"], "momentum": float(np.abs(b["momentum"]).max()),
                       "identity": float(np.abs(b["identity"]).max())}
    print(out, flush=True)
    return out


def case_gci_refine():
    """Solution verification on a mesh that cannot be remeshed: refine it, then GCI.

    A coarse wall-wedge / tet-core pipe stands in for the coronary mesh. It
    is refined twice by uniform conforming subdivision, which keeps its
    (faceted) geometry. So the grid study measures discretisation error on
    that geometry, exactly as a coronary study would.
    """
    from zvcfd.mesh.refine import refine
    from zvcfd.verification import grid_study

    R, L, DP, MU = 0.5, 3.0, 1.0, 1.0
    meshes = [tube(R, L, n_core=2, n_ring=2, n_axial=6, kind="mixed", layers=1)]
    for _ in range(2):
        meshes.append(refine(meshes[-1]))
    rows = []
    for m in meshes:
        zid = {z.name: k for k, z in m.zones.items()}
        bcs = {zid["inlet"]: {"kind": "pressure", "value": DP},
               zid["outlet"]: {"kind": "pressure", "value": 0.0}, zid["wall"]: {"kind": "wall"}}
        t = time.time()
        s = ReferenceSolver(m, Fluid(1.0, MU), bcs, stokes=True)
        s.solve(max_iterations=40, tol=1e-12)
        q = s.zone_flows()
        F = s.zone_forces()
        axis = int(np.argmin(np.linalg.norm(m.nodes - [0.0, 0.0, L / 2], axis=1)))
        area = sum(abs(np.einsum("fik->k", S)[2]) for zz, (f, S) in s.sub.items()
                   if m.zones[zz].name == "inlet")
        rows.append({"nodes": m.n_nodes, "elements": m.counts(), "q": -q[zid["inlet"]],
                     "u_centre": float(s.U[axis, 2]),
                     # Stokes flow: the wall force must equal dp x inlet area exactly
                     "wall_force_z": float(F[zid["wall"]][2]),
                     "dp_area": DP * area, "seconds": time.time() - t})
        print("gci", rows[-1], flush=True)
    study = {k: grid_study(rows[2][k], rows[1][k], rows[0][k], r=2.0).as_dict()
             for k in ("q", "u_centre")}
    return {"levels": rows, "study": study}


CASES = {"exact": case_exact, "mms_stokes": case_mms_stokes, "mms_ns": case_mms_ns,
         "mms_noslip": case_mms_noslip, "mms_outlet": case_mms_outlet, "pipe": case_pipe,
         "womersley": case_womersley, "ethier_steinman": case_ethier_steinman,
         "kovasznay": case_kovasznay, "dfg": case_dfg, "cavity": case_cavity,
         "invariance": case_invariance, "gci_refine": case_gci_refine}


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names or list(CASES):
        t = time.time()
        res = CASES[name]()
        res["_seconds"] = time.time() - t
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(f"{name}: {time.time() - t:.0f} s -> {OUT / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
