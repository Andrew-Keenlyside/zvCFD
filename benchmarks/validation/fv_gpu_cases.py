"""Validation cases for the GPU finite-volume solver that are too long for the test suite.

    ZVCFD_AMGX_LIB=... PYTHONPATH=. python benchmarks/validation/fv_gpu_cases.py dfg2d2 [m ...]
    ZVCFD_AMGX_LIB=... PYTHONPATH=. python benchmarks/validation/fv_gpu_cases.py developed_pipe \
        [mixed|wedge|delaunay] [nc ...]

``dfg2d2``: the DFG cylinder benchmark 2D-2 (Schäfer & Turek 1996), Re = 100:
periodic vortex shedding in the quasi-2-D channel (one cell deep, symmetry
planes front and back), High Resolution advection, BDF2. The run starts
from the inflow profile with a small asymmetric perturbation in the wake,
marches to 6 s (shedding is established by about 3 s), then measures,
over the last lift periods, the maximum drag and lift coefficients, the
Strouhal number (from the lift period) and the pressure difference half a
period after maximum lift, against the benchmark's intervals
(``exact.DFG_2D2``). Results:
``benchmarks/results/validation/fv/dfg2d2.json``.

``developed_pipe``: pressure-driven pipe flow with fixed pressure at both
ends and natural velocity there (radius 0.5, length 4, Re = 50 on the
developed mean velocity). Developed Poiseuille flow solves the Navier–Stokes
equations and both boundary conditions, so the flow rate is the same at
every Reynolds number: the discrete Stokes solution's, converging to
Poiseuille's on the polygonal cross-section. Two mesh families: ``mixed``
(a tetrahedral core inside wedge layers graded toward the wall, the
OpenFOAM comparison's mesh) and ``wedge`` (triangles extruded along the
axis, rings graded toward the wall), and ``delaunay`` (the Delaunay
tetrahedralisation of a jittered lattice in the same prism: unstructured
connectivity, as in vessel meshes). Recorded at each level: Stokes, High
Resolution, a specified blend of 1, upwind and the opt-in second-order
boundary reconstruction, their flow rates against Stokes, the centreline
velocity at the inlet against mid-pipe (an inflow jet shows there), outer
iterations, and the observed orders. Results:
``benchmarks/results/validation/fv/developed_pipe.json``,
``developed_pipe_wedge.json`` and ``developed_pipe_delaunay.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact  # noqa: E402
from exact import DFG_2D2  # noqa: E402

from zvcfd.fv.linear import amgx_library  # noqa: E402
from zvcfd.fv.reference import Fluid  # noqa: E402
from zvcfd.fv.solver import GPUSolver  # noqa: E402
from zvcfd.mesh.generate import cylinder_channel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "results" / "validation" / "fv"
H, UMAX, NU, D = 0.41, 1.5, 1e-3, 0.1
UMEAN = 2 * UMAX / 3


def _crossings(t, y):
    """Times where ``y`` crosses zero upwards (linear interpolation)."""
    i = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0))
    return t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i])


def dfg2d2(m: int = 16, dt: float = 0.005, t_end: float = 6.0, loops: int = 6,
           log=print) -> dict:
    msh = cylinder_channel(m)
    z = {b.name: k for k, b in msh.zones.items()}

    def inflow(x):
        return np.stack([4 * UMAX * x[:, 1] * (H - x[:, 1]) / H ** 2, 0 * x[:, 0], 0 * x[:, 0]], 1)

    bcs = {z["inlet"]: {"kind": "velocity", "value": inflow},
           z["outlet"]: {"kind": "pressure", "value": 0.0}, z["walls"]: {"kind": "wall"},
           z["cylinder"]: {"kind": "wall"}, z["front"]: {"kind": "symmetry"},
           z["back"]: {"kind": "symmetry"}}
    linear = "amgx" if amgx_library() else "acm"
    s = GPUSolver(msh, Fluid(1.0, NU), bcs, linear=linear, freeze_limiter=None)
    x = msh.nodes

    def start(p):                          # inflow profile, a small asymmetric kick behind
        u = inflow(p)
        u[:, 1] = 0.05 * UMAX * np.exp(-((p[:, 0] - 0.35) ** 2 + (p[:, 1] - 0.22) ** 2) / 0.003)
        return u

    s.initialise(U=start)
    dz = x[:, 2].max()
    ia = np.argmin(np.hypot(x[:, 0] - 0.15, x[:, 1] - 0.2) + 10 * x[:, 2])
    ie = np.argmin(np.hypot(x[:, 0] - 0.25, x[:, 1] - 0.2) + 10 * x[:, 2])
    rec = []

    def after(solver):
        F = solver.zone_forces()[z["cylinder"]] / dz
        P = solver.P
        rec.append((solver.t, 2 * F[0] / (UMEAN ** 2 * D), 2 * F[1] / (UMEAN ** 2 * D),
                    float(P[int(ia)] - P[int(ie)])))
        if len(rec) % 100 == 0:
            log(f"  t {solver.t:.3f}  cd {rec[-1][1]:.4f}  cl {rec[-1][2]:+.4f}")

    t0 = time.time()
    rep = s.solve_transient(dt, int(round(t_end / dt)), loops=loops, tol=1e-7, callback=after)
    t, cd, cl, dp = (np.array(c) for c in zip(*rec))
    # the last full lift periods
    up = _crossings(t, cl)
    periods = np.diff(up)
    last = t >= up[-4] if len(up) >= 4 else t >= t[len(t) // 2]
    f = 1.0 / periods[-3:].mean()
    k = np.flatnonzero(last)[np.argmax(cl[last])]
    t_half = t[k] + 0.5 / f
    out = {"m": m, "nodes": msh.n_nodes, "dt": dt, "steps": len(t), "loops": loops,
           "linear": linear, "seconds": time.time() - t0, "coefficient_loops": rep.iterations,
           "cd_max": float(cd[last].max()), "cl_max": float(cl[last].max()),
           "cd_mean": float(cd[last].mean()),
           "st": float(f * D / UMEAN), "dp": float(np.interp(t_half, t, dp)),
           "period_spread": float(np.ptp(periods[-3:]) / periods[-3:].mean()),
           "interval": DFG_2D2["interval"]}
    out["inside"] = {k: bool(lo <= out[k] <= hi) for k, (lo, hi) in DFG_2D2["interval"].items()}
    out["history"] = {"t": t[::5].tolist(), "cd": cd[::5].tolist(), "cl": cl[::5].tolist()}
    return out


class _Diverged(Exception):
    pass


def delaunay_tube(nc: int, R: float = 0.5, L: float = 4.0, *, seed: int = 0,
                  jitter: float = 0.3):
    """A tetrahedral tube with unstructured connectivity: the Delaunay tetrahedralisation of
    a jittered lattice inside the ``4 nc``-sided prism of :func:`zvcfd.mesh.generate.tube`
    (``8 nc`` layers, wall polygon nodes on every layer, no wall layers)."""
    from scipy.spatial import Delaunay

    from zvcfd.mesh import UnstructuredMesh
    from zvcfd.mesh.generate import _orient, _zones_from_faces

    rng = np.random.default_rng(seed)
    N, nz = 4 * nc, 8 * nc
    h = 2 * np.pi * R / N
    ang = 2 * np.pi * np.arange(N) / N
    poly = R * np.c_[np.cos(ang), np.sin(ang)]
    apo = R * np.cos(np.pi / N)

    def inside(p, margin):
        th = np.arctan2(p[:, 1], p[:, 0]) % (2 * np.pi)
        mid = (np.floor(th / (2 * np.pi / N)) + 0.5) * 2 * np.pi / N
        return np.hypot(p[:, 0], p[:, 1]) * np.cos(th - mid) < apo - margin

    g = np.arange(-R, R + h, h)
    X, Y = np.meshgrid(g, g * np.sqrt(3) / 2)
    X = X + (np.arange(X.shape[0])[:, None] % 2) * h / 2
    sec = np.c_[X.ravel(), Y.ravel()]
    sec = sec[inside(sec, 0.5 * h)]
    pts = []
    for k, z in enumerate(np.linspace(0, L, nz + 1)):
        s = sec + rng.uniform(-jitter, jitter, sec.shape) * h
        s = s[inside(s, 0.25 * h)]
        zz = np.full(len(s), z)
        if 0 < k < nz:
            zz = zz + rng.uniform(-jitter, jitter, len(s)) * (L / nz)
        pts += [np.c_[s, zz], np.c_[poly, np.full(N, z)]]
    P = np.concatenate(pts)
    tet = Delaunay(P, qhull_options="Qbb Qc Qz Q12").simplices.astype(np.int64)
    x = P[tet]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]),
                           x[:, 3] - x[:, 0])) / 6
    m = UnstructuredMesh(P, {"tet": _orient(P, "tet", tet[vol > 1e-12 * h ** 3])})

    def rule(c, nrm):
        which = np.full(len(c), 2)
        which[np.abs(c[:, 2]) < 1e-9] = 0
        which[np.abs(c[:, 2] - L) < 1e-9] = 1
        return {"which": which, "names": [("inlet", "velocity-inlet"),
                                           ("outlet", "pressure-outlet"), ("wall", "wall")]}

    m.zones = _zones_from_faces(m, rule)
    return m


def developed_pipe(nc: int, kind: str = "mixed") -> dict:
    """One level of the developed-pipe case (see the module notes)."""
    from zvcfd.mesh.generate import tube

    R, L, MU, RHO, DP = 0.5, 4.0, 0.02, 1.0, 2.56
    if kind == "delaunay":
        m = delaunay_tube(nc, R, L)
    else:
        extra = {"layers": max(1, nc // 2)} if kind == "mixed" else {}
        m = tube(R, L, n_core=nc, n_ring=nc, n_axial=8 * nc, kind=kind, growth=0.85, **extra)
    zid = {z.name: k for k, z in m.zones.items()}
    bcs = {zid["inlet"]: {"kind": "pressure", "value": DP},
           zid["outlet"]: {"kind": "pressure", "value": 0.0}, zid["wall"]: {"kind": "wall"}}
    axis = np.hypot(m.nodes[:, 0], m.nodes[:, 1]) < 1e-9
    z = m.nodes[:, 2]
    q_circle = np.pi * R ** 4 * DP / (8 * MU * L)
    out = {"kind": kind, "nc": nc, "nodes": m.n_nodes, "elements": m.counts(),
           "wall_polygon_area_ratio": float((4 * nc / (2 * np.pi)) * np.sin(2 * np.pi / (4 * nc))),
           "q_exact_polygon": exact.polygon_poiseuille_flux(4 * nc, R, DP / (MU * L))}
    for tag, kw in (("stokes", {"stokes": True}), ("high_resolution", {}),
                    ("blend_1", {"advection": 1.0}), ("upwind", {"advection": "upwind"}),
                    ("reconstruction", {})):
        t = time.time()
        s = GPUSolver(m, Fluid(RHO, MU), bcs, linear="amgx" if amgx_library() else "acm", **kw)
        # the opt-in second-order boundary reconstruction: unstable on coarse meshes,
        # so stopped once it has clearly diverged (each solve then runs to its limit)
        s.boundary_reconstruction = tag == "reconstruction"
        diverged = []

        def watch(msg, diverged=diverged):
            it, du = int(msg.split()[1]), float(msg.split("du ")[1].split()[0])
            if it > 30 and (du > 0.5 or not np.isfinite(du)):
                diverged.append(it)
                raise _Diverged

        try:
            rep = s.solve(max_iterations=400 if tag == "reconstruction" else 800, tol=1e-9,
                          log=watch if tag == "reconstruction" else None)
        except _Diverged:
            out[tag] = {"q": float("nan"), "q_over_poiseuille_circle": float("nan"),
                        "centreline_in_mid_out": [None] * 3, "outer_iterations": diverged[0],
                        "converged": False, "diverged": True, "seconds": time.time() - t}
            if hasattr(s.linear, "close"):
                s.linear.close()
            continue
        U = s.fields()["U"]
        c = [float(U[axis & np.isclose(z, zk), 2].mean()) if axis.any() else None
             for zk in (0.0, L / 2, L)]                  # an odd nc has no node on the axis
        q = -s.zone_flows()[zid["inlet"]] / RHO
        out[tag] = {"q": q, "q_over_poiseuille_circle": q / q_circle,
                    "centreline_in_mid_out": c, "outer_iterations": rep.iterations,
                    "converged": bool(rep.converged), "seconds": time.time() - t}
        if hasattr(s.linear, "close"):
            s.linear.close()
    qs, qe = out["stokes"]["q"], out["q_exact_polygon"]
    out["stokes"]["q_over_exact"] = qs / qe
    for tag in ("high_resolution", "blend_1", "upwind", "reconstruction"):
        out[tag]["q_over_stokes_minus_1"] = out[tag]["q"] / qs - 1
        out[tag]["q_over_exact_minus_1"] = out[tag]["q"] / qe - 1
        c = out[tag]["centreline_in_mid_out"]
        out[tag]["inlet_jet"] = c[0] / c[1] - 1 if c[0] is not None and c[1] else None
    return out


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    name = argv[0] if argv else "dfg2d2"
    if name == "developed_pipe":
        kind = argv[1] if len(argv) > 1 and not argv[1].isdigit() else "mixed"
        levels = [int(a) for a in argv[1:] if a.isdigit()] or (
            {"mixed": [3, 6, 12, 18], "wedge": [6, 12, 18], "delaunay": [6, 12, 18, 24]}[kind])
        res = {"kind": kind, "levels": []}
        name = name if kind == "mixed" else f"{name}_{kind}"
        for nc in levels:
            r = developed_pipe(nc, kind)
            print(json.dumps(r, default=float), flush=True)
            res["levels"].append(r)
            if len(res["levels"]) >= 2:
                lv = res["levels"]
                for tag in ("high_resolution", "blend_1", "upwind"):
                    e = [abs(v[tag]["q_over_stokes_minus_1"]) for v in lv]
                    res[f"observed_order_{tag}"] = [
                        float(np.log(a / b) / np.log(vb["nc"] / va["nc"])) if b > 0 else None
                        for a, b, va, vb in zip(e, e[1:], lv, lv[1:])]
            (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
        return
    levels = [int(a) for a in argv[1:]] or [16]
    res = {"levels": []}
    for m in levels:
        r = dfg2d2(m)
        print({k: v for k, v in r.items() if k != "history"}, flush=True)
        res["levels"].append(r)
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
