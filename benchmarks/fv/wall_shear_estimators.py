"""Wall shear stress estimators against exact solutions, with and without boundary layers.

    PYTHONPATH=. python benchmarks/fv/wall_shear_estimators.py pipe       # Poiseuille
    PYTHONPATH=. python benchmarks/fv/wall_shear_estimators.py hiemenz    # stagnation flow
    PYTHONPATH=. python benchmarks/fv/wall_shear_estimators.py all

Three estimators on the same steady Navier-Stokes solution:

- ``gradient``: ``GPUSolver.wall_shear()``, the default, as Ansys CFX evaluates
  laminar walls: shape-function velocity gradients at the wall integration
  points (:mod:`zvcfd.fv.wall`);
- ``reaction``: ``wall_shear("reaction")``, consistent reactions of the full
  momentum equation (what the zone forces use);
- ``stokes-op``: the same reactions with advection left out (the viscous and
  pressure operator applied to the Navier-Stokes state).

Cases (ρ = 1060 kg/m³, μ = 4 mPa s, blood-like):

- ``pipe``: developed flow, R = 1 mm, U = 0.2 m/s (Re 106), exact
  ``τ = 4 μ U / R``; on Delaunay tetrahedra (no layers) at R/h = 2.8–8.3,
  and on an O-grid tube with three prism layers whose wall cell shrinks
  (32-sided section, area deficit 0.6 %);
- ``hiemenz``: planar stagnation-point flow (a carina), a = 300 1/s,
  δ = √(ν/a) = 0.11 mm, exact ``τ_x = μ a x f''(0) / δ``; on Delaunay
  tetrahedra of size h, and on boundary-layer meshes (hexahedra, and the
  same cut into tetrahedra) 1.5 δ along the wall with the wall cell graded
  down to h_w.

Errors are relative L2 over the wall nodes away from the ends and edges, and
the bias of the mean. Needs a GPU and AmgX (``ZVCFD_AMGX_LIB``).
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from zvcfd.fv.reference import Fluid  # noqa: E402
from zvcfd.fv.solver import GPUSolver  # noqa: E402
from zvcfd.mesh.generate import box, delaunay_box, delaunay_tube, tube  # noqa: E402

RHO, MU = 1060.0, 0.004
NU = MU / RHO
METHODS = ("gradient", "reaction", "stokes-op")


def pure_wall(s):
    """Wall nodes that touch no other zone's face (no rim effects)."""
    other = np.zeros(s.N, bool)
    for zid, (f, S) in s.sub.items():
        if s.bcs[zid]["kind"] != "wall":
            other[f[f >= 0]] = True
    return ~other[s._wall_geometry()[0]]


def stokes_reaction(s):
    """Reaction wall shear from the viscous + pressure operator only, on the current state."""
    was = (s.stokes, s.asm.stokes)
    s.stokes, s.asm.stokes = True, 1
    s.assemble()
    _, tau = s.wall_shear("reaction")
    s.stokes, s.asm.stokes = was
    s.assemble()
    return tau


def estimators(mesh, bcs):
    s = GPUSolver(mesh, Fluid(RHO, MU), bcs, linear="amgx", linear_rtol=0.01,
                  linear_options={"precision": "double"})
    rep = s.solve(max_iterations=300, tol=1e-9)
    nodes, grad = s.wall_shear()
    _, react = s.wall_shear("reaction")
    return s, nodes, {"gradient": grad, "reaction": react, "stokes-op": stokes_reaction(s)}, \
        rep.converged


def stats(tau, exact, sel):
    e = tau[sel] - exact[sel]
    rel = np.sqrt((e ** 2).sum() / (exact[sel] ** 2).sum()) * 100
    bias = (np.linalg.norm(tau[sel], axis=1).mean() / np.linalg.norm(exact[sel], axis=1).mean()
            - 1) * 100
    return rel, bias


def row(label, nn, ests, exact, sel, conv):
    st = {m: stats(ests[m], exact, sel) for m in METHODS}
    cells = "  ".join(f"{st[m][0]:6.1f} / {st[m][1]:+6.1f}" for m in METHODS)
    print(f"{label:34s} {nn:7d} | {cells}  ({sel.sum()} nodes{'' if conv else ', NOT converged'})",
          flush=True)


def header(title):
    print(f"\n{title}\n{'mesh':34s} {'nodes':>7s} | " + "  ".join(f"{m + ' L2/bias %':>15s}"
                                                              for m in METHODS))


# ---------------------------------------------------------------- pipe

def _pipe_bcs(mesh, R, U):
    ids = {z.name: zid for zid, z in mesh.zones.items()}

    def vin(x, t=0.0):
        v = np.zeros((len(x), 3))
        v[:, 2] = 2 * U * np.maximum(1 - (x[:, 0] ** 2 + x[:, 1] ** 2) / R ** 2, 0)
        return v
    return {ids["wall"]: {"kind": "wall"}, ids["inlet"]: {"kind": "velocity", "value": vin},
            ids["outlet"]: {"kind": "pressure", "value": lambda x, t=0.0: np.zeros(len(x))}}


def pipe_case(mesh, R=1e-3, L=8e-3, U=0.2):
    s, nodes, ests, conv = estimators(mesh, _pipe_bcs(mesh, R, U))
    x = mesh.nodes[nodes]
    sel = pure_wall(s) & (x[:, 2] > 0.3 * L) & (x[:, 2] < 0.8 * L)
    exact = np.zeros((len(nodes), 3))
    exact[:, 2] = 4 * MU * U / R
    return mesh.n_nodes, ests, exact, sel, conv


def pipe():
    R, L = 1e-3, 8e-3
    header("Poiseuille, R = 1 mm, U = 0.2 m/s, Re 106; exact |tau| = 3.2 Pa")
    for h in (0.36e-3, 0.25e-3, 0.18e-3, 0.12e-3):
        m = delaunay_tube(R, L, h=h, jitter=0.3, seed=1)
        row(f"Delaunay tets, R/h = {R / h:.1f}", *pipe_case(m))
    for growth in (1.0, 0.625, 0.4):                    # < 1: thinner cells toward the wall
        m = tube(R, L, n_core=8, n_ring=5, n_axial=24, kind="mixed", layers=3, growth=growth)
        r = np.hypot(m.nodes[:, 0], m.nodes[:, 1])
        hw = R - np.sort(np.unique(np.round(r[r < R * (1 - 1e-9)], 12)))[-1]
        row(f"prism layers, wall cell R/{R / hw:.0f}", *pipe_case(m))


# ---------------------------------------------------------------- Hiemenz

def _hiemenz_profile():
    from scipy.integrate import solve_bvp
    eta = np.linspace(0, 10, 2001)
    y0 = np.vstack([eta - 1 + np.exp(-eta), 1 - np.exp(-eta), np.exp(-eta)])
    return solve_bvp(lambda e, y: np.vstack([y[1], y[2], -y[0] * y[2] - 1 + y[1] ** 2]),
                     lambda a, b: np.array([a[0], a[1], b[1] - 1]), eta, y0, tol=1e-10,
                     max_nodes=100000)


SOL = _hiemenz_profile()
FPP0 = float(SOL.sol(0.0)[2])
A_STRAIN = 300.0
DELTA = np.sqrt(NU / A_STRAIN)
XB, YB, ZB = 1.5e-3, 8 * DELTA, 0.3e-3


def hiemenz_case(mesh):
    a, delta = A_STRAIN, DELTA
    ids = {z.name: zid for zid, z in mesh.zones.items()}

    def vel(x, t=0.0):
        fv = SOL.sol(np.clip(x[:, 1] / delta, 0, 10))
        v = np.zeros((len(x), 3))
        v[:, 0] = a * x[:, 0] * fv[1]
        v[:, 1] = -np.sqrt(a * NU) * fv[0]
        return v
    bcs = {zid: ({"kind": "wall"} if name == "ymin" else {"kind": "velocity", "value": vel})
           for name, zid in ids.items()}
    s, nodes, ests, conv = estimators(mesh, bcs)
    x = mesh.nodes[nodes]
    exact = np.zeros((len(nodes), 3))
    exact[:, 0] = MU * a * x[:, 0] * FPP0 / delta
    sel = pure_wall(s) & (np.abs(x[:, 0]) < 0.8 * XB) & (np.abs(x[:, 0]) > 0.1 * XB)
    return mesh.n_nodes, ests, exact, sel, conv


def graded_box(kind, h_wall, h_along=1.5 * DELTA, ny=None):
    """A box graded geometrically in y from ``h_wall`` at the wall (a boundary-layer mesh)."""
    nx, nz = int(round(2 * XB / h_along)), max(3, int(round(ZB / h_along)))
    if ny is None:
        # geometric growth 1.2 from h_wall until cells reach h_along, then uniform
        sizes, h = [], h_wall
        while sum(sizes) < YB:
            sizes.append(min(h, h_along))
            h *= 1.2
        ny = len(sizes)
    m = box((nx, ny, nz), (2 * XB, YB, ZB), kind=kind, origin=(-XB, 0.0, 0.0))
    s = np.cumsum([0.0] + sizes)
    s *= YB / s[-1]
    eta = m.nodes[:, 1] / YB * ny
    k = np.clip(np.floor(eta).astype(int), 0, ny - 1)
    m.nodes[:, 1] = s[k] + (eta - k) * (s[k + 1] - s[k])
    return m


def hiemenz():
    header(f"Hiemenz stagnation flow, a = {A_STRAIN:g}/s, delta = {DELTA * 1e3:.3f} mm "
           f"(f''(0) = {FPP0:.5f})")
    for hd in (1.5, 1.0, 0.5, 0.25):
        h = hd * DELTA
        n = (int(round(2 * XB / h)), int(round(YB / h)), max(3, int(round(ZB / h))))
        m = delaunay_box(n, (2 * XB, YB, ZB), origin=(-XB, 0.0, 0.0), jitter=0.3, seed=1)
        row(f"Delaunay tets, h = {hd:g} delta", *hiemenz_case(m))
    for kind in ("hex", "tet"):
        for hw in (0.5, 0.2, 0.1):
            m = graded_box(kind, hw * DELTA)
            row(f"{kind} layers 1.5d along, wall {hw:g} d", *hiemenz_case(m))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("pipe", "all"):
        pipe()
    if which in ("hiemenz", "all"):
        hiemenz()
