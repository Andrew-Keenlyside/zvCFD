"""Fast regression versions of the finite-volume validation cases (``fv_cases.py``).

Each asserts the error level and the observed order that the full case
measured, with some margin, so a later change cannot quietly break an
earlier result. Thresholds are set from ``benchmarks/results/validation/fv``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "validation"))
import fv_cases as C  # noqa: E402
from exact import DFG_2D1, ethier_steinman, kovasznay, manufactured_noslip  # noqa: E402

from zvcfd.fv.reference import Fluid, ReferenceSolver  # noqa: E402
from zvcfd.mesh.generate import box, cylinder_channel  # noqa: E402


def _order(e0, e1):
    return float(np.log2(e0 / e1))


def test_noslip_manufactured_hex_and_delaunay():
    for kind in ("hex", "delaunay"):
        errs = []
        for n in (4, 8):
            m = C._mesh(kind, n)

            def ex(x):
                return manufactured_noslip(x, stokes=True)

            u, p, _ = ex(m.nodes)
            node = int(np.argmin(np.linalg.norm(m.nodes - 0.5, axis=1)))
            s = ReferenceSolver(m, Fluid(), {z: {"kind": "wall"} for z in m.zones},
                                source=lambda x: ex(x)[2], stokes=True,
                                reference_pressure=(node, float(p[node])))
            s.solve(max_iterations=3)
            errs.append(C._l2(s.V, s.U, u))
        assert _order(*errs) > 1.6, kind


@pytest.mark.slow
def test_outlet_manufactured_tet_newtonian_and_carreau():
    from exact import manufactured_gn

    from zvcfd.rheology import CarreauYasuda

    cy = CarreauYasuda(mu_0=0.5, mu_inf=0.05, lam=2.0)
    for gn in (False, True):
        mu2 = cy.of_gamma_squared if gn else (lambda g2: 0 * g2 + 0.1)
        errs = []
        for n in (4, 8):
            m = box((n, n, n), kind="tet", warp=0.03)

            def ex(x):
                return manufactured_gn(x, mu2, shift=(0.2, 0.1, 0.3), mean=(1.5, 0.0, 0.0))

            z = {b.name: k for k, b in m.zones.items()}
            bcs = {k: {"kind": "velocity", "value": lambda x: ex(x)[0]} for k in m.zones}
            bcs[z["xmax"]] = {"kind": "pressure", "value": lambda x: ex(x)[1],
                              "grad": lambda x: ex(x)[3]}
            fl = Fluid(1.0, 0.1, viscosity=(lambda g: cy.of_gamma_squared(g * g)) if gn
                       else None)
            s = ReferenceSolver(m, fl, bcs, source=lambda x: ex(x)[2], transpose=True)
            rep = s.solve(max_iterations=80, tol=1e-11)
            assert rep.converged and abs(s.balances()["mass"]) < 1e-12
            errs.append(C._l2(s.V, s.U, ex(m.nodes)[0]))
        assert _order(*errs) > 1.5 and errs[1] < 0.02, gn


def test_kovasznay_second_order():
    errs = []
    for n in (8, 16):
        m = box((int(1.5 * n), 2 * n, 1), (1.5, 2.0, 1.0 / n), origin=(-0.5, -0.5, 0.0),
                kind="hex")
        z = {b.name: k for k, b in m.zones.items()}
        bcs = {k: {"kind": "velocity", "value": lambda x: kovasznay(x)[0]} for k in m.zones}
        bcs[z["zmin"]] = bcs[z["zmax"]] = {"kind": "symmetry"}
        u, p = kovasznay(m.nodes)
        node = int(np.argmin(np.linalg.norm(m.nodes[:, :2] - [0.25, 0.5], axis=1)))
        s = ReferenceSolver(m, Fluid(1.0, 1.0 / 40.0), bcs,
                            reference_pressure=(node, float(p[node])))
        s.solve(max_iterations=80, tol=1e-11)
        errs.append(C._l2(s.V, s.U, u))
    assert errs[1] < 2e-3 and _order(*errs) > 1.7


def test_womersley_second_order_in_space():
    errs = []
    for nc in (2, 4):
        m, s, u, rep = C._womersley(nc, 1 / 50, 50, "bdf2", stokes=True)
        errs.append(C._l2(s.V, s.U, u))
        b = s.balances()
        assert abs(b["mass"]) < 1e-13 and np.abs(b["momentum"]).max() < 1e-12
    assert _order(*errs) > 1.8


def test_womersley_time_orders():
    _, ref, _, _ = C._womersley(2, 1 / 400, 400, "bdf2", stokes=True)
    for scheme, lo in (("bdf1", 0.9), ("bdf2", 1.8)):
        e = []
        for k in (25, 50):
            _, s, _, _ = C._womersley(2, 1 / k, k, scheme, stokes=True)
            e.append(C._l2(s.V, s.U, ref.U))
        assert _order(*e) > lo, scheme


@pytest.mark.slow
def test_ethier_steinman_transient_navier_stokes():
    T = 0.02
    errs = []
    for n in (4, 8):
        m, s, rep = C._ethier("hex", n, T / 10, 10, "bdf2")
        errs.append(C._l2(s.V, s.U, ethier_steinman(m.nodes, T)[0]))
    assert _order(*errs) > 1.7 and errs[1] < 1e-2


def test_cavity_re100_against_ghia():
    gu = np.loadtxt(C.DATA / "ghia1982_u.txt")
    n = 16
    m = box((n, n, 1), (1.0, 1.0, 1.0 / n), kind="hex")
    z = {b.name: k for k, b in m.zones.items()}
    bcs = {k: {"kind": "wall"} for k in m.zones}
    bcs[z["ymax"]] = {"kind": "velocity", "value": (1.0, 0.0, 0.0)}
    bcs[z["zmin"]] = bcs[z["zmax"]] = {"kind": "symmetry"}
    s = ReferenceSolver(m, Fluid(1.0, 0.01), bcs)
    assert s.solve(max_iterations=150, tol=1e-10).converged
    x = m.nodes
    vl = np.isclose(x[:, 2], 0.0) & np.isclose(x[:, 0], 0.5)
    o = np.argsort(x[vl, 1])
    dev = np.interp(gu[1:-1, 0], x[vl, 1][o], s.U[vl, 0][o]) - gu[1:-1, 1]
    assert np.sqrt(np.mean(dev ** 2)) < 0.01


@pytest.mark.slow
def test_dfg_2d1_coarse():
    msh = cylinder_channel(8)
    z = {b.name: k for k, b in msh.zones.items()}
    H, UMAX = 0.41, 0.3

    def inflow(x):
        return np.stack([4 * UMAX * x[:, 1] * (H - x[:, 1]) / H ** 2, 0 * x[:, 0], 0 * x[:, 0]],
                        1)

    bcs = {z["inlet"]: {"kind": "velocity", "value": inflow},
           z["outlet"]: {"kind": "pressure", "value": 0.0}, z["walls"]: {"kind": "wall"},
           z["cylinder"]: {"kind": "wall"}, z["front"]: {"kind": "symmetry"},
           z["back"]: {"kind": "symmetry"}}
    s = ReferenceSolver(msh, Fluid(1.0, 1e-3), bcs)
    assert s.solve(max_iterations=100, tol=1e-10).converged
    F = s.zone_forces()[z["cylinder"]] / msh.nodes[:, 2].max()
    cd = 2 * F[0] / (0.2 ** 2 * 0.1)
    assert abs(cd / DFG_2D1["cd"] - 1) < 0.05          # +4.1 % at m = 8, +0.23 % at m = 32
