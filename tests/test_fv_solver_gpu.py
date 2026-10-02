"""The GPU solver (zvcfd.fv.solver) reaches the CPU reference's solutions.

The GPU lags the Rhie–Chow gradient and solves each linear system only to a
relative tolerance, as CFX does. At convergence it must reproduce the
reference solver: to round-off with a fixed blend (any iteration path), and
with High Resolution when both freeze the limiter on the same path (the
reference in its lagged mode).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from zvcfd.fv.linear import amgx_library
from zvcfd.fv.reference import Fluid, ReferenceSolver
from zvcfd.mesh.generate import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "validation"))
from test_fv_reference import _pipe_case  # noqa: E402

pytestmark = pytest.mark.gpu
LINEAR = ["host-direct", "acm"] + (["amgx"] if amgx_library() else [])


@pytest.mark.parametrize("linear", LINEAR)
def test_pipe_matches_reference(linear):
    from zvcfd.fv.solver import GPUSolver

    m, bcs = _pipe_case()
    ref = ReferenceSolver(m, Fluid(1.0, 0.2), bcs, advection=1.0)
    ref.solve(max_iterations=200, tol=1e-13)
    g = GPUSolver(m, Fluid(1.0, 0.2), bcs, advection=1.0, linear=linear,
                  linear_rtol=0.0 if linear == "host-direct" else 0.1)
    rep = g.solve(max_iterations=600, tol=1e-11)
    assert rep.converged
    f = g.fields()
    np.testing.assert_allclose(f["U"], ref.U, atol=1e-9)
    np.testing.assert_allclose(f["P"], ref.P, atol=1e-9 * np.ptp(ref.P))
    assert abs(rep.history[-1]["imbalance"]) < 1e-12


def test_high_resolution_matches_lagged_reference():
    from zvcfd.fv.solver import GPUSolver

    m, bcs = _pipe_case()
    ref = ReferenceSolver(m, Fluid(1.0, 0.2), bcs, lag_rhie_chow=True)
    ref.solve(max_iterations=400, tol=1e-12)
    g = GPUSolver(m, Fluid(1.0, 0.2), bcs, linear="host-direct", linear_rtol=0.0)
    g.solve(max_iterations=400, tol=1e-12)
    np.testing.assert_allclose(g.fields()["U"], ref.U, atol=1e-8)


def test_symmetry_and_false_time_step():
    from zvcfd.fv.solver import GPUSolver

    m = box((6, 4, 2), (3.0, 1.0, 0.5), kind="hex")
    z = {b.name: k for k, b in m.zones.items()}

    def inflow(x):
        return np.stack([x[:, 1] * (2 - x[:, 1]), 0 * x[:, 0], 0 * x[:, 0]], 1)

    bcs = {z["xmin"]: {"kind": "velocity", "value": inflow},
           z["xmax"]: {"kind": "pressure", "value": 0.0}, z["ymin"]: {"kind": "wall"},
           z["ymax"]: {"kind": "symmetry"}, z["zmin"]: {"kind": "symmetry"},
           z["zmax"]: {"kind": "symmetry"}}
    ref = ReferenceSolver(m, Fluid(1.0, 0.5), bcs, advection=1.0)
    ref.solve(max_iterations=100, tol=1e-13)
    g = GPUSolver(m, Fluid(1.0, 0.5), bcs, advection=1.0, dt=0.2, linear="acm")
    assert g.solve(max_iterations=600, tol=1e-11).converged
    np.testing.assert_allclose(g.fields()["U"], ref.U, atol=1e-9)


def test_carreau_outlet_manufactured():
    """Generalised-Newtonian flow crossing a pressure outlet with prescribed traction."""
    from exact import manufactured_gn

    from zvcfd.fv.solver import GPUSolver
    from zvcfd.rheology import CarreauYasuda

    cy = CarreauYasuda(mu_0=0.5, mu_inf=0.05, lam=2.0)
    m = box((4, 4, 4), kind="tet", warp=0.03)

    def ex(x):
        return manufactured_gn(x, cy.of_gamma_squared, shift=(0.2, 0.1, 0.3),
                               mean=(1.5, 0.0, 0.0))

    z = {b.name: k for k, b in m.zones.items()}
    bcs = {k: {"kind": "velocity", "value": lambda x: ex(x)[0]} for k in m.zones}
    bcs[z["xmax"]] = {"kind": "pressure", "value": lambda x: ex(x)[1],
                      "grad": lambda x: ex(x)[3]}
    fl = Fluid(1.0, 0.1, viscosity=cy)
    ref = ReferenceSolver(m, Fluid(1.0, 0.1, viscosity=lambda g: cy.of_gamma_squared(g * g)),
                          bcs, source=lambda x: ex(x)[2], advection=1.0, transpose=True)
    ref.solve(max_iterations=100, tol=1e-12)
    g = GPUSolver(m, fl, bcs, source=lambda x: ex(x)[2], advection=1.0, transpose=True,
                  linear="host-direct", linear_rtol=0.0)
    assert g.solve(max_iterations=400, tol=1e-11).converged
    np.testing.assert_allclose(g.fields()["U"], ref.U, atol=1e-8)


def test_transient_womersley_matches_reference():
    from exact import womersley_pipe

    from zvcfd.fv.solver import GPUSolver
    from zvcfd.mesh.generate import tube

    R, L, w = 0.5, 0.5, 2 * np.pi
    nu = R * R * w / 16
    m = tube(R, L, n_core=2, n_ring=2, n_axial=2, kind="hex")
    zid = {z.name: k for k, z in m.zones.items()}
    bcs = {zid["inlet"]: {"kind": "pressure",
                          "value": lambda x, t: L * np.cos(w * t) * np.ones(len(x))},
           zid["outlet"]: {"kind": "pressure", "value": 0.0}, zid["wall"]: {"kind": "wall"}}
    r = np.hypot(m.nodes[:, 0], m.nodes[:, 1])
    U0 = np.stack([0 * r, 0 * r, womersley_pipe(r, R, 1.0, w, nu, 0.0)], 1)
    ref = ReferenceSolver(m, Fluid(1.0, nu), bcs, advection=1.0, lag_rhie_chow=True)
    ref.initialise(U=U0, P=lambda x: L - x[:, 2])
    ref.solve_transient(0.05, 10, loops=60, tol=1e-12)
    g = GPUSolver(m, Fluid(1.0, nu), bcs, advection=1.0, linear="host-direct",
                  linear_rtol=0.0)
    g.initialise(U=U0, P=lambda x: L - x[:, 2])
    g.solve_transient(0.05, 10, loops=60, tol=1e-12)
    np.testing.assert_allclose(g.fields()["U"], ref.U, atol=1e-9)


def test_zone_forces_match_reference():
    """Consistent-reaction forces: walls, inlet, outlet and symmetry, GPU = CPU reference."""
    from zvcfd.fv.solver import GPUSolver

    m, bcs = _pipe_case()
    ref = ReferenceSolver(m, Fluid(1.0, 0.2), bcs, advection=1.0)
    ref.solve(max_iterations=200, tol=1e-13)
    g = GPUSolver(m, Fluid(1.0, 0.2), bcs, advection=1.0, linear="host-direct", linear_rtol=0.0)
    g.solve(max_iterations=600, tol=1e-11)
    fr, fg = ref.zone_forces(), g.zone_forces()
    scale = max(np.abs(v).max() for v in fr.values())
    for z in fr:
        np.testing.assert_allclose(fg[z], fr[z], atol=1e-8 * scale)
