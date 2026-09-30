"""CPU reference finite-volume solver: exact where the scheme must be exact, orders elsewhere."""

import sys
from pathlib import Path

import numpy as np
import pytest

from zvcfd.fv.reference import Fluid, ReferenceSolver
from zvcfd.mesh.generate import box, tube

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "validation"))
from exact import manufactured  # noqa: E402

KINDS = ("tet", "pyramid", "wedge", "hex")


def _interior(m):
    bnd = np.zeros(m.n_nodes, bool)
    for z in m.zones.values():
        bnd[z.faces[z.faces >= 0]] = True
    return ~bnd


@pytest.mark.parametrize("kind,perturb", [("tet", 0.25)] + [(k, 0.0) for k in KINDS])
def test_hydrostatic_is_exact(kind, perturb):
    """u = 0 and a linear pressure balancing a uniform body force, to round-off."""
    m = box((5, 5, 5), kind=kind, perturb=perturb)
    g = np.array([0.3, -1.1, 0.7])
    p = m.nodes @ g
    s = ReferenceSolver(m, Fluid(), {z: {"kind": "wall"} for z in m.zones}, stokes=True,
                        source=lambda x: np.broadcast_to(g, (len(x), 3)),
                        reference_pressure=(0, float(p[0])))
    s.solve(max_iterations=3)
    assert np.abs(s.U).max() < 1e-12
    assert np.abs(s.P - p).max() < 1e-11


@pytest.mark.parametrize("perturb", [0.0, 0.2])
@pytest.mark.parametrize("kind", KINDS)
def test_couette(kind, perturb):
    """A linear shear flow: exact on every affine element (all tetrahedra, undistorted others).

    Two planar flux points per ip face make linear fluxes exact on affine
    elements; on distorted non-affine elements a small consistency error
    remains.
    """
    m = box((4, 4, 4), kind=kind, perturb=perturb)
    def lin(x):
        return np.stack([0.5 + 2.0 * x[:, 2] - x[:, 1], 0 * x[:, 0], 0 * x[:, 0]], 1)

    s = ReferenceSolver(m, Fluid(1.0, 0.7), {z: {"kind": "velocity", "value": lin}
                                             for z in m.zones}, stokes=True,
                        reference_pressure=(0, 0.0))
    s.solve(max_iterations=3)
    err = np.abs(s.U - lin(m.nodes)).max() / 2.5
    affine = kind == "tet" or perturb == 0
    assert err < (1e-12 if affine else 5e-4)
    if affine:
        assert np.abs(s.P).max() < 1e-10


def test_stokes_manufactured_order():
    errs = []
    for n in (4, 8):
        m = box((n, n, n), kind="hex")
        def ex(x):
            return manufactured(x, stokes=True)

        u, p, _ = ex(m.nodes)
        node = int(np.argmin(np.linalg.norm(m.nodes - 0.5, axis=1)))
        s = ReferenceSolver(m, Fluid(), {z: {"kind": "velocity", "value": lambda x: ex(x)[0]}
                                         for z in m.zones}, source=lambda x: ex(x)[2],
                            stokes=True, reference_pressure=(node, float(p[node])))
        s.solve(max_iterations=3)
        errs.append(np.sqrt((s.V[:, None] * (s.U - u) ** 2).sum()
                            / (s.V[:, None] * u ** 2).sum()))
    assert errs[1] < 0.015
    assert np.log2(errs[0] / errs[1]) > 1.9


def test_pipe_mass_balance_and_order():
    """Poiseuille pipe: inflow = outflow to round-off; pressure gradient second order."""
    R, L = 0.5, 3.0
    out = []
    for nc in (2, 4):
        m = tube(R, L, n_core=nc, n_ring=nc, n_axial=6 * nc, kind="hex")
        zid = {z.name: k for k, z in m.zones.items()}
        def prof(x):
            r2 = (x[:, 0] ** 2 + x[:, 1] ** 2) / R ** 2
            return np.stack([0 * r2, 0 * r2, 2 * np.clip(1 - r2, 0, None)], 1)

        s = ReferenceSolver(m, Fluid(), {zid["inlet"]: {"kind": "velocity", "value": prof},
                                         zid["outlet"]: {"kind": "pressure", "value": 0.0},
                                         zid["wall"]: {"kind": "wall"}}, stokes=True)
        s.solve(max_iterations=30, tol=1e-10)
        q = s.zone_flows()
        assert abs(q[zid["outlet"]] + q[zid["inlet"]]) < 1e-12 * abs(q[zid["inlet"]])
        z = m.nodes[:, 2]
        p1 = s.P[np.isclose(z, 0.25 * L)].mean()
        p2 = s.P[np.isclose(z, 0.75 * L)].mean()
        out.append(abs((p1 - p2) / (0.5 * L) / (8 / R ** 2) - 1))
    assert out[1] < 0.04
    assert np.log2(out[0] / out[1]) > 1.8


@pytest.mark.parametrize("stokes", [True, False])
def test_symmetry_plane_reproduces_full_channel(stokes):
    """Half a channel with a symmetry plane = the full channel, to round-off (hex, mirrored).

    Also exercises pressure outlets with outflowing momentum (Navier–Stokes)
    and quasi-2-D symmetry planes on the z faces.
    """
    res = {}
    for H, ny in ((2.0, 8), (1.0, 4)):
        m = box((6, ny, 2), (3.0, H, 0.5), kind="hex")
        z = {b.name: k for k, b in m.zones.items()}

        def inflow(x):
            return np.stack([x[:, 1] * (2 - x[:, 1]), 0 * x[:, 0], 0 * x[:, 0]], 1)

        bcs = {z["xmin"]: {"kind": "velocity", "value": inflow},
               z["xmax"]: {"kind": "pressure", "value": 0.0}, z["ymin"]: {"kind": "wall"},
               z["zmin"]: {"kind": "symmetry"}, z["zmax"]: {"kind": "symmetry"},
               z["ymax"]: {"kind": "wall"} if H == 2 else {"kind": "symmetry"}}
        s = ReferenceSolver(m, Fluid(1.0, 0.5), bcs, stokes=stokes)
        s.solve(max_iterations=60, tol=1e-12)
        q = s.zone_flows()
        assert abs(sum(q.values())) < 1e-13 * abs(q[z["xmin"]])
        half = m.nodes[:, 1] <= 1 + 1e-9
        o = np.lexsort(np.round(m.nodes[half] * 1e6).astype(np.int64).T[::-1])
        res[H] = (s.U[half][o], s.P[half][o])
    np.testing.assert_allclose(res[1.0][0], res[2.0][0], atol=1e-12)
    np.testing.assert_allclose(res[1.0][1], res[2.0][1], atol=1e-12)


def test_high_resolution_picard_converges():
    """Navier–Stokes with the High Resolution limiter: Picard converges once the limiter freezes."""
    m = box((4, 4, 4), kind="tet")

    def ex(x):
        return manufactured(x, rho=1.0, mu=0.1)

    u, p, _ = ex(m.nodes)
    s = ReferenceSolver(m, Fluid(1.0, 0.1), {z: {"kind": "velocity", "value": lambda x: ex(x)[0]}
                                             for z in m.zones}, source=lambda x: ex(x)[2],
                        advection="high-resolution", reference_pressure=(31, float(p[31])))
    rep = s.solve(max_iterations=60, tol=1e-10)
    assert rep.converged and rep.iterations < 40


def _pipe_case():
    R = 0.5
    m = tube(R, 2.0, n_core=2, n_ring=2, n_axial=8, kind="mixed", layers=1)
    zid = {z.name: k for k, z in m.zones.items()}

    def prof(x):
        r2 = (x[:, 0] ** 2 + x[:, 1] ** 2) / R ** 2
        return np.stack([0 * r2, 0 * r2, 2 * np.clip(1 - r2, 0, None)], 1)

    return m, {zid["inlet"]: {"kind": "velocity", "value": prof},
               zid["outlet"]: {"kind": "pressure", "value": 0.0}, zid["wall"]: {"kind": "wall"}}


def test_steady_state_does_not_depend_on_time_step():
    """False time steps and time-marching reach the direct steady solution (Rhie–Chow, Choi)."""
    m, bcs = _pipe_case()
    ref = ReferenceSolver(m, Fluid(1.0, 0.2), bcs, advection=1.0)
    ref.solve(max_iterations=100, tol=1e-13)
    for dt in (0.05, 0.5):
        s = ReferenceSolver(m, Fluid(1.0, 0.2), bcs, dt=dt, advection=1.0)
        s.solve(max_iterations=400, tol=1e-13)
        np.testing.assert_allclose(s.P, ref.P, atol=1e-11 * np.ptp(ref.P))
    s = ReferenceSolver(m, Fluid(1.0, 0.2), bcs, advection=1.0)
    s.solve_transient(0.2, 80, loops=8, tol=1e-12)
    np.testing.assert_allclose(s.U, ref.U, atol=1e-10)
    np.testing.assert_allclose(s.P, ref.P, atol=1e-10 * np.ptp(ref.P))
    b = s.balances()
    assert abs(b["mass"]) < 1e-13 and np.abs(b["identity"]).max() < 1e-12


@pytest.mark.parametrize("stokes", [True, False])
def test_lagged_rhie_chow_reaches_the_implicit_solution(stokes):
    """Lagging ∇̄p (as CFX does, and the GPU solver will) converges to the same answer."""
    m, bcs = _pipe_case()
    ref = ReferenceSolver(m, Fluid(1.0, 0.2), bcs, advection=1.0, stokes=stokes)
    ref.solve(max_iterations=100, tol=1e-13)
    lag = ReferenceSolver(m, Fluid(1.0, 0.2), bcs, advection=1.0, stokes=stokes,
                          lag_rhie_chow=True)
    rep = lag.solve(max_iterations=400, tol=1e-12)
    assert rep.converged
    np.testing.assert_allclose(lag.U, ref.U, atol=1e-9)
    np.testing.assert_allclose(lag.P, ref.P, atol=1e-9 * np.ptp(ref.P))
