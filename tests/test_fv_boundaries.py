"""Boundary conditions of the finite-volume solvers beyond fixed values (Phases 3 and 4).

Each is checked against an exact statement rather than a tolerance on a
flow field:

- a flow-rate inlet carries exactly ``ρ Q``; the developed profile of a
  circular face is the parabola;
- an implicitly coupled lumped outlet reaches ``p = a + r Q`` exactly, and its
  flow field is the fixed-pressure solution at that pressure, with one
  outlet or two with resistances a thousand times the domain's (where a
  lagged coupling diverges);
- in time, the outlet pressure is the standalone 0-D model driven by the
  3-D outflow;
- backflow stabilisation, variable time steps: the GPU solver equals the
  CPU reference; BDF2 on a variable step is second order;
- average static pressure has the prescribed area mean; an opening takes
  total pressure where flow enters;
- wall shear from consistent reactions matches Poiseuille's ``4 μ Q / (π R³)``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from zvcfd.fv.linear import amgx_library
from zvcfd.fv.reference import Fluid, ReferenceSolver, bdf_coefficients
from zvcfd.lumped import RCR
from zvcfd.mesh.core import BoundaryZone
from zvcfd.mesh.generate import box, tube

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "validation"))

gpu = pytest.mark.gpu
LINEAR = ["host-direct", "acm"] + (["amgx"] if amgx_library() else [])
R = 0.5


def _pipe(n=2, axial=8, layers=1, growth=1.0):
    m = tube(R, 2.0, n_core=n, n_ring=n, n_axial=axial, kind="mixed", layers=layers,
             growth=growth)
    return m, {z.name: k for k, z in m.zones.items()}


def _gpu(m, bcs, fluid=Fluid(1.0, 0.2), linear="host-direct", **kw):
    from zvcfd.fv.solver import GPUSolver

    return GPUSolver(m, fluid, bcs, linear=linear,
                     linear_rtol=0.0 if linear == "host-direct" else 0.1, **kw)


# ---------------------------------------------------------------- inlets

def test_poiseuille_profile_of_a_circle():
    from zvcfd.fv.profiles import poiseuille

    m, z = _pipe(n=6)
    u = poiseuille(m.nodes, m.zones[z["inlet"]].faces)
    nodes = np.array(list(u))
    r2 = (m.nodes[nodes, :2] ** 2).sum(1) / R ** 2
    exact = 1 - r2
    np.testing.assert_allclose(np.array(list(u.values())), exact, atol=0.03)


def test_flow_rate_inlet_is_exact():
    m, z = _pipe()
    Q = 0.3
    bcs = {z["inlet"]: {"kind": "velocity", "flow_rate": [(0.0, Q), (1.0, 2 * Q), (2.0, Q)],
                        "profile": "poiseuille"},
           z["outlet"]: {"kind": "pressure", "value": 0.0}, z["wall"]: {"kind": "wall"}}
    s = ReferenceSolver(m, Fluid(1.3, 0.2), bcs, advection=1.0)
    for t, q in ((0.0, Q), (0.5, 1.5 * Q), (1.0, 2 * Q)):
        s._update_boundary(t)
        assert abs(s.sub_flow[z["inlet"]].sum() + 1.3 * q) < 1e-13 * q


# ---------------------------------------------------------------- lumped outlets

def _lumped_pipe(model, Q=0.3):
    m, z = _pipe()
    return m, z, {z["inlet"]: {"kind": "velocity", "flow_rate": Q, "profile": "poiseuille"},
                  z["outlet"]: {"kind": "pressure", "lumped": model},
                  z["wall"]: {"kind": "wall"}}


@gpu
@pytest.mark.parametrize("linear", LINEAR)
def test_steady_rcr_outlet(linear):
    model = RCR(rp=2.0, c=0.1, rd=5.0, pv=1.0)
    m, z, bcs = _lumped_pipe(model)
    g = _gpu(m, bcs, linear=linear, advection=1.0)
    assert g.solve(max_iterations=300, tol=1e-11).converged
    q = g.patch_flows()[z["outlet"]]
    assert abs(q - 0.3) < 1e-9
    P = g.fields()["P"]
    f = m.zones[z["outlet"]].faces
    out = np.unique(f[f >= 0])
    np.testing.assert_allclose(P[out], 1.0 + 7.0 * q, atol=1e-9)
    # the same field as a fixed-pressure outlet at that pressure
    fixed = dict(bcs)
    fixed[z["outlet"]] = {"kind": "pressure", "value": 1.0 + 7.0 * q}
    ref = ReferenceSolver(m, Fluid(1.0, 0.2), fixed, advection=1.0)
    ref.solve(max_iterations=100, tol=1e-13)
    np.testing.assert_allclose(g.fields()["U"], ref.U, atol=1e-8)
    np.testing.assert_allclose(P, ref.P, atol=1e-8 * np.ptp(ref.P))
    # the model was put at rest at the converged flow
    assert abs(model.pressure() - (1.0 + 7.0 * q)) < 1e-9


@gpu
def test_transient_rcr_follows_the_0d_model():
    def Q(t):
        return 0.3 + 0.1 * np.sin(2 * np.pi * t)

    model = RCR(rp=2.0, c=0.1, rd=5.0, pv=1.0)
    m, z, bcs = _lumped_pipe(model)
    bcs[z["inlet"]]["flow_rate"] = Q
    g = _gpu(m, bcs, advection=1.0)
    g.solve(max_iterations=300, tol=1e-12)
    rec = []
    g.solve_transient(0.05, 12, loops=40, tol=1e-12,
                      callback=lambda s: rec.append((s.t, s.patch_flows()[z["outlet"]],
                                                     s.patch_pressures()[z["outlet"]])))
    alone = RCR(rp=2.0, c=0.1, rd=5.0, pv=1.0)
    alone.initialise(Q(0.0), 0.0)
    for t, q, p in rec:
        assert abs(q - Q(t)) < 1e-9                 # rigid, incompressible: out = in
        assert abs(alone.advance(q, t, 0.05) - p) < 1e-9


def _two_outlet_box():
    """A channel with two separate outlets on its x = 1 face, split by a wall strip."""
    m = box((10, 10, 2), (1.0, 1.0, 0.2), kind="hex")
    z = {b.name: k for k, b in m.zones.items()}
    xmax = m.zones.pop(z["xmax"])
    y = m.nodes[:, 1][np.where(xmax.faces >= 0, xmax.faces, 0)]
    lo = (y <= 0.4 + 1e-12).all(1)
    hi = (y >= 0.6 - 1e-12).all(1)
    mid = ~(lo | hi)
    nid = max(m.zones) + 1
    for k, (name, sel) in enumerate((("out_a", lo), ("out_b", hi), ("strip", mid))):
        m.zones[nid + k] = BoundaryZone(nid + k, "wall" if name == "strip" else
                                        "pressure-outlet", name, xmax.faces[sel], xmax.cells[sel])
    return m, {b.name: k for k, b in m.zones.items()}


@gpu
@pytest.mark.parametrize("linear", ["acm"] + (["amgx"] if amgx_library() else []))
def test_two_stiff_lumped_outlets(linear):
    from zvcfd.fv.solver import GPUSolver

    m, z = _two_outlet_box()
    walls = {k: {"kind": "wall"} for n, k in z.items() if n not in ("xmin", "out_a", "out_b")}
    inlet = {z["xmin"]: {"kind": "velocity", "flow_rate": 0.02, "profile": "poiseuille"}}
    fl = Fluid(1.0, 0.05)
    # the domain's own resistance, from fixed equal pressures
    g0 = GPUSolver(m, fl, inlet | walls | {z["out_a"]: {"kind": "pressure", "value": 0.0},
                                           z["out_b"]: {"kind": "pressure", "value": 0.0}},
                   linear=linear, advection=1.0)
    g0.solve(max_iterations=200, tol=1e-10)
    rdom = g0.patch_pressures()[z["xmin"]] / 0.02
    ra, rb = 1e3 * rdom, 2e3 * rdom
    ma, mb = RCR(rp=0.1 * ra, c=1.0, rd=0.9 * ra), RCR(rp=0.1 * rb, c=1.0, rd=0.9 * rb)
    bcs = inlet | walls | {z["out_a"]: {"kind": "pressure", "lumped": ma},
                           z["out_b"]: {"kind": "pressure", "lumped": mb}}
    g = GPUSolver(m, fl, bcs, linear=linear, advection=1.0)
    tol = 1e-8                                        # dp's floor: r ≫ 1 amplifies round-off
    rep = g.solve(max_iterations=200, tol=tol)
    assert rep.converged and rep.iterations < 80
    q, p = g.patch_flows(), g.patch_pressures()
    for name, r in (("out_a", ra), ("out_b", rb)):
        # holds to the outer tolerance (measured 1.1e-8 of p with AmgX, below with ACM)
        assert abs(p[z[name]] - r * q[z[name]]) < 3 * tol * abs(p[z[name]])
    assert abs(q[z["out_a"]] + q[z["out_b"]] - 0.02) < 1e-7 * 0.02
    assert abs(q[z["out_a"]] / q[z["out_b"]] - 2.0) < 2e-3        # resistances dominate


# ---------------------------------------------------------------- outlet options

def _backflow_box(beta):
    m = box((6, 4, 3), (1.5, 1.0, 0.6), kind="tet", warp=0.03)
    z = {b.name: k for k, b in m.zones.items()}
    bcs = {k: {"kind": "wall"} for k in m.zones}
    bcs[z["xmin"]] = {"kind": "velocity", "value": (1.0, 0.0, 0.0)}
    bcs[z["xmax"]] = {"kind": "pressure", "value": lambda x: 1.5 * np.cos(np.pi * x[:, 1]),
                      "backflow_stabilisation": beta}
    return m, z, bcs


@gpu
def test_backflow_stabilisation_matches_reference():
    m, z, bcs = _backflow_box(0.5)
    fl = Fluid(1.0, 0.05)
    ref = ReferenceSolver(m, fl, bcs, advection=1.0)
    ref.solve(max_iterations=200, tol=1e-12)
    f = m.zones[z["xmax"]].faces
    assert (ref._mb[np.unique(f[f >= 0])] < 0).any()                   # there is backflow
    g = _gpu(m, bcs, fluid=fl, advection=1.0)
    assert g.solve(max_iterations=400, tol=1e-11).converged
    np.testing.assert_allclose(g.fields()["U"], ref.U, atol=1e-8)
    plain = ReferenceSolver(m, fl, _backflow_box(0.0)[2], advection=1.0)
    plain.solve(max_iterations=200, tol=1e-12)
    assert np.abs(plain.U - ref.U).max() > 1e-4                         # and it acts
    assert np.abs(ref.balances()["identity"]).max() < 1e-10


@gpu
def test_average_static_pressure():
    m, z = _pipe()
    bcs = {z["inlet"]: {"kind": "velocity", "flow_rate": 0.3},
           z["outlet"]: {"kind": "pressure", "value": 2.0, "profile": "average"},
           z["wall"]: {"kind": "wall"}}
    g = _gpu(m, bcs, advection=1.0)
    assert g.solve(max_iterations=300, tol=1e-10).converged
    assert abs(g.patch_pressures()[z["outlet"]] - 2.0) < 1e-10
    fixed = dict(bcs)
    fixed[z["outlet"]] = {"kind": "pressure", "value": 2.0}
    h = _gpu(m, fixed, advection=1.0)
    h.solve(max_iterations=300, tol=1e-10)
    dp = h.patch_pressures()[z["inlet"]] - 2.0
    assert abs(g.patch_pressures()[z["inlet"]] - h.patch_pressures()[z["inlet"]]) < 0.05 * dp


@gpu
def test_opening_takes_total_pressure_on_inflow():
    m, z = _pipe()
    bcs = {z["inlet"]: {"kind": "pressure", "value": 1.0, "opening": True},
           z["outlet"]: {"kind": "pressure", "value": 0.0}, z["wall"]: {"kind": "wall"}}
    g = _gpu(m, bcs, advection=1.0)
    assert g.solve(max_iterations=300, tol=1e-11).converged
    f = g.fields()
    fc = m.zones[z["inlet"]].faces
    nodes = np.unique(fc[fc >= 0])
    np.testing.assert_allclose(f["P"][nodes], 1.0 - 0.5 * (f["U"][nodes] ** 2).sum(1),
                               atol=1e-9)
    q = g.patch_flows()
    assert q[z["inlet"]] < 0 and abs(sum(q.values())) < 1e-10 * abs(q[z["inlet"]])


# ---------------------------------------------------------------- time

def test_variable_bdf2_coefficients():
    # exact for quadratics on any step ratio
    for w in (0.5, 1.0, 1.7):
        h, hp = 0.1 * w, 0.1
        c0, c1, c2 = bdf_coefficients("bdf2", h, hp, 2)
        f = np.poly1d([3.0, -2.0, 0.7])
        t1, t0, tm = 1.0, 1.0 - h, 1.0 - h - hp
        assert abs((c0 * f(t1) - c1 * f(t0) - c2 * f(tm)) / h - f.deriv()(t1)) < 1e-10


def _womersley_case():
    w = 2 * np.pi
    nu = R * R * w / 16
    m = tube(R, 0.5, n_core=2, n_ring=2, n_axial=2, kind="hex")
    z = {b.name: k for k, b in m.zones.items()}
    bcs = {z["inlet"]: {"kind": "pressure", "value": lambda x, t: 0.5 * np.cos(w * t)
                        * np.ones(len(x))},
           z["outlet"]: {"kind": "pressure", "value": 0.0}, z["wall"]: {"kind": "wall"}}
    return m, bcs, nu


@gpu
def test_variable_steps_match_reference():
    m, bcs, nu = _womersley_case()
    steps = [0.05, 0.08, 0.04, 0.06, 0.03, 0.07]
    ref = ReferenceSolver(m, Fluid(1.0, nu), bcs, advection=1.0, lag_rhie_chow=True)
    ref.solve_transient(steps, loops=60, tol=1e-12)
    g = _gpu(m, bcs, fluid=Fluid(1.0, nu), advection=1.0)
    g.solve_transient(steps, loops=60, tol=1e-12)
    np.testing.assert_allclose(g.fields()["U"], ref.U, atol=1e-9)


def test_variable_step_bdf2_is_second_order():
    m, bcs, nu = _womersley_case()
    T = 0.6

    def run(n):
        base = np.where(np.arange(n) % 2 == 0, 1.25, 0.75) * T / n   # ratios 0.6, 1.67
        s = ReferenceSolver(m, Fluid(1.0, nu), bcs, advection=1.0)
        s.solve_transient(base, loops=30, tol=1e-13)
        return s.U

    ref = run(384)
    e = [np.abs(run(n) - ref).max() for n in (24, 48, 96)]
    orders = np.log2(np.array(e[:-1]) / np.array(e[1:]))
    assert (orders > 1.85).all(), orders


# ---------------------------------------------------------------- wall shear

@gpu
def test_wall_shear_poiseuille():
    """``wall_shear("reaction")``: consistent-reaction WSS closes the axial force balance
    of developed flow exactly; against Poiseuille the error is the polygonal section's
    (``A/πR²`` = 0.974 here: 5.6 %, and 1.5 % on the next mesh, docs/validation)."""
    from zvcfd.fv.profiles import _triangles, rim_nodes

    m, z = _pipe(n=4, axial=8, layers=2, growth=0.8)
    Q, mu = 0.2, 0.1
    bcs = {z["inlet"]: {"kind": "velocity", "flow_rate": Q, "profile": "poiseuille"},
           z["outlet"]: {"kind": "pressure", "value": 0.0}, z["wall"]: {"kind": "wall"}}
    g = _gpu(m, bcs, fluid=Fluid(1.0, mu), stokes=True)
    g.solve(max_iterations=50, tol=1e-12)
    nodes, tau = g.wall_shear("reaction")
    x = m.nodes
    mid = (x[nodes, 2] > 0.6) & (x[nodes, 2] < 1.4)
    # section area and perimeter, and the developed pressure gradient on the axis
    fc = m.zones[z["inlet"]].faces
    t = _triangles(fc)
    area = 0.5 * np.abs(np.cross(x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]])[:, 2]).sum()
    rim = rim_nodes(fc)
    ring = x[rim][np.argsort(np.arctan2(x[rim, 1], x[rim, 0]))]
    perimeter = np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1).sum()
    axis = np.flatnonzero((np.hypot(x[:, 0], x[:, 1]) < 1e-9) & (x[:, 2] > 0.5) & (x[:, 2] < 1.5))
    dpdz = np.polyfit(x[axis, 2], g.fields()["P"][axis], 1)[0]
    assert abs(tau[mid, 2].mean() / (-dpdz * area / perimeter) - 1) < 5e-4
    assert tau[mid, 2].std() < 5e-3 * tau[mid, 2].mean()
    exact = 4 * mu * Q / (np.pi * R ** 3)
    assert abs(tau[mid, 2].mean() / exact - 1) < 0.06
    assert np.abs(tau[mid, :2]).max() < 0.03 * exact          # nodal-normal projection


@gpu
@pytest.mark.parametrize("method", ["gradient", "reaction"])
def test_wall_shear_at_rims_ignores_the_pressure_level(method):
    """At nodes on the wall and an inlet, the inlet's pressure force (tangential to the wall)
    is not wall shear: rim values stay near the developed value at any pressure level."""
    m, z = _pipe(n=4, axial=8, layers=2, growth=0.8)
    Q, mu = 0.2, 0.1
    taus = []
    for level in (0.0, 1e4):
        bcs = {z["inlet"]: {"kind": "velocity", "flow_rate": Q, "profile": "poiseuille"},
               z["outlet"]: {"kind": "pressure", "value": level}, z["wall"]: {"kind": "wall"}}
        g = _gpu(m, bcs, fluid=Fluid(1.0, mu), stokes=True)
        g.solve(max_iterations=50, tol=1e-12)
        nodes, tau = g.wall_shear(method)
        taus.append(tau)
    x = m.nodes[nodes]
    rim = x[:, 2] < 1e-9
    mid = (x[:, 2] > 0.6) & (x[:, 2] < 1.4)
    np.testing.assert_allclose(taus[1], taus[0], atol=1e-9 * 1e4)       # level-independent
    ref = taus[0][mid, 2].mean()
    assert np.abs(taus[0][rim, 2]).max() < 3 * ref
