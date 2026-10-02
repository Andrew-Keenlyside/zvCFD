"""Wall shear stress from velocity gradients at the wall integration points (CFX's method).

The shape functions reproduce linear fields, so for any affine velocity the
wall-ip gradient, and the wall shear stress, are exact on every element type,
distorted or not. Against Poiseuille flow the error is the near-wall
discretisation's: small with prism layers, first order on tetrahedra.
"""

import numpy as np
import pytest

from zvcfd.fv.wall import WallGradientShear
from zvcfd.mesh.generate import box, tube

gpu = pytest.mark.gpu


def _wall(m, name="ymin"):
    return [z for z, zz in m.zones.items() if zz.name == name]


@pytest.mark.parametrize("kind", ["tet", "wedge", "hex", "pyramid"])
@pytest.mark.parametrize("distort", ["perturb", "warp"])
def test_affine_field_is_exact(kind, distort):
    kw = {"perturb": 0.25} if distort == "perturb" else {"warp": 0.04}
    m = box((4, 3, 3), (1.0, 0.6, 0.7), kind=kind, seed=3, **kw)
    A = np.array([[0.3, 2.0, -0.4], [0.1, -0.5, 0.7], [-1.1, 0.8, 0.2]])
    U = m.nodes @ A.T + np.array([0.2, -0.1, 0.3])
    mu = 0.004
    op = WallGradientShear(m, _wall(m))
    tau = op(U, mu=mu)
    n = np.array([0.0, -1.0, 0.0])                       # ymin: outward normal
    t = mu * (A + A.T) @ n
    exact = -(t - (t @ n) * n)
    f = m.zones[_wall(m)[0]].faces
    nodes = np.unique(f[f >= 0])                         # quads on wedge and hex walls
    np.testing.assert_allclose(tau[nodes], np.broadcast_to(exact, (len(nodes), 3)),
                               atol=1e-12 * np.abs(exact).max(), rtol=0)
    off = np.setdiff1d(np.arange(m.n_nodes), nodes)
    assert np.abs(tau[off]).max() == 0.0


def test_shear_thinning_viscosity_at_the_ip_rate():
    from zvcfd.rheology import CarreauYasuda

    m = box((3, 3, 3), kind="tet", perturb=0.2, seed=1)
    cy = CarreauYasuda()
    gamma = 37.0
    U = np.zeros_like(m.nodes)
    U[:, 0] = gamma * m.nodes[:, 1]
    tau = WallGradientShear(m, _wall(m))(U, viscosity=cy)
    f = m.zones[_wall(m)[0]].faces
    nodes = np.unique(f[f >= 0])
    np.testing.assert_allclose(tau[nodes, 0], float(cy(np.array([gamma]))[0]) * gamma, rtol=1e-12)


def _poiseuille(kind, growth, R=0.5, Q=0.2, mu=0.1):
    """Developed Stokes flow in a 16-sided pipe: mean wall shear (gradient method) over the
    exact 4μQ/(πR³) and over the section's own force balance ``−(dp/dz) A / P``."""
    from zvcfd.fv.profiles import _triangles, rim_nodes
    from zvcfd.fv.reference import Fluid
    from zvcfd.fv.solver import GPUSolver

    m = tube(R, 2.0, n_core=4, n_ring=4, n_axial=8, kind=kind, layers=2, growth=growth)
    z = {zz.name: k for k, zz in m.zones.items()}
    bcs = {z["inlet"]: {"kind": "velocity", "flow_rate": Q, "profile": "poiseuille"},
           z["outlet"]: {"kind": "pressure", "value": 0.0}, z["wall"]: {"kind": "wall"}}
    g = GPUSolver(m, Fluid(1.0, mu), bcs, linear="host-direct", linear_rtol=0.0, stokes=True)
    g.solve(max_iterations=50, tol=1e-12)
    nodes, tau = g.wall_shear()
    x = m.nodes
    mid = (x[nodes, 2] > 0.6) & (x[nodes, 2] < 1.4)
    exact = 4 * mu * Q / (np.pi * R ** 3)
    assert np.abs(tau[mid, :2]).max() < 0.03 * exact          # along the axis
    fc = m.zones[z["inlet"]].faces
    t = _triangles(fc)
    area = 0.5 * np.abs(np.cross(x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]])[:, 2]).sum()
    rim = rim_nodes(fc)
    ring = x[rim][np.argsort(np.arctan2(x[rim, 1], x[rim, 0]))]
    perimeter = np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1).sum()
    axis = np.flatnonzero((np.hypot(x[:, 0], x[:, 1]) < 1e-9) & (x[:, 2] > 0.5) & (x[:, 2] < 1.5))
    dpdz = np.polyfit(x[axis, 2], g.fields()["P"][axis], 1)[0]
    balance = -dpdz * area / perimeter
    return tau[mid, 2].mean() / exact, tau[mid, 2].mean() / balance


@gpu
def test_poiseuille_prism_layers_converge():
    """On prism layers the wall gradient is the secant slope across the first layer, low by
    about h_n / 2R; refining the layers (``growth`` < 1 shrinks the wall cell in
    :func:`zvcfd.mesh.generate.tube`) brings it to the section's force balance."""
    errs = [abs(_poiseuille("mixed", g)[1] - 1) for g in (1.0, 0.625, 0.4)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 0.015      # 7.4, 3.2, 1.2 %


@gpu
def test_poiseuille_tetrahedra():
    """On tetrahedra the wall gradient is the element's constant (P1) one: first order."""
    _, coarse = _poiseuille("tet", 1.0)
    _, fine = _poiseuille("tet", 0.5)
    assert coarse < fine < 1.02 and 1 - coarse < 0.08
