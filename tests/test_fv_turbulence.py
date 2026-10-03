"""The k-kL turbulence model (k-kL-MEAH2015m) on the finite-volume solver."""

import numpy as np
import pytest

from zvcfd.fv.turbulence import _point_triangle, tmr_freestream, wall_distance
from zvcfd.mesh.generate import box

gpu = pytest.mark.gpu


def test_point_triangle_distance_is_exact():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(200, 3, 3))
    p = rng.normal(size=(200, 3)) * 2
    d = _point_triangle(p, t)
    u = rng.random((40000, 2))
    m = u.sum(1) > 1
    u[m] = 1 - u[m]
    for i in range(200):
        q = t[i, 0] + u[:, :1] * (t[i, 1] - t[i, 0]) + u[:, 1:] * (t[i, 2] - t[i, 0])
        sampled = np.linalg.norm(q - p[i], axis=1).min()
        assert d[i] <= sampled + 1e-12 and sampled - d[i] < 0.03


def test_wall_distance_to_a_plane():
    m = box((6, 5, 4), (1.0, 0.5, 0.4), kind="tet", perturb=0.2, seed=2)
    wall = [z for z in m.zones.values() if z.name == "ymin"][0]
    d = wall_distance(m.nodes, m.nodes[wall.triangles()])
    np.testing.assert_allclose(d, m.nodes[:, 1], atol=1e-12)


def test_tmr_freestream():
    # NASA TMR: k = 9e-9 a^2, kL = 1.5589e-6 mu a / rho, a = U / M
    k, kl = tmr_freestream(1.2, 1.8e-5, 34.0, 0.1)
    assert k == pytest.approx(9e-9 * 340.0 ** 2)
    assert kl == pytest.approx(1.5589e-6 * 1.8e-5 * 340 / 1.2)


def test_config():
    from zvcfd.config import from_dict

    src = {"kind": "mesh", "path": "x.msh"}
    cfg = from_dict({"source": src, "fv": {"turbulence": {"model": "k-kl", "speed": 1.0,
                                                           "mach": 0.15}}})
    assert cfg.fv.turbulence["model"] == "k-kl"
    with pytest.raises(ValueError, match="model"):
        from_dict({"source": src, "fv": {"turbulence": {"model": "k-epsilon", "speed": 1,
                                                         "mach": 0.1}}})
    with pytest.raises(ValueError, match="k_inf and kl_inf, or speed and mach"):
        from_dict({"source": src, "fv": {"turbulence": {"model": "k-kl", "speed": 1.0}}})
    with pytest.raises(ValueError, match="solver.method fv"):
        from_dict({"source": src | {"voxel_size": 10.0},
                   "fv": {"turbulence": {"model": "k-kl", "k_inf": 1e-4, "kl_inf": 1e-8}}})


def _channel(turbulent: bool, Re: float = 2e4):
    from zvcfd.fv.reference import Fluid
    from zvcfd.fv.solver import GPUSolver
    from zvcfd.fv.turbulence import KkLModel

    H, L = 1.0, 10.0
    m = box((20, 48, 1), (L, 2 * H, 0.2), kind="hex")
    yy = m.nodes[:, 1] / (2 * H)
    m.nodes[:, 1] = H * (1 + np.tanh(2.5 * (2 * yy - 1)) / np.tanh(2.5))
    ids = {z.name: k for k, z in m.zones.items()}
    mu = H / Re
    vin = lambda x, t=0.0: np.tile([1.0, 0.0, 0.0], (len(x), 1))  # noqa: E731
    bcs = {ids["xmin"]: {"kind": "velocity", "value": vin},
           ids["xmax"]: {"kind": "pressure", "value": 0.0},
           ids["ymin"]: {"kind": "wall"}, ids["ymax"]: {"kind": "wall"},
           ids["zmin"]: {"kind": "symmetry"}, ids["zmax"]: {"kind": "symmetry"}}
    s = GPUSolver(m, Fluid(1.0, mu), bcs, linear="auto", linear_rtol=0.1, dt=2.0)
    s.initialise(U=np.tile([1.0, 0.0, 0.0], (m.n_nodes, 1)), P=np.zeros(m.n_nodes))
    tm = None
    if turbulent:
        k = 1e-3
        tm = KkLModel(s, k, 10 * mu * np.sqrt(k) / 0.09 ** 0.25).attach()
    rep = s.solve(max_iterations=300, tol=1e-6)
    nodes, tau = s.wall_shear()
    x = m.nodes[nodes]
    far = x[:, 0] > 0.8 * L
    return rep, tm, m, float(np.abs(tau[far, 0]).mean())


@gpu
def test_turbulent_channel():
    """k-kL converges; k, kL >= 0; mu_t = 0 at the walls and > 0 inside; wall shear well above
    the laminar flow's at the same Reynolds number."""
    rep, tm, m, tau_t = _channel(True)
    assert rep.converged
    k, phi, mut = (tm.cp.asnumpy(a) for a in (tm.k, tm.phi, tm.mu_t))
    wall = tm.cp.asnumpy(tm.wall_d)
    assert (k >= 0).all() and (phi >= 0).all() and (mut[wall] == 0).all()
    assert mut[~wall].max() > 5 * 1.0 / 2e4
    _, _, _, tau_l = _channel(False)
    assert tau_t > 1.5 * tau_l


@gpu
def test_scalar_matrix_is_an_m_matrix_on_stretched_hexahedra():
    """Edge-based diffusion and upwind advection keep every off-diagonal of the k and kL
    matrices non-positive on cells 50 times longer than thick (the shape-function diffusion
    gives positive ones there)."""
    from zvcfd.fv.reference import Fluid
    from zvcfd.fv.solver import GPUSolver
    from zvcfd.fv.turbulence import KkLModel

    m = box((12, 16, 1), (1.0, 0.05, 0.004), kind="hex")
    yy = m.nodes[:, 1] / 0.05
    m.nodes[:, 1] = 0.05 * yy ** 2
    ids = {z.name: k for k, z in m.zones.items()}
    vin = lambda x, t=0.0: np.tile([1.0, 0.0, 0.0], (len(x), 1))  # noqa: E731
    bcs = {ids["xmin"]: {"kind": "velocity", "value": vin},
           ids["xmax"]: {"kind": "pressure", "value": 0.0},
           ids["ymin"]: {"kind": "wall"}, ids["ymax"]: {"kind": "symmetry"},
           ids["zmin"]: {"kind": "symmetry"}, ids["zmax"]: {"kind": "symmetry"}}
    s = GPUSolver(m, Fluid(1.0, 1e-5), bcs, linear="host-direct", dt=1.0)
    s.initialise(U=np.tile([1.0, 0.0, 0.0], (m.n_nodes, 1)), P=np.zeros(m.n_nodes))
    s.solve(max_iterations=5, tol=0.0)
    tm = KkLModel(s, 1e-4, 1e-8)
    seen = []
    solve = tm._linear_solve

    def grab(data, b, x0):
        seen.append(tm.cp.asnumpy(data))
        return solve(data, b, x0)
    tm._linear_solve = grab
    tm.update(s)
    rows = tm.cp.asnumpy(tm._rows)
    cols = tm.cp.asnumpy(tm.indices)
    off = rows != cols
    for data in seen:
        assert data[off].max() <= 1e-12 * np.abs(data).max()
        assert (data[~off] > 0).all()
