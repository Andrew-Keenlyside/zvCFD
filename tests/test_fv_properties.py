"""Invariances the discrete equations must keep, whatever the mesh: they catch what orders miss.

- rotation, translation and reflection: the solution moves with the mesh;
- renumbering nodes and reordering elements: nothing changes;
- dynamic similarity: at fixed Reynolds number the dimensionless solution is fixed.

All on an unstructured (Delaunay) pipe with an inlet, a pressure outlet and walls.
"""

import numpy as np
import pytest

from zvcfd.fv.reference import Fluid, ReferenceSolver
from zvcfd.mesh.core import UnstructuredMesh
from zvcfd.mesh.generate import delaunay_tube

R, L = 0.5, 1.5


@pytest.fixture(scope="module")
def pipe():
    return delaunay_tube(R, L, h=0.16, seed=4)


def _solve(mesh, axis, centre, *, stokes, advection=1.0, rho=1.0, mu=0.05, U=1.0, scale=1.0):
    """Parabolic inflow along ``axis`` (a unit vector) through the zone named inlet."""
    axis = np.asarray(axis, float)
    centre = np.asarray(centre, float)

    def inflow(x):
        d = x - centre
        r2 = (d * d).sum(1) - (d @ axis) ** 2
        return 2 * U * np.clip(1 - r2 / (R * scale) ** 2, 0, None)[:, None] * axis

    zid = {z.name: k for k, z in mesh.zones.items()}
    s = ReferenceSolver(mesh, Fluid(rho, mu), {zid["inlet"]: {"kind": "velocity", "value": inflow},
                                               zid["outlet"]: {"kind": "pressure", "value": 0.0},
                                               zid["wall"]: {"kind": "wall"}},
                        stokes=stokes, advection=advection)
    s.solve(max_iterations=80, tol=1e-13)
    return s


def _random_orthogonal(seed, reflect):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if (np.linalg.det(q) < 0) != reflect:
        q[:, 0] *= -1
    return q


@pytest.mark.parametrize("stokes", [True, False])
@pytest.mark.parametrize("reflect", [False, True])
def test_rotation_and_reflection(pipe, stokes, reflect):
    base = _solve(pipe, (0, 0, 1), (0, 0, 0), stokes=stokes)
    Q = _random_orthogonal(7, reflect)
    t = np.array([0.3, -1.2, 2.5])
    moved = pipe.transformed(Q, t)
    s = _solve(moved, Q @ [0, 0, 1], t, stokes=stokes)
    scale = np.abs(base.U).max()
    np.testing.assert_allclose(s.U, base.U @ Q.T, atol=1e-10 * scale)
    np.testing.assert_allclose(s.P, base.P, atol=1e-10 * np.ptp(base.P))


def test_high_resolution_limiter_is_not_rotation_invariant(pipe):
    """The Barth–Jespersen limiter works on Cartesian components, as CFX's High
    Resolution does, so a rotated run differs slightly. Recorded, not required."""
    base = _solve(pipe, (0, 0, 1), (0, 0, 0), stokes=False, advection="high-resolution")
    Q = _random_orthogonal(7, False)
    s = _solve(pipe.transformed(Q), Q @ [0, 0, 1], (0, 0, 0), stokes=False,
               advection="high-resolution")
    diff = np.abs(s.U - base.U @ Q.T).max() / np.abs(base.U).max()
    assert diff < 1e-2


def _permute_elements(mesh: UnstructuredMesh, seed) -> UnstructuredMesh:
    rng = np.random.default_rng(seed)
    elements = {}
    for k, e in mesh.elements.items():
        e = e[rng.permutation(len(e))]
        # rotate each element's local numbering where the type allows it (tet: even permutation)
        if k == "tet":
            e = e[:, [1, 2, 0, 3]]
        elements[k] = e
    return UnstructuredMesh(mesh.nodes, elements, dict(mesh.zones), unit=mesh.unit)


@pytest.mark.parametrize("stokes", [True, False])
def test_renumbering(pipe, stokes):
    base = _solve(pipe, (0, 0, 1), (0, 0, 0), stokes=stokes)
    order = np.random.default_rng(11).permutation(pipe.n_nodes)
    re = _permute_elements(pipe.renumbered(order), 12)
    s = _solve(re, (0, 0, 1), (0, 0, 0), stokes=stokes)
    scale = np.abs(base.U).max()
    np.testing.assert_allclose(s.U, base.U[order], atol=1e-10 * scale)
    np.testing.assert_allclose(s.P, base.P[order], atol=1e-10 * np.ptp(base.P))


def test_dynamic_similarity(pipe):
    """Lengths × 3, speed × 0.2, viscosity × 0.6 (Re fixed): u/U and p/(ρU²) unchanged."""
    a, b = 3.0, 0.2
    base = _solve(pipe, (0, 0, 1), (0, 0, 0), stokes=False, mu=0.05, U=1.0)
    big = pipe.transformed(np.eye(3))
    big.nodes = pipe.nodes * a
    s = _solve(big, (0, 0, 1), (0, 0, 0), stokes=False, mu=0.05 * a * b, U=b, scale=a)
    np.testing.assert_allclose(s.U / b, base.U, atol=1e-10)
    np.testing.assert_allclose(s.P / b ** 2, base.P, atol=1e-10 * np.ptp(base.P))


def test_balances_close(pipe):
    s = _solve(pipe, (0, 0, 1), (0, 0, 0), stokes=False)
    b = s.balances()
    assert abs(b["mass"]) < 1e-13
    assert np.abs(b["identity"]).max() < 1e-12
    assert np.abs(b["momentum"]).max() < 1e-9
