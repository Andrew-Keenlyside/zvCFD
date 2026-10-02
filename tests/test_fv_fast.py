"""The fast kernels (zvcfd.fv.fast) evaluate the same discrete equations as the CPU reference.

Random states on distorted meshes of every element type, with and without
stored geometry, a generalised-Newtonian fluid and a time term: nodal
gradients, node volumes, the momentum diagonal (static viscous part plus
the advective part the mass-flow pass returns), the block matrix and
right-hand side, the limiter and the mass flows must match the CPU
reference to round-off. Colouring and the pattern are built on the device.
"""

import numpy as np
import pytest

from zvcfd.fv.reference import Fluid, ReferenceSolver
from zvcfd.mesh.generate import box, delaunay_box, tube

pytestmark = pytest.mark.gpu

MESHES = {
    "tet": lambda: box((3, 3, 3), kind="tet", perturb=0.25, seed=1),
    "pyramid": lambda: box((2, 2, 2), kind="pyramid", perturb=0.15, seed=2),
    "wedge": lambda: box((3, 3, 3), kind="wedge", perturb=0.25, seed=3),
    "hex": lambda: box((3, 3, 3), kind="hex", perturb=0.25, seed=4),
    "mixed": lambda: tube(0.5, 1.0, n_core=2, n_ring=2, n_axial=3, kind="mixed", layers=1,
                          perturb=0.1),
    "delaunay": lambda: delaunay_box((3, 3, 3), seed=5),
}


def _random_state(s, rng):
    s.U = rng.normal(size=(s.N, 3))
    s.P = rng.normal(size=s.N)
    for K in s.kinds:
        K.mdot = rng.normal(size=K.mdot.shape)
    s._frozen_beta = rng.uniform(0, 1, (s.N, 3))


def test_colouring_and_pattern():
    import cupy as cp

    from zvcfd.fv.fast import gpu_colour, gpu_node_graph
    from zvcfd.fv.pattern import node_graph

    m = tube(0.5, 2.0, n_core=6, n_ring=6, n_axial=24, kind="mixed", layers=3)
    for e in m.elements.values():
        col = cp.asnumpy(gpu_colour(cp.asarray(e), m.n_nodes))
        assert (col >= 0).all()
        for c in range(col.max() + 1):
            nodes = e[col == c].ravel()
            assert len(np.unique(nodes)) == len(nodes)
        again = cp.asnumpy(gpu_colour(cp.asarray(e), m.n_nodes))
        np.testing.assert_array_equal(col, again)               # deterministic
    ip, ix = gpu_node_graph(m.elements, m.n_nodes)
    hp, hx = node_graph(m)
    np.testing.assert_array_equal(cp.asnumpy(ip), hp)
    np.testing.assert_array_equal(cp.asnumpy(ix), hx)


def _box_tets(n):
    """Five tetrahedra per cube of an ``n³`` grid of cubes, as a connectivity array only."""
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    v = lambda a, b, c: ((i + a) * (n + 1) + (j + b)) * (n + 1) + (k + c)  # noqa: E731
    c = [v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0),
         v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)]
    tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7), (1, 3, 4, 6)]
    return np.stack([np.stack([c[t].ravel() for t in tet], 1) for tet in tets], 0) \
        .reshape(-1, 4).astype(np.int32), (n + 1) ** 3


def _check_colouring(conn, col):
    order = np.argsort(col, kind="stable")
    for c in np.unique(col):
        nodes = conn[order][col[order] == c].ravel()
        assert len(np.unique(nodes)) == len(nodes), c


def test_colouring_of_large_and_high_valence_meshes():
    import cupy as cp

    from zvcfd.fv.fast import gpu_colour

    # more than 2²⁰ elements: most lose the first round (a count that once overflowed
    # into the out-of-colours flag)
    conn, nn = _box_tets(76)
    assert len(conn) > 2 ** 21
    col = cp.asnumpy(gpu_colour(cp.asarray(conn), nn))
    assert col.min() == 0 and col.max() < 64
    _check_colouring(conn, col)
    # 200 elements sharing one node need 200 colours: one 64-bit word overflows twice
    fan = np.column_stack([np.zeros(200, np.int64), 3 * np.arange(200) + 1,
                           3 * np.arange(200) + 2, 3 * np.arange(200) + 3]).astype(np.int32)
    narrow = cp.asnumpy(gpu_colour(cp.asarray(fan), 601, words=1))
    wide = cp.asnumpy(gpu_colour(cp.asarray(fan), 601, words=4))
    np.testing.assert_array_equal(narrow, wide)
    assert len(np.unique(narrow)) == 200


@pytest.mark.parametrize("store", [True, False])
@pytest.mark.parametrize("dt", [None, 0.3])
@pytest.mark.parametrize("gn", [False, True])
@pytest.mark.parametrize("name", sorted(MESHES))
def test_fast_matches_reference(name, gn, dt, store):
    import cupy as cp
    import scipy.sparse as sp

    from zvcfd.fv.fast import FastAssembler
    from zvcfd.rheology import CarreauYasuda

    m = MESHES[name]()
    rho, mu = 1.3, 0.7
    cy = CarreauYasuda(mu_0=0.9, mu_inf=0.1, lam=1.5)
    fl = Fluid(rho, mu, viscosity=cy if gn else None)
    s = ReferenceSolver(m, fl, {z: {"kind": "wall"} for z in m.zones},
                        advection="high-resolution", lag_rhie_chow=True, dt=dt, transpose=gn)
    rng = np.random.default_rng(11)
    _random_state(s, rng)
    s._store_old(first=True)
    s.assemble()
    M_raw, b_raw = s._raw
    if dt:
        t = rho * s.V / dt
        M_raw = M_raw - sp.diags(np.repeat(t, 4) * np.tile([1, 1, 1, 0], s.N))
        b_raw = b_raw - np.column_stack([t[:, None] * s.U, np.zeros(s.N)]).reshape(-1)
    g = FastAssembler(m, rho=rho, mu=mu, rheology=cy if gn else None, transpose=gn,
                      store_geometry=store)
    np.testing.assert_allclose(cp.asnumpy(g.node_volume), s.V, rtol=1e-12)
    gU, gP = g.gradient(s.U), g.gradient(s.P)
    np.testing.assert_allclose(cp.asnumpy(gU), s.nodal_gradient(s.U), atol=1e-11)
    np.testing.assert_allclose(cp.asnumpy(gP), s.nodal_gradient(s.P), atol=1e-11)
    mdot = g.to_slots({K.kind: cp.asarray(K.mdot.copy()) for K in s.kinds})
    diag = g.diagonal(mdot, s.U)
    np.testing.assert_allclose(cp.asnumpy(diag), s._a_space, rtol=1e-11, atol=1e-12)
    V = cp.asarray(s.V)
    tdiag = rho * V / dt if dt else 0.0
    dnode, snode = V / (diag + tdiag), V / diag
    beta = g.limiter(s.U, gU)
    np.testing.assert_allclose(cp.asnumpy(beta), s._beta(s.nodal_gradient(s.U)), atol=1e-12)
    levels = [(1.0, s.U.copy(), mdot)] if dt else []
    data, b = g.assemble(s.U, mdot, gU, s._frozen_beta, gP, dnode, snode, levels=levels)
    indptr, indices = (cp.asnumpy(a) for a in g.pattern)
    A = sp.bsr_matrix((cp.asnumpy(data), indices, indptr), shape=(4 * s.N, 4 * s.N)).tocsr()
    assert abs(A - M_raw).max() < 1e-11 * abs(M_raw).max()
    np.testing.assert_allclose(cp.asnumpy(b).reshape(-1), b_raw, atol=1e-11 * np.abs(b_raw).max())
    # mass flows from a new state, the previous one as the old level
    U_new, P_new = rng.normal(size=(s.N, 3)), rng.normal(size=s.N)
    U_old = s.U.copy()
    s.U, s.P = U_new, P_new
    s.update_mass_flows()
    new, imb, adv = g.massflow(U_new, P_new, g.gradient(P_new), dnode, snode,
                               levels=[(1.0, U_old, mdot)] if dt else [])
    back = g.from_slots(new)
    for K in s.kinds:
        np.testing.assert_allclose(cp.asnumpy(back[K.kind]), K.mdot,
                                   atol=1e-11 * np.abs(K.mdot).max())
    np.testing.assert_allclose(cp.asnumpy(imb), s._interior_out, atol=1e-11 * np.abs(imb).max())
    if not gn:     # the advective diagonal of the new flows: dvisc + adv = full diagonal
        full = g.diagonal(new, U_new)
        np.testing.assert_allclose(cp.asnumpy(g.dvisc + adv), cp.asnumpy(full),
                                   rtol=1e-12, atol=1e-12)


def test_fast_is_deterministic():
    import cupy as cp

    from zvcfd.fv.fast import FastAssembler

    m = MESHES["mixed"]()
    rng = np.random.default_rng(0)
    U, P = rng.normal(size=(m.n_nodes, 3)), rng.normal(size=m.n_nodes)
    out = []
    for _ in range(2):
        g = FastAssembler(m, rho=1.0, mu=0.5)
        md = g.zero_mdot()
        gU = g.gradient(U)
        d, b = g.assemble(U, md, gU, cp.ones((m.n_nodes, 3)), g.gradient(P),
                          cp.ones(m.n_nodes), cp.ones(m.n_nodes))
        out.append(cp.asnumpy(d).tobytes() + cp.asnumpy(b).tobytes())
    assert out[0] == out[1]
