"""The GPU kernels and the CPU reference evaluate the same discrete equations (fuzzed).

Random states on distorted meshes of every element type: node velocities
and pressures, lagged ip mass flows and limiter blend factors. The GPU's
nodal gradients, momentum diagonal and ip-flux residual must match the
CPU reference's to round-off. The kernels are written from the spec, not
ported, so agreement checks both.
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
    N = s.N
    s.U = rng.normal(size=(N, 3))
    s.P = rng.normal(size=N)
    for K in s.kinds:
        K.mdot = rng.normal(size=K.mdot.shape)
    s._frozen_beta = rng.uniform(0, 1, (N, 3))


@pytest.mark.parametrize("store", [True, False])
@pytest.mark.parametrize("stokes", [True, False])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("name", sorted(MESHES))
def test_gpu_matches_reference(name, transpose, stokes, store):
    import cupy as cp

    from zvcfd.fv.gpu import GPUAssembler

    m = MESHES[name]()
    rho, mu = 1.3, 0.7
    s = ReferenceSolver(m, Fluid(rho, mu), {z: {"kind": "wall"} for z in m.zones},
                        advection="high-resolution", stokes=stokes, transpose=transpose)
    rng = np.random.default_rng(hash((name, transpose, stokes)) % 2 ** 32)
    _random_state(s, rng)
    M, b = s.assemble()
    M_raw, b_raw = s._raw
    x = np.concatenate([s.U, s.P[:, None]], 1).reshape(-1)
    r_cpu = (M_raw @ x - b_raw).reshape(-1, 4)

    g = GPUAssembler(m, rho=rho, mu=mu, transpose=transpose, stokes=stokes, store_geometry=store)
    gU = g.gradient(s.U)
    gP = g.gradient(s.P)
    np.testing.assert_allclose(cp.asnumpy(g.node_volume), s.V, rtol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(gU), s.nodal_gradient(s.U), atol=1e-11)
    np.testing.assert_allclose(cp.asnumpy(gP), s.nodal_gradient(s.P), atol=1e-11)
    mdot = {K.kind: K.mdot for K in s.kinds}
    diag = g.diagonal(mdot, s.U)
    np.testing.assert_allclose(cp.asnumpy(diag), s._a_space, rtol=1e-11, atol=1e-12)
    dnode = s.rhie_chow * cp.asarray(s.V) / diag
    beta = s._frozen_beta if not stokes else np.zeros((s.N, 3))
    r_gpu = cp.asnumpy(g.residual(s.U, s.P, mdot, gU, beta, gP, dnode))
    scale = np.abs(r_cpu).max()
    np.testing.assert_allclose(r_gpu, r_cpu, atol=1e-11 * scale)


def test_gpu_is_deterministic():
    import cupy as cp

    from zvcfd.fv.gpu import GPUAssembler

    m = MESHES["mixed"]()
    g = GPUAssembler(m, rho=1.0, mu=0.5)
    rng = np.random.default_rng(0)
    U, P = rng.normal(size=(m.n_nodes, 3)), rng.normal(size=m.n_nodes)
    runs = [cp.asnumpy(g.gradient(U)).tobytes() + cp.asnumpy(g.gradient(P)).tobytes()
            for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_colouring_is_valid():
    from zvcfd.fv.gpu import colour_elements

    m = MESHES["delaunay"]()
    e = m.elements["tet"]
    cols = colour_elements(e, m.n_nodes)
    assert sum(len(c) for c in cols) == len(e)
    for c in cols:
        nodes = e[c].reshape(-1)
        assert len(np.unique(nodes)) == len(nodes)


def _pattern(m):
    import cupy as cp

    from zvcfd.fv.pattern import node_graph

    indptr, indices = node_graph(m)
    return indptr, indices, (cp.asarray(indptr, dtype=cp.int64), cp.asarray(indices,
                                                                            dtype=cp.int32))


@pytest.mark.parametrize("store", [True, "tets", False])
@pytest.mark.parametrize("dt", [None, 0.3])
@pytest.mark.parametrize("gn", [False, True])
@pytest.mark.parametrize("name", ["tet", "wedge", "hex", "pyramid", "mixed", "delaunay"])
def test_gpu_assembly_matches_reference(name, gn, dt, store):
    """Block matrix and right-hand side (lagged Rhie–Chow), limiter and mass flows."""
    import cupy as cp
    import scipy.sparse as sp

    from zvcfd.fv.gpu import GPUAssembler
    from zvcfd.rheology import CarreauYasuda

    m = MESHES[name]()
    rho, mu = 1.3, 0.7
    cy = CarreauYasuda(mu_0=0.9, mu_inf=0.1, lam=1.5)
    fl = Fluid(rho, mu, viscosity=cy if gn else None)
    s = ReferenceSolver(m, fl, {z: {"kind": "wall"} for z in m.zones},
                        advection="high-resolution", lag_rhie_chow=True, dt=dt,
                        transpose=gn)
    rng = np.random.default_rng(7)
    _random_state(s, rng)
    s._store_old(first=True)          # false time step: the old level is the current state
    s.assemble()
    M_raw, b_raw = s._raw
    # time term the caller adds: remove it from the CPU matrix for the comparison
    if dt:
        t = rho * s.V / dt
        M_raw = M_raw - sp.diags(np.repeat(t, 4) * np.tile([1, 1, 1, 0], s.N))
        b_raw = b_raw - np.column_stack([t[:, None] * s.U, np.zeros(s.N)]).reshape(-1)
    g = GPUAssembler(m, rho=rho, mu=mu, rheology=cy if gn else None, transpose=gn,
                     store_geometry=store)
    indptr, indices, pat = _pattern(m)
    mdot = {K.kind: K.mdot.copy() for K in s.kinds}      # the CPU updates K.mdot in place
    gU, gP = g.gradient(s.U), g.gradient(s.P)
    diag = g.diagonal(mdot, s.U)
    np.testing.assert_allclose(cp.asnumpy(diag), s._a_space, rtol=1e-11, atol=1e-12)
    tdiag = rho * cp.asarray(s.V) / dt if dt else 0.0
    V = cp.asarray(s.V)
    dnode, snode = V / (diag + tdiag), V / diag
    beta = g.limiter(s.U, gU)
    np.testing.assert_allclose(cp.asnumpy(beta), s._beta(s.nodal_gradient(s.U)), atol=1e-12)
    levels = [(1.0, s.U.copy(), mdot)] if dt else []
    data, b = g.assemble(pat, s.U, mdot, gU, s._frozen_beta, gP, dnode, snode, levels=levels)
    A = sp.bsr_matrix((cp.asnumpy(data), indices, indptr), shape=(4 * s.N, 4 * s.N)).tocsr()
    diff = abs(A - M_raw)
    assert diff.max() < 1e-11 * abs(M_raw).max()
    np.testing.assert_allclose(cp.asnumpy(b).reshape(-1), b_raw, atol=1e-11 * np.abs(b_raw).max())
    # mass flows from a new state, with the previous state as the old level
    U_new, P_new = rng.normal(size=(s.N, 3)), rng.normal(size=s.N)
    U_old = s.U.copy()
    s.U, s.P = U_new, P_new
    s.update_mass_flows()
    new, imb = g.massflow(U_new, P_new, g.gradient(P_new), dnode, snode,
                          levels=[(1.0, U_old, mdot)] if dt else [])
    for K in s.kinds:
        np.testing.assert_allclose(cp.asnumpy(new[K.kind]), K.mdot,
                                   atol=1e-11 * np.abs(K.mdot).max())
    np.testing.assert_allclose(cp.asnumpy(imb), s._interior_out, atol=1e-11 * np.abs(imb).max())
