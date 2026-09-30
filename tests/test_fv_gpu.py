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


@pytest.mark.parametrize("stokes", [True, False])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("name", sorted(MESHES))
def test_gpu_matches_reference(name, transpose, stokes):
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

    g = GPUAssembler(m, rho=rho, mu=mu, transpose=transpose, stokes=stokes)
    gU = g.gradient(s.U)
    gP = g.gradient(s.P)
    np.testing.assert_allclose(cp.asnumpy(g.node_volume), s.V, rtol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(gU), s.nodal_gradient(s.U), atol=1e-11)
    np.testing.assert_allclose(cp.asnumpy(gP), s.nodal_gradient(s.P), atol=1e-11)
    mdot = {K.kind: K.mdot for K in s.kinds}
    diag = g.diagonal(mdot)
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
