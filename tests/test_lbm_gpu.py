import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_sparse_equals_dense(porous64):
    import cupy as cp

    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import DenseLBM, SparseLBM, dense_from_bricks

    dom = BrickDomain.from_flags(porous64, periodic=True)
    d = DenseLBM(porous64, force=(1e-5, 0, 0))
    s = SparseLBM(dom, force=(1e-5, 0, 0))
    d.step(200)
    s.step(200)
    fluid = porous64 == 0
    for k, v in d.fields().items():
        sv = dense_from_bricks(dom, cp.asnumpy(s.fields()[k]))
        assert np.array_equal(cp.asnumpy(v)[fluid], sv[fluid]), k


def test_plane_poiseuille():
    """Force-driven flow between half-way bounce-back walls: u(z) = F/(2 nu) z (H - z)."""
    import cupy as cp

    from zvcfd.lbm import DenseLBM

    H, F, tau = 32, 1e-6, 1.0
    flags = np.zeros((H + 2, 4, 64), np.uint8)
    flags[0] = flags[-1] = 1
    sim = DenseLBM(flags, tau=tau, force=(F, 0, 0))
    sim.step(12000)
    ux = cp.asnumpy(sim.fields()["ux"])[1:-1, 0, 0]
    zw = np.arange(H) + 0.5
    exact = F / (2 * sim.nu) * zw * (H - zw)
    assert np.abs(ux - exact).max() / exact.max() < 0.01


def test_fp16_tracks_fp32(porous64):
    import cupy as cp

    from zvcfd.lbm import DenseLBM

    a = DenseLBM(porous64, force=(1e-5, 0, 0))
    b = DenseLBM(porous64, force=(1e-5, 0, 0), half=True)
    a.step(300)
    b.step(300)
    fluid = porous64 == 0
    ua = cp.asnumpy(a.fields()["ux"])[fluid]
    ub = cp.asnumpy(b.fields()["ux"])[fluid]
    assert np.abs(ua - ub).max() / np.abs(ua).max() < 0.05
