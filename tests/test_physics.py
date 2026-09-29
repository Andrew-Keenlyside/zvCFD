"""Physics checks for the MVP solver: TRT walls, pressure/velocity patches,
Windkessel outlets, Carreau-Yasuda rheology, and multi-partition bit-identity."""

import numpy as np
import pytest

from zvcfd.boundary import Patch, face_patches
from zvcfd.domain import BrickDomain

H = 30          # fluid layers between the plates


def channel(nx=64):
    """Plates at z = 0 and z = H+1, periodic in y, open (patch) ends in x."""
    flags = np.zeros((H + 2, 8, nx), np.uint8)
    flags[0] = flags[-1] = 1
    return flags


def profile(sim, dom, x):
    import cupy as cp

    ux = dom.to_dense(cp.asnumpy(sim.fields()["ux"]))
    return ux[1:-1, 4, x]


def test_rcr_steady_state():
    p = Patch("o", "rcr", pressure=100.0, rcr=(1e8, 1e-10, 9e8))
    q = 2e-6
    for _ in range(20000):
        pr = p.rcr_pressure(q, 1e-3)
    assert pr == pytest.approx(100.0 + (1e8 + 9e8) * q, rel=1e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("tau", [0.6, 1.5])
def test_trt_poiseuille_exact_walls(tau):
    from zvcfd.lbm import SparseLBM

    F = 1e-6
    flags = channel(16)
    dom = BrickDomain.from_flags(flags, periodic=(False, True, True))
    sim = SparseLBM(dom, tau=tau, force=(F, 0, 0), collision="trt")
    nu = (tau - 0.5) / 3
    sim.step(int(12 * H * H / nu / 9))
    zw = np.arange(H) + 0.5
    exact = F / (2 * nu) * zw * (H - zw)
    got = profile(sim, dom, 8)
    # TRT with Lambda = 3/16 is exact for this flow at any tau (walls mid-link)
    assert np.abs(got - exact).max() / exact.max() < 1e-4


@pytest.mark.gpu
def test_pressure_patches_drive_poiseuille():
    from zvcfd.lbm import SparseLBM

    nx = 64
    dom = BrickDomain.from_flags(channel(nx), periodic=(False, True, False))
    b = face_patches(dom, {"xmin": Patch("in", "pressure"), "xmax": Patch("out", "pressure")})
    tau = 0.8
    sim = SparseLBM(dom, tau=tau, collision="trt", boundary=b)
    drho = 1e-3
    sim.set_patch(0, rho=1 + drho / 2)
    sim.set_patch(1, rho=1 - drho / 2)
    nu = (tau - 0.5) / 3
    sim.step(30000)
    grad = drho / 3 / (nx - 1)                 # pressure gradient, cs^2 = 1/3
    zw = np.arange(H) + 0.5
    exact = grad / (2 * nu) * zw * (H - zw)
    got = profile(sim, dom, nx // 2)
    assert np.abs(got - exact).max() / exact.max() < 0.02


@pytest.mark.gpu
def test_patch_flux_is_the_interior_mass_flux():
    import cupy as cp

    from zvcfd.lbm import SparseLBM

    nx = 64
    dom = BrickDomain.from_flags(channel(nx), periodic=(False, True, False))
    b = face_patches(dom, {"xmin": Patch("in", "velocity"), "xmax": Patch("out", "pressure")})
    sim = SparseLBM(dom, tau=0.8, collision="trt", boundary=b)
    sim.set_patch(0, u_zyx=(0.0, 0.0, 0.01))     # inward = +x
    sim.set_patch(1, rho=1.0)
    sim.step(20000)
    f = {k: dom.to_dense(cp.asnumpy(v)) for k, v in sim.fields().items()}
    interior = (f["rho"][1:-1, :, 32] * f["ux"][1:-1, :, 32]).sum()
    q_in, q_out = sim.patch_flux()
    assert q_in == pytest.approx(interior, rel=2e-3)
    assert -q_out == pytest.approx(interior, rel=2e-3)


@pytest.mark.gpu
def test_flow_rate_control_hits_target():
    from zvcfd.lbm import SparseLBM

    nx = 64
    dom = BrickDomain.from_flags(channel(nx), periodic=(False, True, False))
    b = face_patches(dom, {"xmin": Patch("in", "velocity"), "xmax": Patch("out", "pressure")})
    sim = SparseLBM(dom, tau=0.8, collision="trt", boundary=b)
    target = 0.01 * H * 8                          # plug 0.01 over the H x 8 cross-section
    u = 0.01
    sim.set_patch(1, rho=1.0)
    for _ in range(40):
        sim.set_patch(0, u_zyx=(0.0, 0.0, u))
        sim.step(1000)
        u *= 1 + 0.5 * (target / sim.patch_flux()[0] - 1)
    assert sim.patch_flux()[0] == pytest.approx(target, rel=2e-3)


@pytest.mark.gpu
def test_carreau_yasuda_shear_thins():
    from zvcfd.lbm import SparseLBM
    from zvcfd.lbm.solver import CarreauYasuda

    F = 2e-6
    flags = channel(16)
    dom = BrickDomain.from_flags(flags, periodic=(False, True, True))
    newt = SparseLBM(dom, tau=1.0, force=(F, 0, 0), collision="trt")
    cy = CarreauYasuda(nu_0=1 / 6, nu_inf=1 / 60, lam=2e4, a=2.0, n=0.35)
    thin = SparseLBM(dom, tau=1.0, force=(F, 0, 0), collision="trt", rheology=cy)
    newt.step(20000)
    thin.step(20000)
    a, b = profile(newt, dom, 8), profile(thin, dom, 8)
    assert a.mean() / a.max() == pytest.approx(2 / 3, abs=0.01)     # parabolic
    assert b.mean() / b.max() > a.mean() / a.max() + 0.02           # blunter
    assert b.max() > a.max()                                         # less viscous overall


@pytest.mark.gpu
def test_multi_partition_is_bit_identical():
    import cupy as cp

    from zvcfd.lbm import MultiLBM, SparseLBM
    from zvcfd.phantoms import vessel_network

    fluid = vessel_network((64, 64, 128), n=10, r=(3.0, 6.0))
    flags = np.where(fluid, 0, 1).astype(np.uint8)
    d1 = BrickDomain.from_flags(flags)
    b1 = face_patches(d1, {"xmin": Patch("in", "velocity"), "xmax": Patch("out", "pressure")})
    d2 = BrickDomain.from_flags(flags)
    b2 = face_patches(d2, {"xmin": Patch("in", "velocity"), "xmax": Patch("out", "pressure")})
    one = SparseLBM(d1, tau=0.7, collision="trt", boundary=b1)
    many = MultiLBM(d2, n_parts=3, chunk_bricks=2, devices=[0], boundary=b2, tau=0.7,
                    collision="trt")
    for s in (one, many):
        s.set_patch(0, u_zyx=(0.0, 0.0, 0.01))
        s.set_patch(1, rho=1.0)
    one.step(300)
    many.step(300)
    assert many.ghost_fraction > 0
    f1 = {k: cp.asnumpy(v) for k, v in one.fields().items()}
    f2 = many.fields()
    for k in f1:
        assert np.array_equal(f1[k], f2[k]), k
