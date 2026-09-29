import numpy as np
import pytest

from zvcfd.geometry import voxelize_mesh


def capped_cylinder(r=1.0, length=4.0, n=64, m=32):
    """Closed triangulated cylinder along x: wall zone 1, inlet cap 2 (x=0), outlet cap 3."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.linspace(0, length, m + 1)
    ring = np.stack([np.cos(th), np.sin(th)], 1) * r
    tris, zone = [], []
    for i in range(m):
        for k in range(n):
            k2 = (k + 1) % n
            p = [np.r_[xs[i], ring[k]], np.r_[xs[i + 1], ring[k]],
                 np.r_[xs[i + 1], ring[k2]], np.r_[xs[i], ring[k2]]]
            tris += [[p[0], p[1], p[2]], [p[0], p[2], p[3]]]
            zone += [1, 1]
    for x, z in ((0.0, 2), (length, 3)):
        c = np.r_[x, 0.0, 0.0]
        for k in range(n):
            tris.append([c, np.r_[x, ring[k]], np.r_[x, ring[(k + 1) % n]]])
            zone.append(z)
    return np.asarray(tris), np.asarray(zone)


def test_cylinder_volume_and_caps():
    tri, zone = capped_cylinder()
    h = 0.05
    dom, bnd, grid, rep = voxelize_mesh(tri, zone, {1: "wall", 2: "velocity-inlet",
                                                   3: "pressure-outlet"}, h)
    assert rep["odd_columns_dropped"] == 0
    exact = np.pi * 1.0 ** 2 * 4.0 * (np.sin(np.pi / 64) * 64 / np.pi) / h ** 3
    assert rep["fluid_voxels"] == pytest.approx(exact, rel=0.02)
    cap = np.pi / h ** 2
    counts = bnd.counts()
    assert counts[0] == pytest.approx(cap, rel=0.1) and counts[1] == pytest.approx(cap, rel=0.1)
    assert bnd.patches[0].kind == "velocity" and bnd.patches[1].kind == "pressure"
    # the inlet cap sits at x = 0, so its outward normal is -x (zyx order)
    assert np.allclose(bnd.patches[0].normal, (0, 0, -1), atol=1e-6)
    assert (bnd.nbr >= 0).all()


@pytest.mark.gpu
def test_cylinder_flow_conserves_mass():
    from zvcfd.lbm import SparseLBM

    tri, zone = capped_cylinder(r=1.0, length=3.0)
    dom, bnd, grid, rep = voxelize_mesh(tri, zone, {1: "wall", 2: "velocity-inlet",
                                                   3: "pressure-outlet"}, 0.1)
    sim = SparseLBM(dom, tau=0.8, collision="trt", boundary=bnd)
    sim.set_patch(0, u_zyx=(0.0, 0.0, 0.01))
    sim.set_patch(1, rho=1.0)
    sim.step(6000)
    q_in, q_out = sim.patch_flux()
    assert q_in > 0 and -q_out == pytest.approx(q_in, rel=5e-3)
