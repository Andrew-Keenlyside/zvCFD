"""Pressure outlets on surface caps: patch links with an anti-bounce-back condition.

The SimVascular comparison found that non-equilibrium extrapolation on
patch cells lost part of the imposed pressure where the flow crosses an
outlet off the lattice axes, growing as tau -> 1/2 (17 % of the pressure
difference in a (1,1,1) pipe at tau = 0.55). Pressure outlets on caps now
carry the condition on the lattice links that cross the cap.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from zvcfd.boundary import FLUID, PRESSURE_FLAG
from zvcfd.geometry import voxelize_mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "simvascular"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _cylinder_boundary(h=0.1):
    from test_geometry import capped_cylinder

    tri, zone = capped_cylinder(r=1.0, length=3.0)
    return voxelize_mesh(tri, zone, {1: "wall", 2: "velocity-inlet", 3: "pressure-outlet"}, h)


def test_cap_links_cover_an_aligned_cap():
    dom, bnd, grid, rep = _cylinder_boundary()
    uses = bnd.uses_links()
    assert list(uses) == [False, True]                  # inlet on cells, outlet on links
    # the outlet's link cells stay fluid; the inlet's patch cells are flagged
    b3 = dom.brick ** 3
    out = bnd.link_cells[bnd.link_patch == 1]
    assert (dom.flags[out // b3, out % b3] == FLUID).all()
    inl = bnd.cells[bnd.patch == 0]
    assert (dom.flags[inl // b3, inl % b3] == 4).all()
    # every link cell of the +x cap has the link arriving along -x (q = 2) ...
    m = bnd.link_mask[bnd.link_patch == 1]
    assert (m & (1 << 2)).all()
    # ... and there is one per voxel of the cap's last layer
    assert len(out) == pytest.approx(np.pi / 0.1 ** 2, rel=0.1)
    assert bnd.counts()[1] == len(out)


def test_kind_decides_the_scheme():
    dom, bnd, grid, rep = _cylinder_boundary()
    from zvcfd.boundary import Patch

    # an outlet turned into a velocity patch goes back to patch cells
    bnd.patches[1] = Patch("out", "velocity", normal=bnd.patches[1].normal)
    bnd.apply_flags(dom)
    assert not bnd.uses_links().any()
    b3 = dom.brick ** 3
    c = bnd.cells[bnd.patch == 1]
    assert (dom.flags[c // b3, c % b3] == 4).all()
    assert (bnd.nbr[bnd.patch == 1] >= 0).all()
    # and a lumped outlet is a pressure patch: links again
    bnd.patches[1] = Patch("out", "rcr", rcr=(1.0, 1.0, 1.0), normal=bnd.patches[1].normal)
    bnd.apply_flags(dom)
    assert bnd.patches[1].flag == PRESSURE_FLAG and bnd.uses_links()[1]
    assert (dom.flags[c // b3, c % b3] == FLUID).all()


@pytest.mark.gpu
@pytest.mark.parametrize("axis", [(0, 0, 1), (1, 1, 1)])
def test_oblique_outlets_hold_their_pressure(axis):
    import oblique_outlet as ob

    rows = [ob.run_case(axis, radius=6.0, length=96.0, tau=tau, log=lambda *_: None)
            for tau in (0.55, 0.8)]
    for r in rows:
        # the pressure the flow feels at each cap is the imposed one, to the staircase
        # placement of the cap (under 1.5 % of the pressure difference); before the fix,
        # the (1,1,1) pipe lost 9 % and 7 % at tau = 0.55
        assert abs(r["jump_in_pct"]) < 1.5 and abs(r["jump_out_pct"]) < 1.5, r
    # and it does not depend on tau
    assert rows[0]["flux_vs_hp_at_imposed_dp_pct"] == pytest.approx(
        rows[1]["flux_vs_hp_at_imposed_dp_pct"], abs=0.2)


@pytest.mark.gpu
def test_links_conserve_mass_and_are_bit_identical_across_partitions():
    import cupy as cp

    from zvcfd.lbm import MultiLBM, SparseLBM

    d1, b1, _, _ = _cylinder_boundary()
    d2, b2, _, _ = _cylinder_boundary()
    one = SparseLBM(d1, tau=0.6, collision="trt", boundary=b1)
    many = MultiLBM(d2, n_parts=3, chunk_bricks=1, devices=[0], boundary=b2, tau=0.6,
                    collision="trt")
    for s in (one, many):
        s.set_patch(0, u_zyx=(0.0, 0.0, 0.01))
        s.set_patch(1, rho=1.0)
    one.step(4000)
    many.step(4000)
    assert many.ghost_fraction > 0
    f1 = {k: cp.asnumpy(v) for k, v in one.fields().items()}
    f2 = many.fields()
    for k in f1:
        assert np.array_equal(f1[k], f2[k]), k
    q1, q2 = one.patch_flux(), many.patch_flux()
    assert np.allclose(q1, q2, rtol=1e-6, atol=0)
    q_in, q_out = q1
    assert q_in > 0 and -q_out == pytest.approx(q_in, rel=5e-3)
