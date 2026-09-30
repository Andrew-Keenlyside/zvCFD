"""Uniform refinement and grid-convergence tools."""

import numpy as np
import pytest

from zvcfd.fv.geometry import dual_geometry
from zvcfd.mesh.generate import box, delaunay_box, tube
from zvcfd.mesh.refine import refine
from zvcfd.verification import gci_two_grids, grid_study, observed_order


@pytest.mark.parametrize("name", ["tet", "wedge", "hex", "pyramid", "mixed", "delaunay"])
def test_refine_is_conforming_and_exact(name):
    m = {"tet": lambda: box((2, 2, 2), kind="tet", perturb=0.2),
         "wedge": lambda: box((2, 2, 2), kind="wedge", perturb=0.2),
         "hex": lambda: box((2, 2, 2), kind="hex", perturb=0.2),
         "pyramid": lambda: box((2, 2, 2), kind="pyramid"),
         "mixed": lambda: tube(0.5, 1.0, kind="mixed", n_axial=2),
         "delaunay": lambda: delaunay_box((3, 3, 3))}[name]()
    f = refine(m)
    per = {"tet": 8, "wedge": 8, "hex": 8}
    if name != "pyramid":
        assert f.counts() == {k: per[k] * v for k, v in m.counts().items()}
    else:
        assert f.counts() == {"tet": 4 * 48, "pyramid": 6 * 48}
    assert f.volume() == pytest.approx(m.volume(), rel=1e-13)
    f.faces()                                           # raises unless conforming
    assert dual_geometry(f).report["closure_max_rel"] < 1e-13
    for k, z in m.zones.items():
        assert f.zones[k].n_faces == 4 * z.n_faces
        assert len(f.zones[k].cells) == f.zones[k].n_faces


def test_grid_study_recovers_order_and_limit():
    h = np.array([0.1, 0.2, 0.4])
    f = 3.0 + 0.7 * h ** 2 + 0.2 * h ** 3
    s = grid_study(*f, r=2.0)
    assert s.order == pytest.approx(2.0, abs=0.2)
    assert s.extrapolated == pytest.approx(3.0, abs=1e-3)
    assert s.gci_fine < 0.01 and not s.oscillatory
    assert s.asymptotic_ratio == pytest.approx(1.0, abs=0.1)
    assert observed_order(1.0, 1.0 + 1e-3, 1.0 + 4e-3, 2.0) == pytest.approx(np.log2(3.0))
    assert gci_two_grids(1.0, 1.03) == pytest.approx(3 * 0.03 / 3)
