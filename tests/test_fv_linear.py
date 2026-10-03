"""Linear solvers on hard coupled systems (GPU, AmgX): thin cells and strongly graded layers.

Point-block smoothers of the coupled 4 × 4 system fail on cells thin in one
direction; the checks here hold the default AmgX configuration (symmetric
multicolour Gauss–Seidel), the robust one, the SIMPLE block preconditioner
and the :class:`~zvcfd.fv.linear.Auto` ladder to a direct solve on such
systems (docs/spec/fv_numerics.md).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from zvcfd.fv.linear import amgx_library
from zvcfd.fv.reference import Fluid
from zvcfd.mesh.generate import cylinder_channel, tube

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "validation"))

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not amgx_library(), reason="needs AmgX (ZVCFD_AMGX_LIB)")]
R = 0.5


def _system(m, bcs, fluid, u0, dt=None):
    import cupy as cp  # noqa: F401

    from zvcfd.fv.linear import HostDirect
    from zvcfd.fv.solver import GPUSolver

    s = GPUSolver(m, fluid, bcs, linear="acm", advection=1.0, dt=dt)
    s.initialise(U=u0)
    s.solve(max_iterations=2, tol=0.0)
    A, b = s.assemble()
    x0 = s._state()
    s._scale_rows(A, b)
    b = b.reshape(-1)
    xd, _ = HostDirect().solve(A, b, x0)
    return A, b, x0, xd


@pytest.fixture(scope="module")
def layered_tube():
    m = tube(R, 2.0, n_core=8, n_ring=8, n_axial=32, kind="mixed", layers=6, growth=0.45)
    z = {b.name: k for k, b in m.zones.items()}

    def prof(x):
        r2 = (x[:, 0] ** 2 + x[:, 1] ** 2) / R ** 2
        return np.stack([0 * r2, 0 * r2, 2 * np.clip(1 - r2, 0, None)], 1)

    bcs = {z["inlet"]: {"kind": "velocity", "value": prof},
           z["outlet"]: {"kind": "pressure", "value": 0.0}, z["wall"]: {"kind": "wall"}}
    return _system(m, bcs, Fluid(1.0, 0.01), prof)


@pytest.fixture(scope="module")
def dfg_slab():
    from exact import DFG_2D2  # noqa: F401

    H, UMAX = 0.41, 1.5
    m = cylinder_channel(16)
    z = {b.name: k for k, b in m.zones.items()}

    def inflow(x):
        return np.stack([4 * UMAX * x[:, 1] * (H - x[:, 1]) / H ** 2, 0 * x[:, 0], 0 * x[:, 0]], 1)

    bcs = {z["inlet"]: {"kind": "velocity", "value": inflow},
           z["outlet"]: {"kind": "pressure", "value": 0.0}, z["walls"]: {"kind": "wall"},
           z["cylinder"]: {"kind": "wall"}, z["front"]: {"kind": "symmetry"},
           z["back"]: {"kind": "symmetry"}}
    return _system(m, bcs, Fluid(1.0, 1e-3), inflow, dt=0.005)


def _check(sol, system, its):
    import cupy as cp

    A, b, x0, xd = system
    x, info = sol.solve(A, b, x0, rtol=1e-6)
    assert info.converged and info.iterations <= its, info
    assert float(cp.abs(x - xd).max()) < 1e-3 * float(cp.abs(xd).max())


@pytest.mark.parametrize("precision", ["double", "mixed"])
def test_default_amgx_on_hard_systems(dfg_slab, layered_tube, precision):
    from zvcfd.fv.linear import AMGX

    _check(AMGX(precision=precision), dfg_slab, 25)          # 17 measured
    _check(AMGX(precision=precision), layered_tube, 70)      # 17 measured


def test_explicit_config_in_mixed_precision(dfg_slab):
    """Mixed precision wraps zvCFD's FGMRES around one V-cycle: an explicit ``config`` must
    shape that V-cycle (it once fell back to the default preset)."""
    from zvcfd.fv.linear import AMGX, DEFAULT_AMGX_CONFIG, amgx_aggregation, amgx_smoother

    cfg = DEFAULT_AMGX_CONFIG % {"rtol": 1e-6, "maxiter": 200, "reuse": 0,
                                 "smoother": amgx_smoother("ilu0"),
                                 "aggregation": amgx_aggregation("SIZE_2", 3)}
    its = []
    for kw in ({"preset": "robust"}, {"config": cfg}, {"preset": "gs"}):
        s = AMGX(rtol=1e-6, precision="mixed", **kw)
        its.append(s.solve(*dfg_slab[:3], rtol=1e-6)[1].iterations)
        s.close()
    assert its[0] == its[1] != its[2]                  # measured 11, 11, 17


def test_robust_amgx_on_the_dfg_slab(dfg_slab):
    from zvcfd.fv.linear import AMGX

    _check(AMGX(preset="robust"), dfg_slab, 40)
    x, info = AMGX(preset="dilu").solve(*dfg_slab[:3], rtol=1e-6)
    assert not info.converged                       # the usual configuration stalls here


def test_simple_on_graded_layers(layered_tube):
    from zvcfd.fv.linear import SIMPLE

    _check(SIMPLE(), layered_tube, 120)


def test_auto_ladder(layered_tube, dfg_slab):
    from zvcfd.fv.linear import Auto

    a = Auto(rtol=1e-6)
    _check(a, dfg_slab, 40)
    assert a.active == "amgx-gs" and not a.switches
    _check(a, layered_tube, 200)
    b = Auto(rtol=1e-6, maxiter=3)                  # no rung reaches 10⁻⁶ in 3 iterations
    x, info = b.solve(*layered_tube[:3], rtol=1e-6)
    assert [sw[:2] for sw in b.switches] == [("amgx-gs", "amgx-robust"),
                                             ("amgx-robust", "amgx-dilu-p"),
                                             ("amgx-dilu-p", "simple")]
    assert b.active == "simple" and x is not None


def test_auto_falls_back_on_a_non_finite_solve(dfg_slab):
    """A diverged smoother returns NaN, and ``NaN > tol`` is False: the ladder must still
    count it as a failure and move on (AmgX gs on a thin slab with the transpose term)."""
    import cupy as cp

    from zvcfd.fv.linear import Auto, SolveInfo

    class Diverged:
        name = "diverged"

        def solve(self, A, b, x0=None, rtol=0.1):
            return cp.full_like(b, cp.nan), SolveInfo(200, float("nan"), 0.0, converged=False)

    a = Auto(rtol=0.1)
    a.ladder[0] = ("diverged", Diverged)
    a.current = Diverged()
    x, info = a.solve(*dfg_slab[:3], rtol=0.1)
    assert a.switches[0][:2] == ("diverged", "amgx-robust")
    assert "non-finite" in a.switches[0][2]
    assert info.converged and bool(cp.isfinite(x).all())
    a.ladder = [("diverged", Diverged)]                 # nothing left to fall back on
    a.index, a.current = 0, Diverged()
    with pytest.raises(RuntimeError, match="every linear solver failed"):
        a.solve(*dfg_slab[:3], rtol=0.1)


def test_auto_probe(dfg_slab, layered_tube):
    from zvcfd.fv.linear import Auto

    a = Auto(rtol=0.1, probe_at=1)
    _, info = a.solve(*dfg_slab[:3], rtol=0.1)
    assert info.converged and a.probe is not None
    assert a.active in ("amgx-gs", "amgx-robust")          # the faster of the two, by time
    if a.active == "amgx-robust":
        assert a.switches[0][2].startswith("probe")
    for sysm in (dfg_slab, layered_tube):                    # later systems still solve
        _, info = a.solve(*sysm[:3], rtol=0.1)
        assert info.converged
    a.close()
