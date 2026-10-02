"""Pressure-driven developed pipe flow: an exact solution at every Reynolds number.

With fixed pressure at both ends and natural velocity there, developed
Poiseuille flow satisfies the Navier–Stokes equations and both boundary
conditions, so the flow rate does not depend on Re: the discrete Stokes flow
rate is the reference.

Momentum entering through a pressure boundary is ``ṁ_b u_node``, while the
faces leaving the node carry the deferred correction. The first row of
control volumes was then inconsistent: the flow entered as a jet, and the
flow rate came out high by an error that did not shrink with refinement
(+1.8 %, +3.4 % and +3.8 % with 4, 6 and 12 cells across). Now the
corrections of those faces stay out of the inflow node's own row and its
boundary flux carries them (``GPUSolver._own_inflow``), plus the cross-stream
part of the difference where those faces do not mirror the boundary faces
(``zvcfd.fv.boundary_advection.inflow_offsets``). On an extruded mesh that
term is zero; with a tetrahedral inlet it takes the error from −1.5 % to
under 0.7 % (docs/spec/fv_numerics.md).
"""

import numpy as np
import pytest

from zvcfd.fv.reference import Fluid
from zvcfd.mesh.generate import tube

pytestmark = pytest.mark.gpu
R, L, MU, RHO, DP = 0.5, 4.0, 0.02, 1.0, 2.56          # Re = 50 on the developed mean velocity


def _pipe(nc, kind="mixed"):
    extra = {"layers": max(1, nc // 2)} if kind == "mixed" else {}
    return tube(R, L, n_core=nc, n_ring=nc, n_axial=8 * nc, kind=kind, growth=0.85, **extra)


def _solve(m, max_its=100, **kw):
    from zvcfd.fv.linear import amgx_library
    from zvcfd.fv.solver import GPUSolver

    zid = {z.name: k for k, z in m.zones.items()}
    bcs = {zid["inlet"]: {"kind": "pressure", "value": DP},
           zid["outlet"]: {"kind": "pressure", "value": 0.0}, zid["wall"]: {"kind": "wall"}}
    lin = {"linear": "amgx"} if amgx_library() else {"linear": "host-direct", "linear_rtol": 0.0}
    s = GPUSolver(m, Fluid(RHO, MU), bcs, **lin, **kw)
    rep = s.solve(max_iterations=300, tol=1e-9)
    if hasattr(s.linear, "close"):
        s.linear.close()
    assert rep.converged and rep.iterations < max_its
    U = s.fields()["U"]
    axis = np.hypot(m.nodes[:, 0], m.nodes[:, 1]) < 1e-9
    z = m.nodes[:, 2]
    centre = [float(U[axis & np.isclose(z, zk), 2].mean()) for zk in (0.0, L / 2)]
    return -s.zone_flows()[zid["inlet"]] / RHO, centre


def test_extruded_inflow_is_consistent():
    errs = []
    for nc in (6, 12):
        m = _pipe(nc, "wedge")
        qs, _ = _solve(m, stokes=True)
        q, centre = _solve(m)                            # High Resolution, the default
        errs.append(q / qs - 1)
        assert centre[0] < 1.02 * centre[1]              # no inflow jet
        if nc == 6:                                      # every edge axial or in the plane:
            qu, _ = _solve(m, advection="upwind")        # upwind is exact in developed flow
            assert abs(qu / qs - 1) < 1e-8
    # measured +0.56 % and +0.20 %; dropping the corrections altogether: +1.30 %, +1.25 %
    assert abs(errs[1]) < 0.5 * abs(errs[0]) and abs(errs[0]) < 0.008


def test_tetrahedral_inflow_is_consistent():
    errs = []
    for nc in (4, 6):
        m = _pipe(nc)
        qs, _ = _solve(m, stokes=True)
        q, centre = _solve(m)
        errs.append(q / qs - 1)
        assert centre[0] < 1.02 * centre[1]
    # measured +0.66 % and +0.08 %; without the cross-stream term -1.53 % and -1.71 %
    assert abs(errs[1]) < 0.5 * abs(errs[0]) and abs(errs[0]) < 0.01


def test_specified_blend_is_path_independent():
    m = _pipe(4)
    q1, _ = _solve(m, advection=1.0)
    q2, _ = _solve(m, 150, advection=1.0, dt=0.5)       # another, slower iteration path
    assert abs(q2 / q1 - 1) < 1e-6
