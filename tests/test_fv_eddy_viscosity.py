"""The eddy-viscosity hook of the finite-volume solvers (``mu_t``, ``turbulence``).

A RANS model sets ``solver.mu_t``, an eddy viscosity at the nodes, from
``turbulence.update(solver)`` after every outer iteration. The checks:

- a constant ``mu_t`` is a laminar fluid with ``μ + mu_t``, to round-off,
  with fused and classic kernels;
- a varying ``mu_t`` (with the transpose term it switches on) gives the same
  solution on the GPU, fused or classic, as on the CPU reference;
- the natural outlet's transpose traction takes ``mu_t`` too;
- ``turbulence.update`` runs once per outer iteration, its convergence
  measure is recorded and holds convergence back until it is small;
- the partitioned solver refuses an eddy viscosity rather than ignore it.
"""

import numpy as np
import pytest

from zvcfd.fv.reference import Fluid, ReferenceSolver
from zvcfd.mesh.generate import tube

pytestmark = pytest.mark.gpu
R, L, MU = 0.5, 2.0, 0.05


def _case():
    m = tube(R, L, n_core=3, n_ring=3, n_axial=12, kind="mixed", layers=1)
    z = {zz.name: k for k, zz in m.zones.items()}
    bcs = {z["inlet"]: {"kind": "velocity", "flow_rate": 0.2},
           z["outlet"]: {"kind": "pressure", "value": 0.0}, z["wall"]: {"kind": "wall"}}
    return m, z, bcs


def _mut_field(m):
    """A smooth eddy viscosity, zero on the wall, up to 0.8 μ on the axis."""
    r = np.hypot(m.nodes[:, 0], m.nodes[:, 1]) / R
    zeta = m.nodes[:, 2] / L
    return 0.8 * MU * np.clip(1 - r ** 2, 0, None) * (1 + 0.5 * np.sin(np.pi * zeta))


def _gpu(m, bcs, mu=MU, **kw):
    from zvcfd.fv.solver import GPUSolver

    return GPUSolver(m, Fluid(1.0, mu), bcs, advection=1.0, linear="host-direct",
                     linear_rtol=0.0, **kw)


@pytest.mark.parametrize("kernels", ["fast", "classic"])
def test_constant_eddy_viscosity_is_a_laminar_fluid(kernels):
    m, z, bcs = _case()
    c = 0.03
    a = _gpu(m, bcs, kernels=kernels)
    a.mu_t = np.full(m.n_nodes, c)
    assert a.transpose
    assert a.solve(max_iterations=200, tol=1e-11).converged
    b = _gpu(m, bcs, mu=MU + c, kernels=kernels, transpose=True)
    assert b.solve(max_iterations=200, tol=1e-11).converged
    Ua, Ub = a.fields()["U"], b.fields()["U"]
    assert np.abs(Ua - Ub).max() < 1e-10 * np.abs(Ub).max()
    assert np.abs(a.fields()["P"] - b.fields()["P"]).max() < 1e-9 * np.ptp(b.fields()["P"])


def test_varying_eddy_viscosity_matches_the_reference():
    m, z, bcs = _case()
    mut = _mut_field(m)
    ref = ReferenceSolver(m, Fluid(1.0, MU), bcs, advection=1.0)
    ref.mu_t = mut
    assert ref.transpose
    assert ref.solve(max_iterations=200, tol=1e-12).converged
    scale = np.abs(ref.U).max()
    for kernels in ("fast", "classic"):
        g = _gpu(m, bcs, kernels=kernels)
        g.mu_t = mut
        assert g.solve(max_iterations=200, tol=1e-12).converged
        assert np.abs(g.fields()["U"] - ref.U).max() < 1e-9 * scale
    # and it acts: the laminar solution is clearly different
    lam = _gpu(m, bcs)
    lam.solve(max_iterations=200, tol=1e-11)
    assert np.abs(lam.fields()["U"] - ref.U).max() > 1e-3 * scale


def test_outlet_traction_takes_the_eddy_viscosity():
    import cupy as cp

    m, z, bcs = _case()
    g = _gpu(m, bcs)
    g.mu_t = _mut_field(m)
    gradU = g._mirror_d(cp.ascontiguousarray(g.asm.gradient(g.U)))
    mu = cp.asnumpy(g._node_viscosity_d(gradU))
    assert np.allclose(mu, MU + _mut_field(m))
    assert g._has_aout                       # built though transpose was off at set-up
    g.mu_t = None
    assert not g.transpose                   # back to the laminar default


class _Model:
    """A stand-in turbulence model: sets ``mu_t`` and reports a shrinking change."""

    def __init__(self, mut, settle=6):
        self.mut, self.settle, self.calls = mut, settle, 0

    def update(self, solver):
        self.calls += 1
        solver.mu_t = self.mut
        return 1.0 if self.calls < self.settle else 0.0


def test_turbulence_hook_runs_each_outer_iteration():
    m, z, bcs = _case()
    g = _gpu(m, bcs)
    model = _Model(_mut_field(m), settle=40)
    g.turbulence = model
    rep = g.solve(max_iterations=200, tol=1e-10)
    assert rep.converged
    assert model.calls == rep.iterations >= 40      # convergence waited for the model
    assert all("turbulence" in h for h in rep.history)
    ref = ReferenceSolver(m, Fluid(1.0, MU), bcs, advection=1.0)
    ref.mu_t = model.mut
    ref.solve(max_iterations=200, tol=1e-12)
    assert np.abs(g.fields()["U"] - ref.U).max() < 1e-8 * np.abs(ref.U).max()


def test_partitioned_solver_refuses_an_eddy_viscosity():
    from zvcfd.fv.multi import MultiGPUSolver

    m, z, bcs = _case()
    s = MultiGPUSolver(m, Fluid(1.0, MU), bcs, parts=2)
    s.mu_t = np.zeros(m.n_nodes)
    with pytest.raises(NotImplementedError):
        s.solve(max_iterations=2)
