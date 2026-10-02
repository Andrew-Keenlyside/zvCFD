"""The partitioned solver (zvcfd.fv.multi) reaches the single-GPU solution.

Partitions run on one device here; halo copies between devices use the
same code path with peer copies. The checks:

- partitions own whole Morton chunks, balanced, and their local meshes hold
  complete control volumes: after the halo exchange, gradients equal the
  global ones at every local node;
- one partition is the single-GPU ACM solver to round-off;
- with several, the converged fields equal the single-GPU ones to the
  solver tolerance (only the preconditioner is block-Jacobi), for High
  Resolution flow, two stiff lumped outlets and a transient.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from zvcfd.fv.reference import Fluid
from zvcfd.lumped import RCR
from zvcfd.mesh.generate import tube

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fv_boundaries import _two_outlet_box, _womersley_case  # noqa: E402
from test_fv_reference import _pipe_case  # noqa: E402

pytestmark = pytest.mark.gpu


def test_partitions_and_halo_exchange():
    import cupy as cp

    from zvcfd.fv.gpu import GPUAssembler
    from zvcfd.fv.multi import MultiGPUSolver
    from zvcfd.fv.partition import morton_owner

    m = tube(0.5, 2.0, n_core=3, n_ring=3, n_axial=16, kind="mixed", layers=1)
    owner = morton_owner(m.nodes, 4)
    counts = np.bincount(owner)
    assert counts.max() / counts.mean() < 1.25
    m2, bcs = _pipe_case()
    s = MultiGPUSolver(m2, Fluid(1.0, 0.2), bcs, parts=3)
    rng = np.random.default_rng(0)
    phi = rng.normal(size=m2.n_nodes)
    ref = cp.asnumpy(GPUAssembler(m2).gradient(phi))
    grads = [p.asm.gradient(cp.asarray(phi[p.lm.nodes])) for p in s.parts]
    s.exchange(grads)
    for p, g in zip(s.parts, grads):
        np.testing.assert_allclose(cp.asnumpy(g), ref[p.lm.nodes], atol=1e-12)
    assert sum(p.n_own for p in s.local) == m2.n_nodes


def _pair(m, bcs, parts, **kw):
    from zvcfd.fv.multi import MultiGPUSolver
    from zvcfd.fv.solver import GPUSolver

    one = GPUSolver(m, kw.pop("fluid", Fluid(1.0, 0.2)), bcs, linear="acm", **kw) \
        if parts == 0 else None
    return one if one is not None else MultiGPUSolver(m, kw.pop("fluid", Fluid(1.0, 0.2)),
                                                      bcs, parts=parts, **kw)


def test_one_partition_is_the_single_gpu_solver():
    m, bcs = _pipe_case()
    a = _pair(m, bcs, 0, advection=1.0)
    b = _pair(m, bcs, 1, advection=1.0)
    ra = a.solve(max_iterations=8, tol=0.0)
    rb = b.solve(max_iterations=8, tol=0.0)
    np.testing.assert_allclose(b.fields()["U"], a.fields()["U"], atol=1e-12)
    for x, y in zip(ra.history, rb.history):
        assert abs(x["rms_u"] - y["rms_u"]) <= 1e-9 * max(x["rms_u"], 1e-30)


@pytest.mark.parametrize("parts", [2, 4])
def test_partitioned_pipe(parts):
    m, bcs = _pipe_case()
    a = _pair(m, bcs, 0, advection=1.0)
    a.solve(max_iterations=400, tol=1e-11)
    b = _pair(m, bcs, parts, advection=1.0)
    rep = b.solve(max_iterations=400, tol=1e-11)
    assert rep.converged
    np.testing.assert_allclose(b.fields()["U"], a.fields()["U"], atol=1e-8)
    np.testing.assert_allclose(b.fields()["P"], a.fields()["P"],
                               atol=1e-8 * np.ptp(a.fields()["P"]))
    assert abs(rep.history[-1]["imbalance"]) < 1e-11


def test_partitioned_high_resolution():
    """High Resolution freezes its limiter on the iteration path, which the
    preconditioner changes: agreement at the known path-dependence level
    (docs/validation, ~10⁻⁴), not round-off."""
    m, bcs = _pipe_case()
    a = _pair(m, bcs, 0)
    a.solve(max_iterations=400, tol=1e-11)
    b = _pair(m, bcs, 3)
    assert b.solve(max_iterations=400, tol=1e-11).converged
    U = a.fields()["U"]
    assert np.abs(b.fields()["U"] - U).max() < 1e-3 * np.abs(U).max()


def test_partitioned_lumped_outlets():
    m, z = _two_outlet_box()
    walls = {k: {"kind": "wall"} for n, k in z.items() if n not in ("xmin", "out_a", "out_b")}
    inlet = {z["xmin"]: {"kind": "velocity", "flow_rate": 0.02, "profile": "poiseuille"}}

    def bcs():
        return inlet | walls | {
            z["out_a"]: {"kind": "pressure", "lumped": RCR(rp=1e4, c=1.0, rd=9e4)},
            z["out_b"]: {"kind": "pressure", "lumped": RCR(rp=2e4, c=1.0, rd=18e4)}}

    fl = Fluid(1.0, 0.05)
    a = _pair(m, bcs(), 0, fluid=fl, advection=1.0)
    a.solve(max_iterations=200, tol=1e-9)
    b = _pair(m, bcs(), 3, fluid=fl, advection=1.0)
    assert b.solve(max_iterations=200, tol=1e-9).converged
    qa, qb = a.patch_flows(), b.patch_flows()
    for name in ("out_a", "out_b"):
        assert abs(qa[z[name]] - qb[z[name]]) < 1e-7 * 0.02
    pa, pb = a.patch_pressures(), b.patch_pressures()
    assert abs(pa[z["out_a"]] - pb[z["out_a"]]) < 1e-6 * pa[z["out_a"]]


def test_partitioned_transient():
    m, bcs, nu = _womersley_case()
    fl = Fluid(1.0, nu)
    a = _pair(m, bcs, 0, fluid=fl, advection=1.0)
    a.solve_transient([0.05, 0.08, 0.04, 0.06], loops=40, tol=1e-12)
    b = _pair(m, bcs, 2, fluid=fl, advection=1.0)
    b.solve_transient([0.05, 0.08, 0.04, 0.06], loops=40, tol=1e-12)
    np.testing.assert_allclose(b.fields()["U"], a.fields()["U"], atol=1e-9)
    nodes_a, tau_a = a.wall_shear()
    nodes_b, tau_b = b.wall_shear()
    np.testing.assert_array_equal(nodes_a, nodes_b)
    np.testing.assert_allclose(tau_b, tau_a, atol=1e-8 * np.abs(tau_a).max())
