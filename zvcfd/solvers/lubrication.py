"""Lubrication (local-Poiseuille) pressure solve on a voxel mask.

``div(k grad p) = 0`` over the fluid voxels with ``k = 3/4 d^2``, where
``d`` is the distance to the half-way wall; for a circular tube this gives
the exact Poiseuille conductance (``int k dA = pi R^4 / 8``). Walls carry
no flux; Dirichlet pressures are set on any voxel set (inlets, and any
number of outlets, e.g. the 77 outlets of the HiP-CT coronary case).

Uses: a seconds-scale flow and pressure estimate on very large masks (the
"tier 0" solver), boundary-pressure estimates for sub-volume LBM runs, and
the elliptic kernel that multigrid is exercised on (``docs/multiresolution.md``).

It is the analogue of Fluent's hybrid initialisation (a Laplace solve), but
with a geometry-aware conductance. As an initial condition for LBM it did
*not* shorten time to steady state in our tests; see the spike results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy import ndimage


@dataclass
class LubricationResult:
    pressure: np.ndarray            # (z, y, x); nan outside the fluid
    velocity: np.ndarray            # (3, z, y, x) as (uz, uy, ux); zero outside the fluid
    iterations: int
    seconds: float
    method: str

    def flux(self, axis: int, index: int, voxel: float = 1.0) -> float:
        """Volume flux through the plane ``index`` normal to ``axis`` (0 z, 1 y, 2 x)."""
        u = np.take(self.velocity[axis], index, axis=axis)
        return float(np.nansum(u)) * voxel ** 2


def conductance(fluid: np.ndarray, voxel: float = 1.0) -> np.ndarray:
    """``k = 3/4 d^2`` with ``d`` the distance (in ``voxel`` units) to the half-way wall."""
    d = ndimage.distance_transform_edt(fluid).astype(np.float64) - 0.5
    return 0.75 * (np.maximum(d, 0.5) * voxel) ** 2


def system(fluid: np.ndarray, fixed: np.ndarray, value: np.ndarray, k: np.ndarray):
    """Assemble the SPD system over free fluid voxels.

    Returns ``(A, b, index)`` with ``index`` the unknown number of each voxel (-1 if not free).
    """
    free = fluid & ~fixed
    index = -np.ones(fluid.shape, np.int64)
    index[free] = np.arange(int(free.sum()))
    n = int(free.sum())
    diag = np.zeros(n)
    rhs = np.zeros(n)
    rows, cols, vals = [], [], []
    for ax in range(3):
        for sgn in (-1, 1):
            nb = np.roll(fluid, -sgn, axis=ax)
            edge = [slice(None)] * 3
            edge[ax] = -1 if sgn == 1 else 0
            nb[tuple(edge)] = False            # no wrap across the box
            ok = free & nb
            kn = np.roll(k, -sgn, axis=ax)
            kf = (2 * k * kn / (k + kn))[ok]   # harmonic mean on the face
            i = index[ok]
            j = np.roll(index, -sgn, axis=ax)[ok]
            fj = np.roll(fixed, -sgn, axis=ax)[ok]
            vj = np.roll(value, -sgn, axis=ax)[ok]
            np.add.at(diag, i, kf)
            np.add.at(rhs, i[fj], kf[fj] * vj[fj])
            rows.append(i[~fj])
            cols.append(j[~fj])
            vals.append(-kf[~fj])
    A = sp.csr_matrix((np.concatenate(vals + [diag]),
                       (np.concatenate(rows + [np.arange(n)]),
                        np.concatenate(cols + [np.arange(n)]))), shape=(n, n))
    return A, rhs, index


def solve(fluid: np.ndarray, fixed: np.ndarray, value: np.ndarray, *, voxel: float = 1.0,
          mu: float = 1.0, method: str = "auto", tol: float = 1e-8) -> LubricationResult:
    """Pressure and Darcy velocity ``u = -(k/mu) grad p`` on ``fluid``.

    Args:
        fluid: Boolean fluid mask (z, y, x).
        fixed: Boolean mask of Dirichlet voxels (subset of ``fluid``).
        value: Pressure on the ``fixed`` voxels (same shape; other entries ignored).
        voxel: Voxel edge (physical length unit); ``k`` and gradients use it.
        mu: Dynamic viscosity.
        method: ``"amg"`` (pyamg smoothed aggregation + CG, CPU), ``"cg"``
            (Jacobi-preconditioned CG; on the GPU when cupy is present), or
            ``"auto"`` (amg if pyamg imports).
    """
    fluid = np.asarray(fluid, bool)
    fixed = np.asarray(fixed, bool) & fluid
    k = conductance(fluid, voxel)
    A, b, index = system(fluid, fixed, np.asarray(value, np.float64), k)
    if method == "auto":
        try:
            import pyamg  # noqa: F401
            method = "amg"
        except ImportError:
            method = "cg"
    x0 = np.full(A.shape[0], float(np.mean(value[fixed])) if fixed.any() else 0.0)
    t = time.time()
    if method == "amg":
        import pyamg

        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=500)
        res: list[float] = []
        x = ml.solve(b, x0=x0, tol=tol, accel="cg", residuals=res)
        iters = len(res) - 1
    elif method == "cg":
        x, iters = _cg(A, b, x0, tol)
    else:
        raise ValueError(f"unknown method {method!r}")
    seconds = time.time() - t
    p = np.full(fluid.shape, np.nan)
    p[index >= 0] = x
    p[fixed] = np.asarray(value)[fixed]
    grads = np.gradient(p, voxel)
    vel = np.stack([np.nan_to_num(-(k / mu) * g, nan=0.0) * fluid for g in grads])
    return LubricationResult(p, vel, iters, seconds, method)


def _cg(A, b, x0, tol):
    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as csl
    except ImportError:
        import scipy.sparse.linalg as sl

        it = [0]
        x, _ = sl.cg(A, b, x0=x0, M=sp.diags(1.0 / A.diagonal()), rtol=tol, maxiter=100_000,
                     callback=lambda _: it.__setitem__(0, it[0] + 1))
        return x, it[0]
    Ad = csp.csr_matrix(A)
    dinv = cp.asarray(1.0 / A.diagonal())
    M = csl.LinearOperator(A.shape, matvec=lambda v: dinv * v)
    it = [0]
    x, _ = csl.cg(Ad, cp.asarray(b), x0=cp.asarray(x0), M=M, tol=tol, maxiter=100_000,
                  callback=lambda _: it.__setitem__(0, it[0] + 1))
    return cp.asnumpy(x), it[0]
