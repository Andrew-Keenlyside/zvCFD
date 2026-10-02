"""Linear solvers for the coupled 4 × 4 block system, on the GPU.

Three back ends behind one interface, ``solver.solve(A, b, x0) -> (x, info)``:

- :class:`HostDirect`: SuperLU on the host. Exact, for validation and
  small meshes only.
- :class:`AMGX`: NVIDIA AmgX (BSD-3) through a thin ctypes binding: FGMRES
  preconditioned by
  aggregation (:data:`AMGX_PRESETS`: symmetric multicolour Gauss–Seidel
  smoothing, pressure-weighted aggregation of four), block size 4.
  The matrix structure is uploaded once; later outer iterations
  replace coefficients and re-set up the hierarchy.
- :class:`ACM`: zvCFD's own additive-correction multigrid (Hutchinson &
  Raithby 1986; Raw 1996), the method CFX describes. Pairwise aggregation
  on block coupling strength (twice per level, about 4:1), coarse operators
  by summing blocks (Galerkin with piecewise-constant transfer), damped
  block-Jacobi smoothing, a dense solve on the coarsest level, and V-cycles
  as the preconditioner of a GPU FGMRES.

The block matrix :class:`BlockMatrix` is block-CSR: sorted column indices
per block row, blocks 4 × 4 row-major, the layout
:meth:`zvcfd.fv.gpu.GPUAssembler.assemble` writes and AmgX reads, plus
optional rank-one terms ``u wᵀ`` (the implicit pressure–flow coupling of
lumped outlets). The Krylov solvers apply them in every product; the
multigrid preconditioners see only the sparse part, so each rank-one
term costs about one extra FGMRES iteration.

References: B. R. Hutchinson, G. D. Raithby, Numer. Heat Transfer 9, 511
(1986); M. J. Raw, AIAA paper 96-0297 (1996); Y. Notay, *An aggregation-based
algebraic multigrid method*, Electron. Trans. Numer. Anal. 37, 123 (2010)
(pairwise aggregation); Y. Saad, *A flexible inner-outer preconditioned
GMRES algorithm*, SIAM J. Sci. Comput. 14, 461 (1993).
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import time
import weakref
from dataclasses import dataclass, field

import numpy as np

_SPMV = r"""
extern "C" __global__ void bsr_mv(long long n, const long long *indptr, const int *indices,
                                  const double *data, const double *x, double *y) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    double acc[4] = {0, 0, 0, 0};
    for (long long k = indptr[i]; k < indptr[i + 1]; ++k) {
        const double *blk = data + k * 16;
        const double *xj = x + (long long)indices[k] * 4;
        for (int r = 0; r < 4; ++r)
            acc[r] += blk[r * 4] * xj[0] + blk[r * 4 + 1] * xj[1] + blk[r * 4 + 2] * xj[2]
                      + blk[r * 4 + 3] * xj[3];
    }
    for (int r = 0; r < 4; ++r) y[i * 4 + r] = acc[r];
}
"""
_kernels: dict = {}


def _spmv_kernel():
    import cupy as cp

    dev = cp.cuda.Device().id
    if dev not in _kernels:
        _kernels[dev] = cp.RawKernel(_SPMV, "bsr_mv")
    return _kernels[dev]


@dataclass
class BlockMatrix:
    """Block-CSR 4 × 4 matrix on the device."""

    indptr: object        # (N + 1,) int64
    indices: object       # (nnzb,) int32
    data: object          # (nnzb, 4, 4) float64
    # rank-one terms u wᵀ: (u_dofs, u_values, w_dofs, w_values), dofs into the flat 4N vector
    lowrank: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.indptr.size) - 1

    @property
    def nnzb(self) -> int:
        return int(self.indices.size)

    def matvec(self, x):
        import cupy as cp

        y = cp.empty(self.n * 4)
        k = _spmv_kernel()
        x = x.reshape(-1)
        k(((self.n + 127) // 128,), (128,), (np.int64(self.n), self.indptr, self.indices,
                                            self.data.reshape(-1), x, y))
        for ui, uv, wi, wv in self.lowrank:
            y[ui] += uv * cp.dot(wv, x[wi])
        return y

    def row_of_block(self):
        import cupy as cp

        rows = getattr(self, "_rows", None)
        if rows is None or rows.size != self.nnzb:
            rows = cp.searchsorted(self.indptr, cp.arange(self.nnzb, dtype=cp.int64),
                                   side="right") - 1
            self._rows = rows
        return rows

    def diagonal_positions(self):
        """Position of each row's diagonal block (the pattern includes it)."""
        import cupy as cp

        rows = self.row_of_block()
        hit = cp.flatnonzero(rows == self.indices.astype(cp.int64))
        return hit

    def to_scipy(self):
        import cupy as cp
        import scipy.sparse as sp

        M = sp.bsr_matrix((cp.asnumpy(self.data), cp.asnumpy(self.indices),
                           cp.asnumpy(self.indptr)), shape=(4 * self.n, 4 * self.n))
        for ui, uv, wi, wv in self.lowrank:
            u = sp.csr_matrix((cp.asnumpy(uv), (cp.asnumpy(ui), np.zeros(ui.size, int))),
                              shape=(4 * self.n, 1))
            w = sp.csr_matrix((cp.asnumpy(wv), (np.zeros(wi.size, int), cp.asnumpy(wi))),
                              shape=(1, 4 * self.n))
            M = M.tocsr() + u @ w
        return M


@dataclass
class SolveInfo:
    iterations: int
    residual: float          # final residual relative to the initial one (x0)
    seconds: float
    setup_seconds: float = 0.0
    converged: bool = True


# ---------------------------------------------------------------- host direct

class HostDirect:
    """SuperLU on the host: exact; for validation and small meshes (ignores ``rtol``)."""

    name = "host-direct"

    def solve(self, A: BlockMatrix, b, x0=None, rtol: float = 0.0):
        import cupy as cp
        import scipy.sparse.linalg as spl

        t = time.time()
        M = A.to_scipy().tocsc()
        bh = cp.asnumpy(b.reshape(-1))
        x = spl.splu(M).solve(bh)
        x0 = np.zeros_like(bh) if x0 is None else cp.asnumpy(x0.reshape(-1))
        r0 = np.linalg.norm(M @ x0 - bh) or 1.0
        return cp.asarray(x), SolveInfo(1, float(np.linalg.norm(M @ x - bh) / r0), time.time() - t)


# ---------------------------------------------------------------- FGMRES

ROUNDOFF = 1e-13


def fgmres(A: BlockMatrix, b, x0, precond, *, rtol=1e-6, restart=30, maxiter=300,
           norm=None, dot=None):
    """Flexible GMRES (Saad 1993) on the device, right-preconditioned by ``precond(r)``.

    Converges when the residual has fallen by ``rtol`` relative to the
    *initial* residual (AmgX's ``RELATIVE_INI``, CFX's residual reduction per
    outer iteration), or to round-off (``ROUNDOFF`` × ‖b‖), since a system
    whose initial guess already solves it cannot be reduced further. Returns
    ``(x, iterations, residual relative to the initial one, converged)``.
    ``norm`` and ``dot`` default to CuPy's; a distributed vector passes its own.
    """
    import cupy as cp

    if norm is None and dot is None and isinstance(x0, cp.ndarray):
        return _fgmres_device(A, b, x0, precond, rtol=rtol, restart=restart, maxiter=maxiter)
    norm = norm or (lambda v: float(cp.linalg.norm(v)))
    dot = dot or (lambda u, v: float(cp.dot(u, v)))
    x = x0.copy()
    bnorm = norm(b - A.matvec(x)) or 1.0
    floor = ROUNDOFF * norm(b) / bnorm
    rtol = max(rtol, floor)
    it = 0
    while True:
        r = b - A.matvec(x)
        beta = norm(r)
        res = beta / bnorm
        if res <= rtol or it >= maxiter:
            break
        V, Z = [r / beta], []
        H = np.zeros((restart + 1, restart))
        g = np.zeros(restart + 1)
        g[0] = beta
        cs, sn = np.zeros(restart), np.zeros(restart)
        kk = 0
        for k in range(restart):
            z = precond(V[k])
            Z.append(z)
            w = A.matvec(z)
            for j in range(k + 1):                       # modified Gram-Schmidt
                H[j, k] = dot(w, V[j])
                w -= H[j, k] * V[j]
            hnext = norm(w)
            H[k + 1, k] = hnext
            for j in range(k):                           # earlier Givens rotations
                t = cs[j] * H[j, k] + sn[j] * H[j + 1, k]
                H[j + 1, k] = -sn[j] * H[j, k] + cs[j] * H[j + 1, k]
                H[j, k] = t
            den = np.hypot(H[k, k], H[k + 1, k])
            cs[k], sn[k] = (H[k, k] / den, H[k + 1, k] / den) if den else (1.0, 0.0)
            H[k, k], H[k + 1, k] = den, 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            it += 1
            kk = k + 1
            res = abs(g[k + 1]) / bnorm
            if res <= rtol or it >= maxiter or hnext == 0.0:
                break
            V.append(w / hnext)
        y = np.linalg.solve(np.triu(H[:kk, :kk]), g[:kk])
        for j in range(kk):
            x += y[j] * Z[j]
        if res <= rtol or it >= maxiter:
            break
    res = norm(b - A.matvec(x)) / bnorm
    return x, it, res, res <= 1.01 * rtol


def _fgmres_device(A, b, x0, precond, *, rtol, restart, maxiter):
    """:func:`fgmres` with the Krylov basis in one device array.

    Classical Gram–Schmidt with one reorthogonalisation pass (Daniel, Gragg,
    Kaufman & Stewart 1976), as four matrix–vector products against the
    stored basis, so each iteration has one host synchronisation (for the
    small Hessenberg update) instead of one per basis vector. The basis
    ``V`` and the preconditioned vectors ``Z`` are kept between calls of the
    same size, and grow in steps of 8 vectors up to ``restart``: with a good
    preconditioner most solves take a few iterations, and a full basis
    (2 × 31 vectors, 0.8 GB at 400 k nodes) would be mostly unused.
    """
    import cupy as cp

    n = int(b.size)

    def workspace(k):
        """``(V, Z)`` with room for ``k + 2`` basis and ``k + 1`` preconditioned vectors."""
        V, Z = _FGMRES_WS.get(n, (None, None))
        if V is None or V.shape[0] < min(k + 2, restart + 1):
            cap = min(restart, max(8, 8 * ((k + 1 + 7) // 8)))
            V2, Z2 = cp.empty((cap + 1, n)), cp.empty((cap, n))
            if V is not None and n in _FGMRES_WS:
                V2[:V.shape[0]] = V
                Z2[:Z.shape[0]] = Z
            else:
                _FGMRES_WS.clear()
            _FGMRES_WS[n] = V, Z = V2, Z2
        return V, Z

    V, Z = workspace(0)
    x = x0.copy()
    r = b - A.matvec(x)
    nb2, nr2 = (float(v) for v in cp.stack([cp.dot(b, b), cp.dot(r, r)]).get())
    bnorm = np.sqrt(nr2) or 1.0
    rtol = max(rtol, ROUNDOFF * np.sqrt(nb2) / bnorm)
    beta = np.sqrt(nr2)
    it = 0
    res = beta / bnorm
    while res > rtol and it < maxiter:
        V[0] = r / beta
        H = np.zeros((restart + 1, restart))
        g = np.zeros(restart + 1)
        g[0] = beta
        cs, sn = np.zeros(restart), np.zeros(restart)
        kk = 0
        for k in range(restart):
            V, Z = workspace(k)
            Z[k] = precond(V[k]).reshape(-1)
            w = A.matvec(Z[k])
            Vk = V[:k + 1]
            h1 = Vk @ w
            w -= h1 @ Vk
            h2 = Vk @ w                                 # reorthogonalise
            w -= h2 @ Vk
            h = h1 + h2
            hn = cp.sqrt(cp.dot(w, w))
            vals = cp.concatenate([h, hn[None]]).get()   # the one synchronisation
            hnext = float(vals[-1])
            H[:k + 1, k] = vals[:-1]
            H[k + 1, k] = hnext
            for j in range(k):
                t = cs[j] * H[j, k] + sn[j] * H[j + 1, k]
                H[j + 1, k] = -sn[j] * H[j, k] + cs[j] * H[j + 1, k]
                H[j, k] = t
            den = np.hypot(H[k, k], H[k + 1, k])
            cs[k], sn[k] = (H[k, k] / den, H[k + 1, k] / den) if den else (1.0, 0.0)
            H[k, k], H[k + 1, k] = den, 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            it += 1
            kk = k + 1
            res = abs(g[k + 1]) / bnorm
            if res <= rtol or it >= maxiter or hnext == 0.0:
                break
            V[k + 1] = w / hnext
        y = np.linalg.solve(np.triu(H[:kk, :kk]), g[:kk])
        x += cp.asarray(y) @ Z[:kk]
        r = b - A.matvec(x)
        beta = float(cp.sqrt(cp.dot(r, r)))
        res = beta / bnorm
    return x, it, res, res <= 1.01 * rtol


_FGMRES_WS: dict = {}


# ---------------------------------------------------------------- own ACM

class ACM:
    """Additive-correction multigrid (block, pairwise aggregation) inside FGMRES.

    Args:
        smooth: damped block-Jacobi sweeps before and after the coarse correction.
        omega: Jacobi damping.
        coarse: stop coarsening below this many block rows (dense solve there).
        passes: pairwise-aggregation passes per level (2: about 4:1).
        rtol, restart, maxiter: FGMRES controls.
        rebuild_every: re-aggregate every this many setups (coefficients are
            re-summed each time; the aggregation is kept between rebuilds).
    """

    name = "acm"

    def __init__(self, *, smooth: int = 2, omega: float = 0.7, coarse: int = 400,
                 passes: int = 2, rtol: float = 0.1, restart: int = 30, maxiter: int = 200,
                 rebuild_every: int = 10):
        self.smooth, self.omega, self.coarse, self.passes = smooth, omega, coarse, passes
        self.rtol, self.restart, self.maxiter = rtol, restart, maxiter
        self.rebuild_every = rebuild_every
        self._aggs = None
        self._setups = 0
        self.levels = []

    # aggregation (host, vectorised) ------------------------------------------------

    @staticmethod
    def _pairwise(n, rows, cols, strength):
        """Match nodes pairwise by strongest coupling; returns aggregate ids (n,) and count."""
        agg = -np.ones(n, np.int64)
        off = rows != cols
        r, c, s = rows[off], cols[off], strength[off]
        nxt = 0
        for _ in range(4):
            free_r = agg[r] < 0
            free_c = agg[c] < 0
            m = free_r & free_c
            if not m.any():
                break
            rr, cc, ss = r[m], c[m], s[m]
            order = np.lexsort((-ss, rr))
            rr, cc = rr[order], cc[order]
            first = np.r_[len(rr) > 0, rr[1:] != rr[:-1]][:len(rr)]
            best = -np.ones(n, np.int64)
            best[rr[first]] = cc[first]
            i = np.flatnonzero(best >= 0)
            j = best[i]
            mutual = (best[j] == i) & (i < j)
            a, bb = i[mutual], j[mutual]
            ids = nxt + np.arange(len(a))
            agg[a] = ids
            agg[bb] = ids
            nxt += len(a)
        # unmatched nodes join their strongest aggregated neighbour, else stay single
        left = np.flatnonzero(agg < 0)
        if len(left):
            m = (agg[r] < 0) & (agg[c] >= 0)
            rr, cc, ss = r[m], c[m], s[m]
            order = np.lexsort((-ss, rr))
            rr, cc = rr[order], cc[order]
            first = np.r_[len(rr) > 0, rr[1:] != rr[:-1]][:len(rr)]
            join = -np.ones(n, np.int64)
            join[rr[first]] = agg[cc[first]]
            to = join[left]
            agg[left[to >= 0]] = to[to >= 0]
            single = left[to < 0]
            agg[single] = nxt + np.arange(len(single))
            nxt += len(single)
        return agg, nxt

    def _aggregate(self, A: BlockMatrix):
        import cupy as cp

        rows = cp.asnumpy(A.row_of_block())
        cols = cp.asnumpy(A.indices).astype(np.int64)
        norms = cp.asnumpy(cp.sqrt((A.data ** 2).sum(axis=(1, 2))))
        n = A.n
        total = np.arange(n)
        count = n
        for _ in range(self.passes):
            diag = np.zeros(count)
            dmask = rows == cols
            diag[rows[dmask]] = norms[dmask]
            s = norms / np.sqrt(np.maximum(diag[rows] * diag[cols], 1e-300))
            a, count = self._pairwise(count, rows, cols, s)
            total = a[total]
            rows, cols = a[rows], a[cols]
            key = rows * count + cols
            uk, inv = np.unique(key, return_inverse=True)
            norms = np.bincount(inv.reshape(-1), norms, len(uk))
            rows, cols = uk // count, uk % count
        return total, count

    # coarse operator (device, deterministic) ----------------------------------------

    @staticmethod
    def _galerkin(A: BlockMatrix, agg, nc):
        import cupy as cp

        rows = agg[A.row_of_block()]
        cols = agg[A.indices.astype(cp.int64)]
        key = rows * nc + cols
        order = cp.argsort(key)
        ks = key[order]
        new = cp.concatenate([cp.ones(1, bool), ks[1:] != ks[:-1]])
        start = cp.flatnonzero(new)
        seg = cp.cumsum(new) - 1
        data = cp.zeros((int(start.size), 4, 4))
        cp.add.at(data, seg, A.data[order])
        uk = ks[start]
        crow = uk // nc
        indptr = cp.zeros(nc + 1, cp.int64)
        indptr[1:] = cp.cumsum(cp.bincount(crow, minlength=nc))
        return BlockMatrix(indptr, (uk % nc).astype(cp.int32), data)

    def setup(self, A: BlockMatrix):
        import cupy as cp

        t = time.time()
        rebuild = self._aggs is None or self._setups % self.rebuild_every == 0
        self._setups += 1
        if rebuild:
            self._aggs = []
        levels = []
        M = A
        depth = 0
        while True:
            dpos = M.diagonal_positions()
            Dinv = cp.linalg.inv(M.data[dpos])
            levels.append({"A": M, "Dinv": Dinv})
            if M.n <= self.coarse:
                dense = M.to_scipy().toarray()
                levels[-1]["dense_inv"] = cp.asarray(np.linalg.inv(dense))
                break
            if rebuild:
                agg_h, nc = self._aggregate(M)
                if nc >= 0.9 * M.n:                  # coarsening stalled: stop here
                    dense = M.to_scipy().toarray() if M.n <= 4000 else None
                    if dense is not None:
                        levels[-1]["dense_inv"] = cp.asarray(np.linalg.inv(dense))
                    break
                self._aggs.append((cp.asarray(agg_h), nc))
            if depth >= len(self._aggs):
                break
            agg, nc = self._aggs[depth]
            levels[-1]["agg"] = agg
            M = self._galerkin(M, agg, nc)
            depth += 1
        self.levels = levels
        return time.time() - t

    def _smooth(self, L, x, b, sweeps):
        import cupy as cp

        for _ in range(sweeps):
            r = (b - L["A"].matvec(x)).reshape(-1, 4)
            x = x + self.omega * cp.einsum("nij,nj->ni", L["Dinv"], r).reshape(-1)
        return x

    def vcycle(self, b, level=0):
        import cupy as cp

        L = self.levels[level]
        if "dense_inv" in L:
            return L["dense_inv"] @ b
        x = self._smooth(L, cp.zeros_like(b), b, self.smooth)
        if "agg" in L and level + 1 < len(self.levels):
            r = (b - L["A"].matvec(x)).reshape(-1, 4)
            nc = self.levels[level + 1]["A"].n
            rc = cp.zeros((nc, 4))
            cp.add.at(rc, L["agg"], r)
            ec = self.vcycle(rc.reshape(-1), level + 1)
            x = x + ec.reshape(-1, 4)[L["agg"]].reshape(-1)
        return self._smooth(L, x, b, self.smooth)

    def solve(self, A: BlockMatrix, b, x0=None, rtol=None):
        import cupy as cp

        ts = self.setup(A)
        t = time.time()
        x0 = cp.zeros(A.n * 4) if x0 is None else x0.reshape(-1).copy()
        x, it, res, ok = fgmres(A, b.reshape(-1), x0, self.vcycle,
                                rtol=self.rtol if rtol is None else rtol,
                                restart=self.restart, maxiter=self.maxiter)
        return x, SolveInfo(it, res, time.time() - t, ts, ok)

    def stats(self) -> dict:
        return {"levels": [lv["A"].n for lv in self.levels],
                "operator_complexity": sum(lv["A"].nnzb for lv in self.levels)
                / max(self.levels[0]["A"].nnzb, 1)}


# ---------------------------------------------------------------- AmgX

AMGX_MODE_DDDI = 8193        # device; double vectors, double matrix, int indices
AMGX_MODE_DDFI = 8449        # device; double vectors, float matrix (and hierarchy)
DEFAULT_AMGX_CONFIG = """{
 "config_version": 2,
 "determinism_flag": 1,
 "solver": {
   "solver": "FGMRES", "gmres_n_restart": 30, "max_iters": %(maxiter)d,
   "convergence": "RELATIVE_INI", "tolerance": %(rtol)g, "norm": "L2", "monitor_residual": 1,
   "use_scalar_norm": 1, "scope": "main", "print_solve_stats": 0, "obtain_timings": 0,
   "preconditioner": {
     "solver": "AMG", "algorithm": "AGGREGATION", %(aggregation)s,
     %(smoother)s, "max_iters": 1,
     "cycle": "V", "coarse_solver": "DENSE_LU_SOLVER",
     "min_coarse_rows": 32, "max_levels": 50, "structure_reuse_levels": %(reuse)d,
     "scope": "amg", "print_grid_stats": 0
   }
 }
}"""

# one V-cycle of the same hierarchy, as the preconditioner of zvCFD's FGMRES
AMGX_PRECOND_CONFIG = """{
 "config_version": 2,
 "determinism_flag": 1,
 "solver": {
   "solver": "AMG", "algorithm": "AGGREGATION", %(aggregation)s,
   %(smoother)s, "max_iters": 1,
   "cycle": "V", "coarse_solver": "DENSE_LU_SOLVER",
   "min_coarse_rows": 32, "max_levels": 50, "structure_reuse_levels": %(reuse)d,
   "monitor_residual": 0, "tolerance": 0.0, "scope": "main", "print_grid_stats": 0
 }
}"""


# AmgX configurations for the coupled 4 × 4 system, tried in this order by :class:`Auto`.
# Measured on four hard systems (quasi-2-D DFG slab, a slab with cells 10× thinner than wide,
# a tube with strongly graded wall layers, the gate B tube; benchmarks/fv/capture_hard_systems.py);
# iterations to 10⁻⁶ ("-": not converged in 300):
#   gs + SIZE_4 + pressure-weighted strength, smoothed coarsest:  17, -, 17, 26
#   ilu0 + pairwise (SIZE_2) + pressure-weighted strength:        11, 283, 9, 13
#   dilu + SIZE_4 + pressure-weighted strength, 0 / 1 sweeps:     -, -, 28, 35
#   dilu + SIZE_4 (AmgX's usual), 1 / 2 sweeps:                   -, -, 24, 24
#
# To the outer loop's 0.1 on the same four (benchmarks/results/fv/amgx_smoothers.json): gs 2,
# -, 3, 3; ILU(0) 1, 12, 2, 2; DILU 1/2 stalls on the DFG slab. ILU(0) needs fewer iterations
# on these small systems, gs sets up about twice as fast. On three
# systems of the SimVascular coronary model (398 k nodes, 2.1 M tetrahedra): gs 1–4 iterations,
# 0.18–0.31 s; ILU(0) 1–3, 0.32–0.42 s (its set-up costs more); DILU 1/2 13–53, 0.9–3.5 s.
# (The file's system_0/2 came from a defective crop and are kept for the record only.)
AMGX_PRESETS = {
    "gs": {"smoother": "gs", "selector": "SIZE_4", "strength": 3, "coarse": "smooth"},
    "robust": {"smoother": "ilu0", "selector": "SIZE_2", "strength": 3, "coarse": "dense"},
    "dilu-p": {"smoother": "dilu", "selector": "SIZE_4", "strength": 3, "coarse": "dense"},
    "dilu": {"smoother": "dilu", "selector": "SIZE_4", "strength": None, "coarse": "dense"},
}


def amgx_smoother(name: str) -> str:
    """The AMG smoother's configuration fragment: ``gs``, ``ilu0``, ``ilu1`` or ``dilu``.

    ``gs`` (the default) is symmetric multicolour block Gauss–Seidel, one
    forward and one backward sweep per smoothing step, damped by 0.9.

    Point-block smoothers of the coupled system (DILU, block Jacobi) only see
    a node's own 4 × 4 block, whose pressure entry is the small Rhie–Chow
    diagonal; on cells thin in one direction (prism layers, a quasi-2-D slab)
    they diverge. Multicolour ILU sees the couplings. ILU(1) is the most
    robust but its fill exceeds AmgX's per-row shared memory on tetrahedral
    meshes (about 20 blocks per row), so the ILU preset uses ILU(0). A
    symmetric Gauss–Seidel sweep over colours also sees them and costs less
    to set up than ILU; it is within one iteration of the best variant on
    the hard systems of :data:`AMGX_PRESETS`.
    """
    if name == "gs":
        return ('"smoother": "MULTICOLOR_GS", "symmetric_GS": 1, "presweeps": 1, '
                '"postsweeps": 2, "relaxation_factor": 0.9, '
                '"matrix_coloring_scheme": "PARALLEL_GREEDY", "max_uncolored_percentage": 0.05')
    if name == "dilu":
        return ('"smoother": "MULTICOLOR_DILU", "presweeps": 1, "postsweeps": 2, '
                '"relaxation_factor": 0.75, "matrix_coloring_scheme": "PARALLEL_GREEDY", '
                '"max_uncolored_percentage": 0.05')
    if name not in ("ilu0", "ilu1"):
        raise ValueError(f"smoother {name!r}: gs, ilu0, ilu1 or dilu")
    level = int(name[-1])
    return ('"smoother": {"scope": "ilu", "solver": "MULTICOLOR_ILU", '
            f'"ilu_sparsity_level": {level}, "reorder_cols_by_color": 1, '
            '"insert_diag_while_reordering": 1, "max_iters": 1, "relaxation_factor": 1.0, '
            f'"monitor_residual": 0, "print_solve_stats": 0, "coloring_level": {level + 1}, '
            '"matrix_coloring_scheme": "MIN_MAX"}, "presweeps": 1, "postsweeps": 1, '
            f'"coloring_level": {level + 1}, "matrix_coloring_scheme": "MIN_MAX"')


def amgx_aggregation(selector: str = "SIZE_2", strength: int | None = 3) -> str:
    """Aggregation fragment: the selector, and which block component measures coupling
    strength (3, the pressure–pressure entry, aggregates along the Rhie–Chow Laplacian,
    which follows the mesh's anisotropy; None, AmgX's default)."""
    frag = f'"selector": "{selector}"'
    if strength is not None:
        frag += f', "aggregation_edge_weight_component": {int(strength)}'
    return frag


_LIVE_AMGX: weakref.WeakSet | None = None
# AmgX resources, one per device, shared by every AMGX object: destroying a resources handle
# also destroys AmgX's process-wide cuBLAS/cuSPARSE handles and memory pools, so it is safe
# only when no other AMGX object is alive ("Cuda failure: invalid argument" otherwise)
_AMGX_RSC: dict = {}          # device -> resources handle
_AMGX_USERS = [0]


def _amgx_resources(lib, cfg):
    import cupy as cp

    dev = cp.cuda.Device().id
    if dev not in _AMGX_RSC:
        rsc = ctypes.c_void_p()
        rc = lib.AMGX_resources_create_simple(ctypes.byref(rsc), cfg)
        if rc != 0:
            raise RuntimeError(f"AmgX resources failed (code {rc})")
        _AMGX_RSC[dev] = rsc
    _AMGX_USERS[0] += 1
    return _AMGX_RSC[dev]


def _amgx_resources_release(lib) -> None:
    _AMGX_USERS[0] -= 1
    if _AMGX_USERS[0] == 0:
        import cupy as cp

        for dev, rsc in list(_AMGX_RSC.items()):
            with cp.cuda.Device(dev):
                lib.AMGX_resources_destroy(rsc)
            del _AMGX_RSC[dev]


def _live_amgx(obj) -> None:
    """Track ``obj`` so that it is closed at exit, while the CUDA context still exists
    (``__del__`` at interpreter teardown runs after it is gone, and AmgX aborts)."""
    global _LIVE_AMGX
    if _LIVE_AMGX is None:
        _LIVE_AMGX = weakref.WeakSet()
        atexit.register(_close_live_amgx)      # after CuPy's: runs before it (LIFO)
    _LIVE_AMGX.add(obj)


def _close_live_amgx() -> None:
    for obj in list(_LIVE_AMGX or ()):
        try:
            obj.close()
        except Exception:
            pass


def amgx_library():
    """Path of ``libamgxsh.so``: ``$ZVCFD_AMGX_LIB``, else None."""
    path = os.environ.get("ZVCFD_AMGX_LIB")
    return path if path and os.path.exists(path) else None


class AMGX:
    """NVIDIA AmgX through its C API (``libamgxsh.so``; set ``ZVCFD_AMGX_LIB``).

    Device pointers are passed straight to AmgX. The first solve uploads the
    structure; later ones replace the coefficients and re-set up the
    hierarchy. ``reuse`` is AmgX's ``structure_reuse_levels``: 0 (default)
    rebuilds the hierarchy for every matrix. Otherwise (``-1``: every level)
    later matrices keep the aggregates and colourings and recompute the
    coarse operators and smoothers (``AMGX_solver_resetup``; plain
    ``AMGX_solver_setup`` would keep the first matrix's whole hierarchy,
    which stalls FGMRES as soon as the linearisation moves away from it), and
    the hierarchy is rebuilt from scratch when a solve fails or needs more
    than ``rebuild_ratio`` times the iterations of the first solve after the
    last rebuild (and at least 2 more). Reuse is experimental: with AmgX 2.5
    it cuts set-up 3–20× and keeps the iterations on a random test matrix and
    the 22 k-node gate B tube, but on the 72 k-node gate B tube and the 9.5 k
    node pressure-driven pipe the solve after the first re-setup fails inside
    AmgX ("Matrix was not initialized"), with every reuse depth tried.

    AmgX's own FGMRES solves plain block matrices; its tolerance is part of
    the configuration, so each tolerance asked for gets its own solver
    handle. A matrix with rank-one terms is solved by zvCFD's
    :func:`fgmres`, with one AmgX V-cycle as the preconditioner.

    ``precision="mixed"`` keeps the hierarchy in single precision (AmgX mode
    dDFI: half the memory, and single-precision arithmetic in the V-cycle)
    and always runs zvCFD's double-precision FGMRES around it, so the
    operator and the residuals stay double: the solution is not limited by
    single precision, only the preconditioner is.
    """

    name = "amgx"
    _initialised = False

    def __init__(self, *, rtol: float = 0.1, maxiter: int = 200, reuse: int = 0,
                 rebuild_ratio: float = 1.5,
                 config: str | None = None, lib: str | None = None, restart: int = 30,
                 precision: str = "double", smoother: str = "gs", selector: str = "SIZE_4",
                 strength: int | None = 3, preset: str | None = None, coarse: str = "smooth"):
        import cupy as cp  # noqa: F401

        path = lib or amgx_library()
        if not path:
            raise RuntimeError("AmgX library not found: set ZVCFD_AMGX_LIB to libamgxsh.so")
        self.lib = ctypes.CDLL(path)
        L = self.lib
        vp = ctypes.c_void_p
        for name in ("AMGX_config_create", "AMGX_resources_create_simple", "AMGX_matrix_create",
                     "AMGX_vector_create", "AMGX_solver_create", "AMGX_matrix_upload_all",
                     "AMGX_matrix_replace_coefficients", "AMGX_vector_upload",
                     "AMGX_vector_download", "AMGX_solver_setup", "AMGX_solver_solve",
                     "AMGX_solver_solve_with_0_initial_guess",
                     "AMGX_solver_get_iterations_number", "AMGX_solver_get_status",
                     "AMGX_solver_resetup",
                     "AMGX_initialize", "AMGX_vector_set_zero"):
            getattr(L, name).restype = ctypes.c_int
        if not AMGX._initialised:
            self._check(L.AMGX_initialize(), "initialize")
            AMGX._initialised = True
        self.rtol, self.maxiter, self.reuse, self.restart = rtol, maxiter, reuse, restart
        self.rebuild_ratio = rebuild_ratio
        self._base_its = None           # iterations of the first solve after a full build
        self._rebuild = False
        self.rebuilds = 0
        self.config = config
        if preset is not None:
            p = AMGX_PRESETS[preset]
            smoother, selector, strength = p["smoother"], p["selector"], p["strength"]
            coarse = p["coarse"]
        if coarse not in ("dense", "smooth"):
            raise ValueError(f"coarse {coarse!r}: dense or smooth")
        self.coarse = coarse
        self.smoother = amgx_smoother(smoother)
        self.aggregation = amgx_aggregation(selector, strength)
        if precision not in ("double", "mixed"):
            raise ValueError(f"precision {precision!r}: double or mixed")
        self.precision = precision
        self.mtx, self.x, self.b = vp(), vp(), vp()
        self.rsc = _amgx_resources(L, self._config(self._text(rtol)))
        self._holds_rsc = True
        mode = ctypes.c_int(AMGX_MODE_DDFI if precision == "mixed" else AMGX_MODE_DDDI)
        self._mode = mode
        self._check(L.AMGX_matrix_create(ctypes.byref(self.mtx), self.rsc, mode), "matrix")
        self._check(L.AMGX_vector_create(ctypes.byref(self.x), self.rsc, mode), "vector x")
        self._check(L.AMGX_vector_create(ctypes.byref(self.b), self.rsc, mode), "vector b")
        self._solvers: dict = {}          # key -> [handle, set up for matrix version]
        _live_amgx(self)
        self._shape = None
        self._version = 0

    @staticmethod
    def _check(rc, what):
        if rc != 0:
            raise RuntimeError(f"AmgX {what} failed (code {rc})")

    def _text(self, rtol):
        if self.config is not None:
            return self.config
        return DEFAULT_AMGX_CONFIG % {"rtol": rtol, "maxiter": self.maxiter, "reuse": self.reuse,
                                      "smoother": self.smoother,
                                      "aggregation": self.aggregation}

    def _config(self, text):
        cfg = ctypes.c_void_p()
        self._check(self.lib.AMGX_config_create(ctypes.byref(cfg), text.encode()), "config")
        return cfg

    def _solver(self, key):
        """A solver handle (``key`` a tolerance, or ``"precond"``), set up for the matrix."""
        L = self.lib
        if key not in self._solvers:
            if key != "precond":
                text = self._text(key)
            elif self.config is not None:
                # one V-cycle of the given configuration's preconditioner
                cfg = json.loads(self.config)
                pre = dict(cfg["solver"]["preconditioner"])
                pre.update({"max_iters": 1, "monitor_residual": 0, "tolerance": 0.0,
                            "scope": "main"})
                text = json.dumps({"config_version": 2,
                                   "determinism_flag": cfg.get("determinism_flag", 1),
                                   "solver": pre})
            else:
                text = AMGX_PRECOND_CONFIG % {"reuse": self.reuse, "smoother": self.smoother,
                                              "aggregation": self.aggregation}
            if self.precision == "mixed" or self.coarse == "smooth":
                # AmgX's dense LU coarse solver fails in dDFI mode ("Fail to get info from
                # cudense"), and wherever the coarsest level stays large or singular (rows of
                # Dirichlet nodes are isolated, so aggregation cannot merge them): smooth the
                # coarsest level instead
                text = text.replace('"coarse_solver": "DENSE_LU_SOLVER"',
                                    '"coarse_solver": "MULTICOLOR_DILU", "coarsest_sweeps": 4')
            slv = ctypes.c_void_p()
            self._check(L.AMGX_solver_create(ctypes.byref(slv), self.rsc, self._mode,
                                             self._config(text)), "solver")
            self._solvers[key] = [slv, -1]
        entry = self._solvers[key]
        if entry[1] != self._version:
            if entry[1] >= 0 and self.reuse != 0 and not self._rebuild:
                # keep the aggregates and colourings, recompute the coarse operators and
                # smoothers from the new coefficients (AMGX_solver_setup would keep the
                # first matrix's whole hierarchy)
                self._check(L.AMGX_solver_resetup(entry[0], self.mtx), "resetup")
            else:
                if entry[1] >= 0 and self.reuse != 0:        # a fresh handle: a full build
                    self.release(key)
                    self.rebuilds += 1
                    return self._solver(key)
                self._check(L.AMGX_solver_setup(entry[0], self.mtx), "setup")
                self._base_its, self._rebuild = None, False
            entry[1] = self._version
        return entry[0]

    def _after_solve(self, its: int, ok: bool) -> None:
        """The rebuild signal of a reused hierarchy (see the class notes)."""
        if self.reuse == 0:
            return
        if self._base_its is None:
            self._base_its = its
        elif not ok or its > max(self.rebuild_ratio * self._base_its, self._base_its + 2):
            self._rebuild = True

    def _upload(self, A: BlockMatrix):
        self.upload(A.indptr, A.indices, A.data)

    def upload(self, indptr, indices, data):
        """Upload a block-CSR matrix of any block size (``data`` ``(nnzb, b, b)``, or ``(nnz,)``
        for scalars): structure on the first call, coefficients after."""
        import cupy as cp

        L = self.lib
        n, nnz = int(indptr.size) - 1, int(indices.size)
        block = 1 if data.ndim == 1 else int(data.shape[1])
        data = cp.ascontiguousarray(data.reshape(-1),
                                    dtype=cp.float32 if self.precision == "mixed" else None)
        if self._shape != (n, nnz, block):
            self._rp = indptr.astype(cp.int32)
            self._ci = cp.ascontiguousarray(indices.astype(cp.int32))
            self._check(L.AMGX_matrix_upload_all(self.mtx, n, nnz, block, block,
                                                 ctypes.c_void_p(self._rp.data.ptr),
                                                 ctypes.c_void_p(self._ci.data.ptr),
                                                 ctypes.c_void_p(data.data.ptr), None), "upload")
            self._shape = (n, nnz, block)
        else:
            self._check(L.AMGX_matrix_replace_coefficients(self.mtx, n, nnz,
                                                           ctypes.c_void_p(data.data.ptr), None),
                        "replace")
        self._version += 1

    def close(self) -> None:
        """Destroy this object's AmgX handles (solvers, vectors, matrix; the shared resources
        with the last object); idempotent.

        AmgX's memory lives outside CuPy's pool, so an AMGX object that is not
        closed keeps its hierarchy on the device until the process ends.
        """
        L = getattr(self, "lib", None)
        if L is None or getattr(self, "_closed", False):
            return
        for key in list(getattr(self, "_solvers", {})):
            self.release(key)
        for h, f in ((getattr(self, "x", None), "AMGX_vector_destroy"),
                     (getattr(self, "b", None), "AMGX_vector_destroy"),
                     (getattr(self, "mtx", None), "AMGX_matrix_destroy")):
            if h is not None and h.value:
                getattr(L, f)(h)
        if getattr(self, "_holds_rsc", False):
            _amgx_resources_release(L)
            self._holds_rsc = False
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def release(self, key) -> None:
        """Destroy the solver handle for ``key`` (a tolerance, or ``"precond"``), and its
        hierarchy."""
        entry = self._solvers.pop(key, None)
        if entry is not None:
            self.lib.AMGX_solver_destroy(entry[0])

    def vcycle(self, r):
        """One V-cycle of the current hierarchy applied to ``r`` (zero initial guess)."""
        import cupy as cp

        L = self.lib
        n, _, block = self._shape
        slv = self._solver("precond")
        rr = cp.ascontiguousarray(r.reshape(-1))
        z = cp.zeros_like(rr)
        self._check(L.AMGX_vector_upload(self.b, n, block, ctypes.c_void_p(rr.data.ptr)), "b")
        self._check(L.AMGX_vector_upload(self.x, n, block, ctypes.c_void_p(z.data.ptr)), "x")
        self._check(L.AMGX_solver_solve_with_0_initial_guess(slv, self.b, self.x), "vcycle")
        self._check(L.AMGX_vector_download(self.x, ctypes.c_void_p(z.data.ptr)), "download")
        return z

    def solve(self, A: BlockMatrix, b, x0=None, rtol=None):
        import cupy as cp

        L = self.lib
        n = A.n
        rtol = self.rtol if rtol is None else rtol
        t = time.time()
        self._upload(A)
        bb = cp.ascontiguousarray(b.reshape(-1))
        x = cp.zeros(n * 4) if x0 is None else cp.ascontiguousarray(x0.reshape(-1).copy())
        if A.lowrank or self.precision == "mixed":
            self._solver("precond")
            ts = time.time() - t
            t = time.time()
            x, it, res, ok = fgmres(A, bb, x, self.vcycle, rtol=rtol, restart=self.restart,
                                    maxiter=self.maxiter)
            cp.cuda.Device().synchronize()
            self._after_solve(it, ok)
            return x, SolveInfo(it, res, time.time() - t, ts, ok)
        slv = self._solver(rtol)
        cp.cuda.Device().synchronize()
        ts = time.time() - t
        r0 = float(cp.linalg.norm(bb - A.matvec(x))) or 1.0
        self._check(L.AMGX_vector_upload(self.b, n, 4, ctypes.c_void_p(bb.data.ptr)), "b")
        self._check(L.AMGX_vector_upload(self.x, n, 4, ctypes.c_void_p(x.data.ptr)), "x")
        t = time.time()
        self._check(L.AMGX_solver_solve(slv, self.b, self.x), "solve")
        self._check(L.AMGX_vector_download(self.x, ctypes.c_void_p(x.data.ptr)), "download")
        cp.cuda.Device().synchronize()
        it, st = ctypes.c_int(), ctypes.c_int()
        L.AMGX_solver_get_iterations_number(slv, ctypes.byref(it))
        L.AMGX_solver_get_status(slv, ctypes.byref(st))
        r = float(cp.linalg.norm(bb - A.matvec(x)))
        ok = st.value == 0 or r <= ROUNDOFF * float(cp.linalg.norm(bb))
        self._after_solve(int(it.value), ok)
        return x, SolveInfo(int(it.value), r / r0, time.time() - t, ts, ok)


class SIMPLE:
    """FGMRES on the coupled system, preconditioned block-wise (SIMPLE type).

    With the coupled matrix split by variable, ``A = [[F, Bᵀ], [B, C]]`` (``F``
    momentum, ``Bᵀ`` pressure gradient, ``B`` the divergence with its Rhie–Chow
    part, ``C`` the Rhie–Chow pressure diagonal block), one application is

    1. ``u* = F̃⁻¹ r_u``: an AMG V-cycle on ``F`` (3 × 3 blocks);
    2. ``p = S̃⁻¹ (r_p − B u*)``: a V-cycle on the pressure Schur complement
       approximated as ``S = C − B diag(F)⁻¹ Bᵀ`` (scalar, formed on the device);
    3. ``u = u* − diag(F)⁻¹ Bᵀ p``.

    FGMRES works on the exact coupled operator (rank-one terms included), so
    only the preconditioner is approximate. Point-block smoothers of the
    coupled 4 × 4 system only see a node's small Rhie–Chow pressure diagonal
    and fail on cells thin in one direction (prism layers, a quasi-2-D
    slab); here each AMG works on an operator of one kind (momentum,
    Laplacian-like pressure), which aggregation AMG handles with moderate
    anisotropy. Both hierarchies are AmgX's (``precision``, ``smoother``).

    References: S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*
    (1980) (SIMPLE); H. Elman, D. Silvester, A. Wathen, *Finite Elements and
    Fast Iterative Solvers*, 2nd ed., OUP (2014), ch. 9 (block preconditioners).
    """

    name = "simple"

    def __init__(self, *, rtol: float = 0.1, maxiter: int = 200, restart: int = 30,
                 precision: str = "double", smoother: str = "dilu", lib: str | None = None,
                 schur: str = "exact"):
        self.rtol, self.maxiter, self.restart = rtol, maxiter, restart
        if schur not in ("exact", "exact-host", "compact", "compact2"):
            raise ValueError(f"schur {schur!r}")
        self.schur = schur
        self.F = AMGX(precision=precision, smoother=smoother, selector="SIZE_4", strength=None,
                      lib=lib, coarse="smooth")
        self.S = AMGX(precision=precision, smoother=smoother, selector="SIZE_4", strength=None,
                      lib=lib, coarse="smooth")

    def setup(self, A: BlockMatrix) -> float:
        import cupy as cp
        import cupyx.scipy.sparse as csp

        t = time.time()
        n = A.n
        rows = A.row_of_block()
        cols = A.indices.astype(cp.int64)
        D = A.data
        k3 = cp.arange(3)
        self.dinv = 1.0 / D[A.diagonal_positions()][:, k3, k3]          # (n, 3)
        r3 = (3 * rows[:, None] + k3).reshape(-1)
        c3 = (3 * cols[:, None] + k3).reshape(-1)
        self.Bt = csp.coo_matrix((D[:, :3, 3].reshape(-1), (r3, cp.repeat(cols, 3))),
                                 shape=(3 * n, n)).tocsr()
        self.Bp = csp.coo_matrix((D[:, 3, :3].reshape(-1), (cp.repeat(rows, 3), c3)),
                                 shape=(n, 3 * n)).tocsr()
        C = csp.coo_matrix((D[:, 3, 3], (rows, cols)), shape=(n, n)).tocsr()
        if self.schur == "exact":
            S = (C - self.Bp @ csp.diags(self.dinv.reshape(-1)) @ self.Bt).tocsr()
        elif self.schur == "exact-host":
            import scipy.sparse as sps

            h = [a.get() for a in (self.Bp, self.Bt, C)]
            Sh = (h[2] - h[0] @ sps.diags(self.dinv.reshape(-1).get()) @ h[1]).tocsr()
            S = csp.csr_matrix(Sh)
        else:
            S = C * (2.0 if self.schur == "compact2" else 1.0)
        S.sum_duplicates()
        S.sort_indices()
        self.F.upload(A.indptr, A.indices, cp.ascontiguousarray(D[:, :3, :3]))
        self.F._solver("precond")
        self.S.upload(S.indptr.astype(cp.int64), S.indices, S.data)
        self.S._solver("precond")
        cp.cuda.Device().synchronize()
        return time.time() - t

    def close(self) -> None:
        self.F.close()
        self.S.close()

    def vcycle(self, r):
        import cupy as cp

        R = r.reshape(-1, 4)
        us = self.F.vcycle(cp.ascontiguousarray(R[:, :3]).reshape(-1))
        p = self.S.vcycle(R[:, 3] - self.Bp @ us)
        u = us - self.dinv.reshape(-1) * (self.Bt @ p)
        return cp.concatenate([u.reshape(-1, 3), p[:, None]], 1).reshape(-1)

    def solve(self, A: BlockMatrix, b, x0=None, rtol=None):
        import cupy as cp

        ts = self.setup(A)
        t = time.time()
        x0 = cp.zeros(A.n * 4) if x0 is None else x0.reshape(-1).copy()
        x, it, res, ok = fgmres(A, b.reshape(-1), x0, self.vcycle,
                                rtol=self.rtol if rtol is None else rtol,
                                restart=self.restart, maxiter=self.maxiter)
        cp.cuda.Device().synchronize()
        return x, SolveInfo(it, res, time.time() - t, ts, ok)


class Auto:
    """A ladder of linear solvers: the first that works on a system is kept.

    Tries, in order, AmgX with the ``gs`` preset (the default), ``robust``, ``dilu-p``, and
    :class:`SIMPLE` (then zvCFD's ACM if AmgX is missing). A solve that
    reduces the residual by less than ``fail_ratio × rtol`` at the iteration
    limit is a failure: the same system is solved again with the next
    solver, which is then kept for later systems (logged in ``switches``).
    A near miss (less than that) is accepted as it is; the outer loop's
    convergence test still sees it as unconverged.

    Which of ``gs`` and ``robust`` is faster depends on the mesh: ``robust``
    needs fewer iterations on small systems of good quality (a steady tube:
    191 ms against 229 ms per linear solve), ``gs`` sets up about twice as
    fast and wins on the coronary systems (0.18–0.31 s against 0.32–0.42 s).
    So on the ``probe_at``-th
    system, the same system is also solved with ``robust``, and ``robust`` is
    kept if it converges in under ``probe_margin`` of the time ``gs`` took
    (logged in ``switches``). If it later fails, the ladder returns to
    ``gs``.
    """

    name = "auto"

    def __init__(self, *, rtol: float = 0.1, maxiter: int = 200, precision: str = "double",
                 fail_ratio: float = 10.0, lib: str | None = None, probe_at: int | None = 3,
                 probe_margin: float = 0.8):
        self.rtol, self.maxiter, self.fail_ratio = rtol, maxiter, fail_ratio
        self.probe_at, self.probe_margin = probe_at, probe_margin
        self.solves = 0
        self.probe = None               # the probe's timings, once made
        self._probed_from = None        # the rung to return to if the probed one fails
        if amgx_library() or lib:
            self.ladder = [
                ("amgx-gs", lambda: AMGX(rtol=rtol, maxiter=maxiter, precision=precision,
                                         preset="gs", lib=lib)),
                ("amgx-robust", lambda: AMGX(rtol=rtol, maxiter=maxiter, precision=precision,
                                             preset="robust", lib=lib)),
                ("amgx-dilu-p", lambda: AMGX(rtol=rtol, maxiter=maxiter, precision=precision,
                                             preset="dilu-p", lib=lib)),
                ("simple", lambda: SIMPLE(rtol=rtol, maxiter=maxiter, lib=lib))]
        else:
            self.ladder = [("acm", lambda: ACM(rtol=rtol, maxiter=maxiter))]
        self.index = 0
        self.current = self.ladder[0][1]()
        self.switches = []

    @property
    def active(self) -> str:
        return self.ladder[self.index][0]

    def close(self) -> None:
        if hasattr(self.current, "close"):
            self.current.close()

    def _switch(self, index: int, why: str, solver=None) -> None:
        self.switches.append((self.active, self.ladder[index][0], why))
        self.close()
        self.index = index
        self.current = solver if solver is not None else self.ladder[index][1]()

    def _maybe_probe(self, A, b, x0, rtol, x, info, seconds):
        names = [n for n, _ in self.ladder]
        if (self.probe_at is None or self.solves != self.probe_at or self.active != "amgx-gs"
                or "amgx-robust" not in names or not info.converged):
            return x, info
        k = names.index("amgx-robust")
        alt = self.ladder[k][1]()
        t = time.time()
        try:
            xa, ia = alt.solve(A, b, x0, rtol=rtol)
            ta = time.time() - t
        except RuntimeError:
            xa, ia, ta = None, None, float("inf")
        self.probe = {"gs_s": seconds, "robust_s": ta,
                      "robust_converged": bool(ia is not None and ia.converged)}
        if ia is not None and ia.converged and ta < self.probe_margin * seconds:
            self._probed_from = self.index
            self._switch(k, f"probe: {ta:.2f} s against {seconds:.2f} s", alt)
            return xa, ia
        if hasattr(alt, "close"):
            alt.close()
        return x, info

    def solve(self, A: BlockMatrix, b, x0=None, rtol=None):
        rtol = self.rtol if rtol is None else rtol
        self.solves += 1
        while True:
            t = time.time()
            try:
                x, info = self.current.solve(A, b, x0, rtol=rtol)
                failed = not info.converged and info.residual > self.fail_ratio * rtol
            except RuntimeError as exc:           # e.g. an AmgX setup limit on this mesh
                x, info, failed = None, None, True
                why = str(exc)
            else:
                why = f"residual {info.residual:.1e} after {info.iterations} iterations"
            if not failed:
                return self._maybe_probe(A, b, x0, rtol, x, info, time.time() - t)
            if self._probed_from is not None:          # the probed rung failed: go back
                back, self._probed_from = self._probed_from, None
                self._switch(back, "probed solver failed: " + why)
                continue
            if self.index + 1 >= len(self.ladder):
                if x is None:
                    raise RuntimeError(f"every linear solver failed ({why})")
                return x, info
            self._switch(self.index + 1, why)


def make(name: str, **kw):
    """A linear solver by name: ``"host-direct"``, ``"acm"``, ``"amgx"``, ``"simple"`` or
    ``"auto"`` (the :class:`Auto` ladder)."""
    return {"host-direct": HostDirect, "acm": ACM, "amgx": AMGX, "simple": SIMPLE,
            "auto": Auto}[name](**kw) if name != "host-direct" else HostDirect()


__all__ = ["ACM", "AMGX", "AMGX_PRESETS", "SIMPLE", "Auto", "BlockMatrix", "HostDirect",
           "SolveInfo", "amgx_aggregation", "amgx_library", "amgx_smoother", "fgmres", "make"]
