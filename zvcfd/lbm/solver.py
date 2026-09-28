"""D3Q19 lattice-Boltzmann solvers on one GPU: dense boxes and sparse brick domains.

Both keep two population buffers and swap them each step. Units are
lattice units throughout; :mod:`zvcfd.units` converts to physical ones.
Multi-GPU runs compose :class:`SparseLBM` instances, one per device, with
halo exchange between them (``docs/architecture.md``); that layer is not
built yet.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from zvcfd.domain import BrickDomain
from zvcfd.lbm._cuda import CX, CY, CZ, SOURCES, Q, W

_kernels: dict[tuple[str, bool], cp.RawKernel] = {}


def kernel(name: str, half: bool) -> cp.RawKernel:
    """The compiled kernel ``name`` for fp16 (``half``) or fp32 population storage."""
    key = (name, half)
    if key not in _kernels:
        _kernels[key] = cp.RawKernel(SOURCES[name], name,
                                     options=(f"-DSTORE_HALF={int(half)}", "-std=c++14"))
    return _kernels[key]


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy=None, uz=None) -> np.ndarray:
    """Equilibrium populations ``(Q, *rho.shape)`` (float32, host)."""
    uy = np.zeros_like(ux) if uy is None else uy
    uz = np.zeros_like(ux) if uz is None else uz
    usq = 1.5 * (ux * ux + uy * uy + uz * uz)
    out = np.empty((Q,) + np.shape(rho), np.float32)
    for q in range(Q):
        cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz
        out[q] = W[q] * rho * (1 + 3 * cu + 4.5 * cu * cu - usq)
    return out


class _LBM:
    """Shared state: two population buffers, flags, relaxation and forcing."""

    def __init__(self, n: int, flags: np.ndarray, *, tau: float, force, rho_in: float,
                 rho_out: float, half: bool, init: np.ndarray | None):
        if Q * n >= 2 ** 31:
            raise ValueError(f"{n} cells exceed one 32-bit-indexed buffer; partition the domain")
        self.n = n
        self.half = half
        self.dtype = cp.float16 if half else cp.float32
        self.flag = cp.asarray(np.ascontiguousarray(flags).reshape(-1), dtype=cp.uint8)
        self.tau = float(tau)
        self.omega = np.float32(1.0 / tau)
        self.force = tuple(np.float32(v) for v in force)
        self.rho_in, self.rho_out = np.float32(rho_in), np.float32(rho_out)
        if init is None:
            f = np.repeat(np.asarray(W, np.float32)[:, None], n, axis=1)
        else:
            f = np.asarray(init, np.float32).reshape(Q, n)
        if half:
            f = f - np.asarray(W, np.float32)[:, None]
        self.f0 = cp.asarray(f.reshape(-1), dtype=self.dtype)
        self.f1 = self.f0.copy()
        self.steps = 0

    @property
    def nu(self) -> float:
        """Kinematic viscosity in lattice units."""
        return (self.tau - 0.5) / 3.0

    @property
    def bytes_per_update(self) -> int:
        """Bytes one fluid-cell update must move: read and write every population, plus its flag."""
        return 2 * Q * (2 if self.half else 4) + 1

    @property
    def device_bytes(self) -> int:
        return 2 * self.f0.nbytes + self.flag.nbytes

    def fields(self) -> dict[str, cp.ndarray]:
        """Density and velocity for every stored cell, as flat device arrays."""
        out = {k: cp.empty(self.n, cp.float32) for k in ("rho", "ux", "uy", "uz")}
        threads = 256
        kernel("macros", self.half)(((self.n + threads - 1) // threads,), (threads,),
                                    (self.f0, self.flag, np.int32(self.n), *self.force,
                                     out["rho"], out["ux"], out["uy"], out["uz"]))
        return out


class DenseLBM(_LBM):
    """Dense periodic box, flags ``(z, y, x)``; reservoir cells take ``rho_in`` for x < nx/2."""

    def __init__(self, flags: np.ndarray, *, tau=1.0, force=(0.0, 0.0, 0.0), rho_in=1.0,
                 rho_out=1.0, half=False, init=None):
        self.nz, self.ny, self.nx = flags.shape
        super().__init__(flags.size, flags, tau=tau, force=force, rho_in=rho_in,
                         rho_out=rho_out, half=half, init=init)
        self.shape = flags.shape
        tx = 128 if self.nx % 128 == 0 else 64
        self._grid = ((self.nx + tx - 1) // tx, self.ny, self.nz)
        self._block = (tx,)
        self._k = kernel("step_dense", half)

    def step(self, k: int = 1) -> None:
        for _ in range(k):
            self._k(self._grid, self._block,
                    (self.f0, self.f1, self.flag, np.int32(self.nx), np.int32(self.ny),
                     np.int32(self.nz), self.omega, *self.force, self.rho_in, self.rho_out))
            self.f0, self.f1 = self.f1, self.f0
            self.steps += 1

    def fields(self) -> dict[str, cp.ndarray]:
        return {k: v.reshape(self.shape) for k, v in super().fields().items()}


class SparseLBM(_LBM):
    """Active 8^3 bricks of a :class:`~zvcfd.domain.BrickDomain` only.

    One CUDA block per brick and one thread per voxel; a missing neighbour
    brick reads as solid. :meth:`fields` returns ``(n_bricks, 512)`` arrays,
    the row layout of a Zarr Vectors brick store (:mod:`zvcfd.io.fields`).
    """

    def __init__(self, domain: BrickDomain, *, tau=1.0, force=(0.0, 0.0, 0.0), rho_in=1.0,
                 rho_out=1.0, half=False, init=None):
        if domain.brick != 8:
            raise ValueError("the sparse kernel is compiled for 8^3 bricks")
        self.domain = domain
        self.nb = domain.n_bricks
        super().__init__(domain.stored_cells, domain.flags, tau=tau, force=force,
                         rho_in=rho_in, rho_out=rho_out, half=half, init=init)
        self.nbr = cp.asarray(domain.neighbours.reshape(-1))
        self.brick_x = cp.asarray(np.ascontiguousarray(domain.coords[:, 2]).astype(np.int32))
        self.nx_cells = np.int32(domain.shape[2])
        self._k = kernel("step_sparse", half)

    @property
    def fluid_cells(self) -> int:
        return self.domain.fluid_cells

    @property
    def device_bytes(self) -> int:
        return super().device_bytes + self.nbr.nbytes

    def step(self, k: int = 1) -> None:
        for _ in range(k):
            self._k((self.nb,), (512,),
                    (self.f0, self.f1, self.flag, self.nbr, self.brick_x, np.int32(self.nb),
                     self.nx_cells, self.omega, *self.force, self.rho_in, self.rho_out))
            self.f0, self.f1 = self.f1, self.f0
            self.steps += 1

    def fields(self) -> dict[str, cp.ndarray]:
        return {k: v.reshape(self.nb, 512) for k, v in super().fields().items()}


def dense_from_bricks(domain: BrickDomain, values: np.ndarray, fill=0.0) -> np.ndarray:
    """Scatter brick rows into a dense volume; see :meth:`BrickDomain.to_dense`."""
    return domain.to_dense(values, fill)
