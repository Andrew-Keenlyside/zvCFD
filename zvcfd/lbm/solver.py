"""D3Q19 lattice-Boltzmann solvers on one GPU: dense boxes and sparse brick domains.

Both keep two population buffers and swap them each step. Units are
lattice units throughout; :mod:`zvcfd.units` converts to physical ones.

:class:`SparseLBM` is the production path: BGK or TRT collision, optional
Carreau–Yasuda rheology, inlet/outlet patches (:mod:`zvcfd.boundary`),
and partition-local domains with ghost bricks, which
:class:`zvcfd.lbm.multi.MultiLBM` composes across GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np

from zvcfd.boundary import BoundarySet
from zvcfd.domain import BrickDomain
from zvcfd.lbm._cuda import CX, CY, CZ, SOURCES, Q, W

_kernels: dict[tuple, cp.RawKernel] = {}


def kernel(name: str, half: bool, *, trt: bool = False, nonnewtonian: bool = False,
           device: int | None = None) -> cp.RawKernel:
    """The compiled kernel ``name`` for the given storage precision and model options."""
    dev = cp.cuda.Device().id if device is None else device
    key = (name, half, trt, nonnewtonian, dev)
    if key not in _kernels:
        opts = (f"-DSTORE_HALF={int(half)}", f"-DCOLLISION_TRT={int(trt)}",
                f"-DNONNEWTONIAN={int(nonnewtonian)}", "-std=c++14")
        _kernels[key] = cp.RawKernel(SOURCES[name], name, options=opts)
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


@dataclass(frozen=True)
class CarreauYasuda:
    """Carreau–Yasuda viscosity, lattice units:
    ``nu = nu_inf + (nu_0 - nu_inf) [1 + (lam * gamma_dot)^a]^((n - 1)/a)``.

    Build it from SI parameters with :meth:`from_si`. Blood (Cho & Kensey
    1991): mu_0 = 0.056 Pa s, mu_inf = 0.00345 Pa s, lambda = 3.313 s,
    n = 0.3568, a = 2.
    """

    nu_0: float
    nu_inf: float
    lam: float
    a: float = 2.0
    n: float = 0.3568

    @classmethod
    def from_si(cls, lattice, *, mu_0=0.056, mu_inf=0.00345, lam=3.313, a=2.0, n=0.3568):
        to_nu = lattice.dt / lattice.dx ** 2 / lattice.rho
        return cls(nu_0=mu_0 * to_nu, nu_inf=mu_inf * to_nu, lam=lam / lattice.dt, a=a, n=n)


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
        self.rho_in, self.rho_out = float(rho_in), float(rho_out)
        # kernels take density deviations and hold shifted populations g = f - w
        self._drho = (np.float32(self.rho_in - 1.0), np.float32(self.rho_out - 1.0))
        if init is None:
            g = np.zeros((Q, n), np.float32)
        else:
            g = (np.asarray(init, np.float64).reshape(Q, n)
                 - np.asarray(W, np.float64)[:, None]).astype(np.float32)
        self.f0 = cp.asarray(g.reshape(-1), dtype=self.dtype)
        self.f1 = self.f0.copy()
        self.steps = 0

    @property
    def nu(self) -> float:
        """Kinematic viscosity in lattice units (Newtonian / zero-shear)."""
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
    """Dense periodic box, flags ``(z, y, x)``; reservoir cells take ``rho_in`` for x < nx/2.

    BGK only; kept for benchmarks and as the reference the sparse kernel is
    checked against.
    """

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
                     np.int32(self.nz), self.omega, *self.force, *self._drho))
            self.f0, self.f1 = self.f1, self.f0
            self.steps += 1

    def fields(self) -> dict[str, cp.ndarray]:
        return {k: v.reshape(self.shape) for k, v in super().fields().items()}


class SparseLBM(_LBM):
    """Active 8³ bricks of a :class:`~zvcfd.domain.BrickDomain` only.

    One CUDA block per brick and one thread per voxel; a missing neighbour
    brick reads as solid. :meth:`fields` returns ``(n_bricks, 512)`` arrays,
    the row layout of a Zarr Vectors brick store (:mod:`zvcfd.io.fields`).

    Args:
        domain: The domain, or a partition-local domain (owned bricks first,
            then ghosts); only owned bricks are updated.
        tau: Relaxation time (zero-shear value with ``rheology``).
        collision: ``"bgk"`` or ``"trt"`` (magic parameter ``magic``, default
            3/16, which puts half-way bounce-back walls exactly mid-link).
        rheology: optional :class:`CarreauYasuda` (lattice units).
        boundary: optional :class:`~zvcfd.boundary.BoundarySet` in the
            domain's cell numbering; its flags must already be applied.
    """

    def __init__(self, domain: BrickDomain, *, tau=1.0, force=(0.0, 0.0, 0.0), rho_in=1.0,
                 rho_out=1.0, half=False, init=None, collision: str = "bgk",
                 magic: float = 3.0 / 16.0, rheology: CarreauYasuda | None = None,
                 boundary: BoundarySet | None = None):
        if domain.brick != 8:
            raise ValueError("the sparse kernel is compiled for 8^3 bricks")
        if collision not in ("bgk", "trt"):
            raise ValueError(f"collision {collision!r}")
        self.domain = domain
        self.nb = domain.n_bricks
        self.n_owned = domain.n_owned
        super().__init__(domain.stored_cells, domain.flags, tau=tau, force=force,
                         rho_in=rho_in, rho_out=rho_out, half=half, init=init)
        self.nbr = cp.asarray(domain.neighbours.reshape(-1))
        self.brick_x = cp.asarray(np.ascontiguousarray(domain.coords[:, 2]).astype(np.int32))
        self.nx_cells = np.int32(domain.shape[2])
        self.collision = collision
        self.rheology = rheology
        r = rheology
        self._params = tuple(np.float32(v) for v in (
            1.0 / tau, magic, r.nu_inf if r else 0.0, r.nu_0 if r else 0.0,
            r.lam if r else 0.0, r.a if r else 2.0, r.n if r else 1.0))
        self._k = kernel("step_sparse", half, trt=collision == "trt",
                         nonnewtonian=rheology is not None)
        self.boundary = None
        if boundary is not None and len(boundary):
            self._set_boundary(boundary)

    # ------------------------------------------------------------ boundary

    def _set_boundary(self, b: BoundarySet) -> None:
        self.boundary = b
        self.bidx = cp.asarray(b.cells.astype(np.int32))
        self.bnbr = cp.asarray(b.nbr.astype(np.int32))
        self.bpid = cp.asarray(b.patch.astype(np.int32))
        self.bscale = cp.asarray(b.scale.astype(np.float32))
        n = max(len(b.patches), 1)
        self.pdrho = cp.zeros(n, cp.float32)         # patch density deviations rho - 1
        self.pu = cp.zeros((3, n), cp.float32)       # rows: ux, uy, uz (lattice)
        self._kb = kernel("boundary_neem", self.half)

    def set_patch(self, index: int, *, rho: float | None = None, u_zyx=None) -> None:
        """Set a patch's lattice density and/or velocity vector (z, y, x)."""
        if rho is not None:
            self.pdrho[index] = float(rho) - 1.0
        if u_zyx is not None:
            uz, uy, ux = (float(v) for v in u_zyx)
            self.pu[:, index] = cp.asarray([ux, uy, uz], cp.float32)

    def apply_boundary(self) -> None:
        """Write the patch cells of the newest buffer (after a step and its swap)."""
        if self.boundary is None or len(self.bidx) == 0:
            return
        nb = len(self.bidx)
        threads = 128
        self._kb(((nb + threads - 1) // threads,), (threads,),
                 (self.f0, np.int32(self.n), self.flag, self.bidx, self.bnbr, self.bpid,
                  self.bscale, np.int32(nb), self.pdrho, self.pu[0], self.pu[1], self.pu[2]))

    # ------------------------------------------------------------ stepping

    @property
    def fluid_cells(self) -> int:
        return self.domain.owned_fluid_cells

    @property
    def device_bytes(self) -> int:
        return super().device_bytes + self.nbr.nbytes

    def step_main(self) -> None:
        """Collide and stream the owned bricks, then swap buffers (no boundary pass)."""
        if self.n_owned:
            self._k((self.n_owned,), (512,),
                    (self.f0, self.f1, self.flag, self.nbr, self.brick_x, np.int32(self.nb),
                     self.nx_cells, *self._params, *self.force, *self._drho))
        self.f0, self.f1 = self.f1, self.f0
        self.steps += 1

    def step(self, k: int = 1) -> None:
        for _ in range(k):
            self.step_main()
            self.apply_boundary()

    def fields(self) -> dict[str, cp.ndarray]:
        return {k: v.reshape(self.nb, 512) for k, v in super().fields().items()}

    def patch_flux(self) -> np.ndarray:
        """Lattice mass flux through each patch per step (positive into the domain).

        Exact for the lattice: counts the populations crossing between patch
        cells and the fluid in the newest buffer.
        """
        if self.boundary is None:
            return np.zeros(0)
        nbnd = len(self.bidx)
        n = len(self.boundary.patches)
        if nbnd == 0:
            return np.zeros(n)
        per = cp.empty(nbnd, cp.float32)
        threads = 128
        kernel("patch_flux", self.half)(((nbnd + threads - 1) // threads,), (threads,),
                                        (self.f0, np.int32(self.n), self.flag, self.nbr,
                                         self.bidx, np.int32(nbnd), per))
        # summed on the host in a fixed order: GPU atomics add in no fixed order, and
        # flow control feeds these sums back into the run (reruns must be bit-identical)
        return np.bincount(self.boundary.patch, weights=per.get().astype(np.float64), minlength=n)

    def patch_velocity(self) -> np.ndarray:
        """Per patch: summed velocity (z, y, x), summed density, and cell count, ``(P, 5)``."""
        if self.boundary is None:
            return np.zeros((0, 5))
        f = super().fields()
        idx = self.bidx
        pid = self.boundary.patch
        n = len(self.boundary.patches)
        out = np.zeros((n, 5), np.float64)
        for c, key in enumerate(("uz", "uy", "ux", "rho")):          # fixed-order host sums
            out[:, c] = np.bincount(pid, weights=f[key][idx].get().astype(np.float64), minlength=n)
        out[:, 4] = np.bincount(pid, minlength=n)
        return out


def dense_from_bricks(domain: BrickDomain, values: np.ndarray, fill=0.0) -> np.ndarray:
    """Scatter brick rows into a dense volume; see :meth:`BrickDomain.to_dense`."""
    return domain.to_dense(values, fill)
