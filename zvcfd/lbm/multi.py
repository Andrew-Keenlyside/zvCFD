"""Multi-GPU lattice-Boltzmann: one partition per device, ghost-brick exchange.

One process drives every device. Each partition is a :class:`SparseLBM`
over a partition-local domain (its owned bricks, then a one-brick ghost
layer). A step is:

1. every partition collides and streams its owned bricks (kernels on each
   device, launched without waiting, so devices run concurrently);
2. ghost bricks are refreshed from their owners (device-to-device copies,
   peer-to-peer over NVLink where the devices allow it);
3. every partition writes its patch cells (the boundary kernel reads
   interior neighbours, which may be ghosts). Patch links are set before
   step 1, from each link cell's own populations;
4. ghosts are refreshed again, so ghost copies of patch cells are current.

The arithmetic per cell is identical to a single-partition run, and the
result is **bit-identical** to it (``tests/test_multi.py``). With more
partitions than devices, partitions share devices round-robin; on a
one-GPU machine that exercises the whole exchange path.

What is not optimised yet: exchanges copy whole ghost bricks, not only the
populations that cross each face, and communication does not overlap the
interior kernel. Both are milestone-1 follow-ups.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from zvcfd.boundary import BoundarySet
from zvcfd.domain import BrickDomain
from zvcfd.lbm._cuda import Q
from zvcfd.lbm.solver import SparseLBM


class MultiLBM:
    """A :class:`SparseLBM` split across partitions and devices.

    Args:
        domain: The whole domain.
        n_parts: Number of partitions (default: number of devices).
        chunk_bricks: Ownership unit; partitions own whole chunks of
            ``chunk_bricks³`` bricks, so they can write their store cells
            without locks.
        devices: Device ids, used round-robin (default: every visible device).
        boundary: Patch cells in the whole domain's numbering.
        **solver_kw: Passed to every :class:`SparseLBM`.
    """

    def __init__(self, domain: BrickDomain, *, n_parts: int | None = None, chunk_bricks: int = 32,
                 devices: list[int] | None = None, boundary: BoundarySet | None = None,
                 **solver_kw):
        if devices is None:
            devices = list(range(cp.cuda.runtime.getDeviceCount()))
        n_parts = n_parts or len(devices)
        self.domain = domain
        self.devices = [devices[p % len(devices)] for p in range(n_parts)]
        self.parts = domain.partition(n_parts, chunk_bricks)
        self.chunk_bricks = chunk_bricks
        self._enable_peer_access()
        self.locals: list[BrickDomain] = []
        self.solvers: list[SparseLBM] = []
        for p in range(n_parts):
            local = domain.local(self.parts, p)
            with cp.cuda.Device(self.devices[p]):
                b = boundary.localize(local) if boundary is not None else None
                self.solvers.append(SparseLBM(local, boundary=b, **solver_kw))
            self.locals.append(local)
        self.boundary = boundary
        self._plan()
        self.steps = 0

    def _enable_peer_access(self) -> None:
        for a in set(self.devices):
            for b in set(self.devices):
                if a != b and cp.cuda.runtime.deviceCanAccessPeer(a, b):
                    with cp.cuda.Device(a):
                        try:
                            cp.cuda.runtime.deviceEnablePeerAccess(b)
                        except cp.cuda.runtime.CUDARuntimeError:
                            pass        # already enabled

    def _plan(self) -> None:
        """For each (receiver p, owner q): ghost slots on p and owned slots on q."""
        owner_slot = np.empty(self.domain.n_bricks, np.int64)
        for p, loc in enumerate(self.locals):
            owner_slot[loc.global_ids[:loc.n_owned]] = np.arange(loc.n_owned)
        self.plan = []
        for p, loc in enumerate(self.locals):
            ghosts = loc.global_ids[loc.n_owned:]
            if not len(ghosts):
                continue
            owners = self.parts[ghosts]
            slots = np.arange(loc.n_owned, loc.n_bricks)
            for q in np.unique(owners):
                sel = owners == q
                with cp.cuda.Device(self.devices[p]):
                    dst = cp.asarray(slots[sel])
                with cp.cuda.Device(self.devices[q]):
                    src = cp.asarray(owner_slot[ghosts[sel]])
                self.plan.append((p, int(q), dst, src))

    @property
    def ghost_fraction(self) -> float:
        g = sum(loc.n_bricks - loc.n_owned for loc in self.locals)
        return g / max(self.domain.n_bricks, 1)

    def _sync(self) -> None:
        for d in set(self.devices):
            cp.cuda.Device(d).synchronize()

    def exchange(self) -> None:
        """Refresh every ghost brick from its owner's newest buffer."""
        self._sync()
        for p, q, dst, src in self.plan:
            sq, sp = self.solvers[q], self.solvers[p]
            with cp.cuda.Device(self.devices[q]):
                buf = sq.f0.reshape(Q, sq.nb, 512)[:, src, :]
            with cp.cuda.Device(self.devices[p]):
                if self.devices[p] != self.devices[q]:
                    buf = cp.asarray(buf)            # device-to-device (peer) copy
                sp.f0.reshape(Q, sp.nb, 512)[:, dst, :] = buf
        self._sync()

    def step(self, k: int = 1) -> None:
        has_boundary = any(s.boundary is not None for s in self.solvers)
        for _ in range(k):
            for d, s in zip(self.devices, self.solvers):
                with cp.cuda.Device(d):
                    s.apply_links()             # owned cells only: no ghost data needed
                    s.step_main()
            self.exchange()
            if has_boundary:
                for d, s in zip(self.devices, self.solvers):
                    with cp.cuda.Device(d):
                        s.apply_boundary()
                self.exchange()
            self.steps += 1

    def set_patch(self, index: int, **kw) -> None:
        for d, s in zip(self.devices, self.solvers):
            if s.boundary is not None:
                with cp.cuda.Device(d):
                    s.set_patch(index, **kw)

    def patch_flux(self) -> np.ndarray:
        out = None
        for d, s in zip(self.devices, self.solvers):
            if s.boundary is None:
                continue
            with cp.cuda.Device(d):
                v = s.patch_flux()
            out = v if out is None else out + v
        return out if out is not None else np.zeros(0)

    def patch_velocity(self) -> np.ndarray:
        out = None
        for d, s in zip(self.devices, self.solvers):
            if s.boundary is None:
                continue
            with cp.cuda.Device(d):
                v = s.patch_velocity()
            out = v if out is None else out + v
        return out if out is not None else np.zeros((0, 5))

    @property
    def fluid_cells(self) -> int:
        return self.domain.fluid_cells

    def fields(self) -> dict[str, np.ndarray]:
        """Owned-brick fields gathered to the host in the whole domain's brick order."""
        out = {k: np.empty((self.domain.n_bricks, 512), np.float32)
               for k in ("rho", "ux", "uy", "uz")}
        for d, s, loc in zip(self.devices, self.solvers, self.locals):
            with cp.cuda.Device(d):
                f = s.fields()
                for k in out:
                    out[k][loc.global_ids[:loc.n_owned]] = cp.asnumpy(f[k][:loc.n_owned])
        return out
