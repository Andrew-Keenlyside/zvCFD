"""Shared pieces of the turbulence benchmarks."""

from __future__ import annotations

import time
from types import SimpleNamespace


def solve_ramped(s, dt: float, *, dt0: float = 0.05, growth: float = 1.5, per_stage: int = 10,
                 max_iterations: int = 1000, tol: float = 1e-8, log=print, every: int = 25):
    """Steady solve with the false time step grown geometrically from ``dt0`` to ``dt``
    (``per_stage`` outer iterations at each step): from a uniform start, the impulsive
    start on a no-slip wall diverges at large steps. Every ``every`` iterations the
    solver's line and the turbulence model's changes go to ``log``. Returns the
    iterations, whether the final stage converged, the history and the wall time."""
    t0 = time.time()
    its, hist = 0, []
    count = [0]

    def every_n(line):
        count[0] += 1
        if log is not None and count[0] % every == 0:
            tm = getattr(s, "turbulence", None)
            h = tm.history[-1] if tm is not None and getattr(tm, "history", None) else {}
            extra = "  turb " + " ".join(f"{k} {v:.1e}" for k, v in h.items()
                                         if isinstance(v, float))
            log(f"{line.strip()}  dt {s.dt:g}{extra if h else ''}  {time.time() - t0:.0f}s")

    d = dt0
    while d < dt and its < max_iterations:
        s.dt = d
        r = s.solve(max_iterations=per_stage, tol=0.0, log=every_n)
        its += r.iterations
        hist += r.history
        d *= growth
    s.dt = dt
    r = s.solve(max_iterations=max(max_iterations - its, 1), tol=tol, log=every_n)
    return SimpleNamespace(iterations=its + r.iterations, converged=r.converged,
                           history=hist + r.history, seconds=time.time() - t0)


def save_fields(path, s, tm, mu: float) -> None:
    """The solution on the z = 0 plane of a one-cell slab, for renderings: node
    coordinates, velocity, pressure, k, kL and mu_t/mu."""
    import numpy as np

    X = s.mesh.nodes
    z0 = np.flatnonzero(np.abs(X[:, 2]) < 1e-12)
    a = s.cp.asnumpy
    U, P = a(s.U)[z0], a(s.P)[z0]
    np.savez_compressed(path, x=X[z0, 0], y=X[z0, 1], u=U[:, 0], v=U[:, 1], p=P,
                        k=a(tm.k)[z0], kl=a(tm.phi)[z0], mut=a(tm.mu_t)[z0] / mu)
