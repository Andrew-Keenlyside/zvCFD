"""Time statistics of wall shear stress: TAWSS, OSI and RRT.

Over the averaging window, with ``τ(t)`` the wall shear vector at a wall node:

    TAWSS = (1/T) ∫ |τ| dt,        OSI = ½ (1 − |∫ τ dt| / ∫ |τ| dt),
    RRT = 1 / ((1 − 2 OSI) TAWSS)

integrated with the trapezoidal rule over the committed time steps.

References: X. He, D. N. Ku, *Pulsatile flow in the human left coronary
artery bifurcation*, J. Biomech. Eng. 118, 74 (1996) (OSI); H. A. Himburg
et al., *Spatial comparison between wall shear stress measures and porcine
arterial endothelial permeability*, Am. J. Physiol. 286, H1916 (2004) (RRT).
"""

from __future__ import annotations

import numpy as np


class WallShearStats:
    """Accumulates ``∫ τ dt`` and ``∫ |τ| dt`` per wall node (trapezoidal)."""

    def __init__(self, n: int):
        self.int_tau = np.zeros((n, 3))
        self.int_mag = np.zeros(n)
        self.T = 0.0
        self._last = None

    def add(self, t: float, tau: np.ndarray) -> None:
        tau = np.asarray(tau, float)
        if self._last is not None:
            t0, tau0 = self._last
            h = t - t0
            self.int_tau += 0.5 * h * (tau0 + tau)
            self.int_mag += 0.5 * h * (np.linalg.norm(tau0, axis=1) + np.linalg.norm(tau, axis=1))
            self.T += h
        self._last = (float(t), tau.copy())

    def results(self) -> dict[str, np.ndarray]:
        if self.T <= 0:
            raise ValueError("need at least two samples")
        tawss = self.int_mag / self.T
        osi = 0.5 * (1 - np.linalg.norm(self.int_tau, axis=1)
                     / np.maximum(self.int_mag, 1e-300))
        with np.errstate(divide="ignore"):
            rrt = 1.0 / np.maximum((1 - 2 * osi) * tawss, 1e-300)
        return {"tawss": tawss, "osi": osi, "rrt": rrt}


__all__ = ["WallShearStats"]
