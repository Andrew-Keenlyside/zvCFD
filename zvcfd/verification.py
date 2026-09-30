"""Solution verification: observed order, Richardson extrapolation, grid convergence index.

For a quantity ``f`` computed on three systematically refined grids (fine
``f1``, medium ``f2``, coarse ``f3``, refinement ratio ``r``):

- observed order ``p = ln |(f3 − f2)/(f2 − f1)| / ln r``;
- Richardson extrapolate ``f_ext = f1 + (f1 − f2)/(r^p − 1)``;
- fine-grid convergence index ``GCI = F_s |(f1 − f2)/f1| / (r^p − 1)``, with
  safety factor ``F_s = 1.25`` for three grids. With only two grids, the
  formal order is assumed and ``F_s = 3``.

The procedure is that of Roache (1994) as standardised in ASME V&V 20
and by Celik et al. (2008). A study is in the asymptotic range when the
observed order is near the formal order, and when ``GCI_23 ≈ r^p GCI_12``.

References: P. J. Roache, *Perspective: a method for uniform reporting of
grid refinement studies*, J. Fluids Eng. 116, 405 (1994); I. B. Celik et
al., *Procedure for estimation and reporting of uncertainty due to
discretization in CFD applications*, J. Fluids Eng. 130, 078001 (2008);
ASME V&V 20-2009.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GridStudy:
    """The result of a three-grid study of one quantity."""

    f1: float
    f2: float
    f3: float
    r: float
    order: float
    extrapolated: float
    gci_fine: float            # relative
    gci_medium: float          # relative
    asymptotic_ratio: float    # GCI_23 / (r^p GCI_12); ≈ 1 in the asymptotic range
    oscillatory: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def observed_order(f1: float, f2: float, f3: float, r: float) -> float:
    """Observed order from three grids refined by a constant ratio ``r``."""
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        return math.inf
    return math.log(abs(e32 / e21)) / math.log(r)


def grid_study(f1: float, f2: float, f3: float, r: float = 2.0, *,
               safety: float = 1.25) -> GridStudy:
    """Three-grid study (fine, medium, coarse) with constant refinement ratio ``r``."""
    p = observed_order(f1, f2, f3, r)
    rp = r ** p if math.isfinite(p) else math.inf
    ext = f1 + (f1 - f2) / (rp - 1) if math.isfinite(rp) and rp != 1 else f1
    g12 = safety * abs((f1 - f2) / f1) / (rp - 1) if f1 and rp != 1 else 0.0
    g23 = safety * abs((f2 - f3) / f2) / (rp - 1) if f2 and rp != 1 else 0.0
    ratio = g23 / (rp * g12) if g12 else math.nan
    osc = (f3 - f2) * (f2 - f1) < 0
    return GridStudy(f1, f2, f3, r, p, ext, g12, g23, ratio, osc)


def gci_two_grids(f1: float, f2: float, r: float = 2.0, order: float = 2.0,
                  safety: float = 3.0) -> float:
    """Relative fine-grid GCI from two grids, assuming the formal ``order``."""
    return safety * abs((f1 - f2) / f1) / (r ** order - 1)


__all__ = ["GridStudy", "gci_two_grids", "grid_study", "observed_order"]
