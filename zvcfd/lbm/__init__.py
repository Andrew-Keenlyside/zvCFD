"""GPU lattice-Boltzmann solvers (needs cupy).

Imported on demand: ``import zvcfd`` never imports cupy. Install the GPU
extra (``pip install "zvcfd[gpu]"``) or, in a conda environment, cupy from
conda-forge.
"""

from __future__ import annotations

try:
    import cupy  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised without cupy
    raise ImportError(
        "zvcfd.lbm needs cupy. Install it with pip install 'zvcfd[gpu]', or in a conda "
        "environment conda install -c conda-forge cupy."
    ) from exc

from zvcfd.lbm.multi import MultiLBM
from zvcfd.lbm.solver import (
    CarreauYasuda,
    DenseLBM,
    SparseLBM,
    dense_from_bricks,
    equilibrium,
    kernel,
)

__all__ = ["CarreauYasuda", "DenseLBM", "MultiLBM", "SparseLBM", "dense_from_bricks",
           "equilibrium", "kernel"]
