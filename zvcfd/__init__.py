"""zvCFD: GPU computational fluid dynamics backed by Zarr Vectors.

``import zvcfd`` needs only numpy, scipy and zarr; GPU modules
(:mod:`zvcfd.lbm`) and store I/O (:mod:`zvcfd.io.fields`, which needs
zarr-vectors) are imported when used.
"""

from __future__ import annotations

try:
    from importlib.metadata import version

    __version__ = version("zvcfd")
except Exception:  # pragma: no cover - source checkout without an install
    __version__ = "0.0.0+unknown"

from zvcfd.capabilities import device_count, require, runtime_capabilities
from zvcfd.config import RunConfig
from zvcfd.config import load as load_config
from zvcfd.domain import BrickDomain
from zvcfd.io.fluent_msh import read_fluent_boundary
from zvcfd.perfmodel import estimate
from zvcfd.units import Lattice

__all__ = [
    "BrickDomain",
    "Lattice",
    "RunConfig",
    "__version__",
    "device_count",
    "estimate",
    "load_config",
    "read_fluent_boundary",
    "require",
    "runtime_capabilities",
]
