import importlib

import numpy as np
import pytest


def _has(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def _gpu() -> bool:
    if not _has("cupy"):
        return False
    import cupy as cp

    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _store() -> bool:
    if not _has("zarr_vectors"):
        return False
    import zarr_vectors as zv

    caps = getattr(zv, "runtime_capabilities", None)
    return bool(caps and caps().get("read_cells") and caps().get("defer_presence"))


HAS_GPU = _gpu()
HAS_STORE = _store()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not HAS_GPU:
            item.add_marker(pytest.mark.skip(reason="needs cupy and a CUDA device"))
        if "store" in item.keywords and not HAS_STORE:
            item.add_marker(pytest.mark.skip(reason="needs zarr-vectors gpu-backend (read_cells)"))


@pytest.fixture
def porous64():
    from zvcfd.phantoms import porous_spheres

    return porous_spheres(64, porosity_target=0.5, radius=5)


@pytest.fixture
def tube_mask():
    """A straight tube of radius 6 along x in a 32x32x64 box."""
    z, y = np.mgrid[:32, :32]
    disk = (z - 15.5) ** 2 + (y - 15.5) ** 2 <= 6.0 ** 2
    return np.repeat(disk[:, :, None], 64, axis=2)
