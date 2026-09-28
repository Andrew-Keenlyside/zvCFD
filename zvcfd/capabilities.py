"""What this install can do: ``zvcfd probe`` and :func:`runtime_capabilities`.

Probe instead of parsing version strings, as zarr-vectors does: keys are
only ever added, and each is a bool. ``probe_device=True`` initialises
CUDA, so a process that will fork workers should probe inside them.
"""

from __future__ import annotations

import importlib


def _imports(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def runtime_capabilities(*, probe_device: bool = False) -> dict[str, bool]:
    """``{capability: available}`` for this install.

    Keys:
        cupy, cuda_device (with ``probe_device``), nccl, peer_access (with
        ``probe_device`` and >= 2 devices), kvikio, nvcomp, pyamg, icechunk,
        zarr_vectors, and the zarr-vectors capabilities zvCFD relies on,
        prefixed ``zv_``: read_cells, device_decode, gpu_codecs, gpu_io,
        defer_presence, append_safe_sharding, dense_manifests.
    """
    caps: dict[str, bool] = {
        "cupy": _imports("cupy"),
        "kvikio": _imports("kvikio"),
        "nvcomp": _imports("nvidia.nvcomp"),
        "pyamg": _imports("pyamg"),
        "icechunk": _imports("icechunk"),
        "zarr_vectors": _imports("zarr_vectors"),
    }
    caps["nccl"] = False
    caps["cuda_device"] = False
    caps["peer_access"] = False
    if caps["cupy"]:
        try:
            from cupy.cuda import nccl  # noqa: F401

            caps["nccl"] = bool(nccl.available) if hasattr(nccl, "available") else True
        except Exception:
            caps["nccl"] = False
        if probe_device:
            import cupy as cp

            try:
                n = cp.cuda.runtime.getDeviceCount()
            except cp.cuda.runtime.CUDARuntimeError:
                n = 0
            caps["cuda_device"] = n > 0
            caps["peer_access"] = n > 1 and bool(cp.cuda.runtime.deviceCanAccessPeer(0, 1))
    zv_keys = ("read_cells", "device_decode", "gpu_codecs", "gpu_io", "defer_presence",
               "append_safe_sharding", "dense_manifests")
    zv: dict[str, bool] = {}
    if caps["zarr_vectors"]:
        import zarr_vectors

        probe = getattr(zarr_vectors, "runtime_capabilities", None)
        if probe is not None:
            try:
                zv = probe(probe_device=probe_device)
            except Exception:
                zv = {}
    for k in zv_keys:
        caps[f"zv_{k}"] = bool(zv.get(k, False))
    return caps


def device_count() -> int:
    """Visible CUDA devices; 0 without cupy or a driver. Initialises CUDA."""
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def require(*names: str, probe_device: bool = True) -> dict[str, bool]:
    """Raise ``RuntimeError`` naming every missing capability."""
    caps = runtime_capabilities(probe_device=probe_device)
    missing = [n for n in names if not caps.get(n, False)]
    if missing:
        raise RuntimeError(f"missing capabilities: {', '.join(missing)} "
                           f"(see `zvcfd probe` and docs/getting_started/installation.md)")
    return caps
