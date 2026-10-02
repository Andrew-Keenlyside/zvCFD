"""Node-field snapshots of the finite-volume solver: one Zarr v3 group per time.

A snapshot group holds arrays in the mesh's node order (the order of
``vertex_attributes/node`` in the volume store, :mod:`zvcfd.io.mesh_store`):

- ``velocity``: ``(N, 3)`` (x, y, z components, m/s);
- ``pressure``: ``(N,)`` Pa;
- optionally wall arrays indexed by ``wall_nodes`` ``(W,)``: ``wss`` ``(W, 3)``
  (Pa), and time statistics ``tawss``, ``osi`` ``(W,)``;
- any other per-node array a caller adds.

Arrays are chunked along the node axis and zstd-compressed. The group's
``zvcfd`` attribute records ``kind: "fv-fields"``, the step, the time, the
node count and the relative path of the mesh store, so a snapshot is
self-describing without the run collection that publishes it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

KIND = "fv-fields"


def write_node_fields(path, fields: dict[str, np.ndarray], *, step: int, time_s: float | None,
                      mesh: str | None = None, chunk: int = 1 << 18,
                      dtype: str | None = "float32", compress: bool = True) -> Path:
    """Write one snapshot group; float arrays are cast to ``dtype`` (``None`` keeps them)."""
    import zarr
    from zarr.codecs import ZstdCodec

    path = Path(path)
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    n = None
    for name, a in fields.items():
        a = np.asarray(a)
        if dtype is not None and a.dtype.kind == "f":
            a = a.astype(dtype)
        chunks = (min(chunk, max(len(a), 1)),) + a.shape[1:]
        arr = root.create_array(name, shape=a.shape, dtype=a.dtype, chunks=chunks,
                                compressors=[ZstdCodec(level=3)] if compress else None)
        arr[...] = a
        if name in ("velocity", "pressure"):
            n = len(a)
    root.attrs["zvcfd"] = {"kind": KIND, "step": int(step),
                           "time_s": None if time_s is None else float(time_s),
                           "nodes": n, "mesh": mesh, "fields": sorted(fields)}
    return path


def read_node_fields(path) -> tuple[dict[str, np.ndarray], dict]:
    """``(fields, attributes)`` of a snapshot group."""
    import zarr

    root = zarr.open_group(str(path), mode="r")
    meta = dict(root.attrs["zvcfd"])
    if meta.get("kind") != KIND:
        raise ValueError(f"{path}: not a {KIND} group")
    return {k: np.asarray(root[k][...]) for k in meta["fields"]}, meta


__all__ = ["KIND", "read_node_fields", "write_node_fields"]
