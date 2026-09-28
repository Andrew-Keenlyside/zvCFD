"""Brick fields in Zarr Vectors stores.

A *brick store* is an ordinary Zarr Vectors store (custom geometry type
``zvcfd:bricks``, spec section 12.5) in which **one vertex is one 8^3 brick**:

- ``vertices``: the brick centre, in physical units, axes (z, y, x);
- ``vertex_attributes/<field>``: that brick's voxel values, ``ncols = 512``
  (``512 * C`` for a C-component field), x fastest within the brick;
- ``vertex_attributes/flags`` (domain store only): uint8 voxel flags.

A store cell is a chunk of ``chunk_bricks^3`` bricks, and is the unit a
GPU worker owns (:meth:`zvcfd.domain.BrickDomain.partition`). Writing
follows the zarr-vectors three-phase contract (``docs/io.md``):

1. coordinator: :func:`create_brick_store` (allocates arrays, defers presence);
2. workers: :func:`write_brick_chunk` for the chunks they own, no locks;
3. coordinator: :func:`finalize_brick_store` (one presence rebuild).

Solid-only bricks are never stored, so a store's size follows the fluid
volume. Every call goes through ``zarr_vectors.building``, the supported
surface for tools that build stores.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from zvcfd.domain import BrickDomain

GEOMETRY_TYPE = "zvcfd:bricks"
AXES = [{"name": n, "type": "space"} for n in ("z", "y", "x")]


def _zb():
    from zarr_vectors import building

    return building


def store_geometry(domain: BrickDomain, voxel_size: float, chunk_bricks: int):
    """``(bounds, chunk_shape)`` in physical units for a domain."""
    extent = [float(s) * voxel_size for s in domain.shape]
    chunk = float(chunk_bricks * domain.brick * voxel_size)
    return ([0.0, 0.0, 0.0], extent), (chunk, chunk, chunk)


def create_brick_store(
    path: Any,
    domain: BrickDomain,
    *,
    voxel_size: float,
    fields: Mapping[str, str] | Sequence[str] = ("rho", "ux", "uy", "uz"),
    chunk_bricks: int = 32,
    unit: str = "micrometer",
    shard_shape: int | Sequence[int] | None = None,
    compressor: Any = None,
    flags: bool = False,
    backend: str | None = None,
):
    """Coordinator: create the store and allocate every array, once.

    Args:
        path: Store path or URL.
        domain: The brick domain (only its shape and brick size are used here).
        voxel_size: Voxel edge in ``unit``.
        fields: Field names, or ``{name: dtype}``; dtype defaults to float32.
        chunk_bricks: Bricks per chunk edge; a chunk is one store cell.
        shard_shape: Cells per shard (declared on the store so every array
            inherits it). With shards, a worker must own whole shards.
        flags: Also allocate the ``flags`` attribute (the domain store).

    Returns:
        The level-0 group, presence deferred.
    """
    zb = _zb()
    bounds, chunk = store_geometry(domain, voxel_size, chunk_bricks)
    root = zb.create_store(path, bounds=bounds, chunk_shape=chunk, axes=AXES,
                           geometry_types=[GEOMETRY_TYPE], unit=unit, shard_shape=shard_shape,
                           compressor=compressor, backend=backend)
    specs = dict(fields) if isinstance(fields, Mapping) else {f: "float32" for f in fields}
    names = ["flags"] * flags + list(specs)
    level = zb.create_resolution_level(root, 0, zb.LevelMetadata(
        level=0, vertex_count=domain.n_bricks, arrays_present=["vertices", "vertex_attributes"]))
    ncols = domain.cells_per_brick
    # attribute arrays take their codecs from the session that creates them
    with zb.open_write_session(level, compressor=compressor, bounds=bounds, chunk_shape=chunk):
        zb.create_vertices_array(level, dtype="float32")
        for name in names:
            dtype = "uint8" if name == "flags" else specs[name]
            zb.create_attribute_array(level, name, dtype=dtype, ncols=ncols)
    zb.defer_presence(level)
    if isinstance(path, (str, os.PathLike)):
        # Namespaced user metadata, through the facade (building has no accessor for it yet).
        import zarr_vectors as zv

        zv.open(str(path), mode="r+").metadata["zvcfd"].update({
            "brick": domain.brick, "chunk_bricks": chunk_bricks, "voxel_size": voxel_size,
            "unit": unit, "shape_zyx": list(domain.shape),
            "fields": {n: specs.get(n, "uint8") for n in names}})
    return level


def open_brick_level(path: Any, mode: str = "r", *, backend: str | None = None):
    zb = _zb()
    return zb.get_resolution_level(zb.open_store(path, mode=mode, backend=backend), 0)


def brick_centres(domain: BrickDomain, bricks: np.ndarray, voxel_size: float) -> np.ndarray:
    c = domain.coords[bricks].astype(np.float64)
    return ((c + 0.5) * domain.brick * voxel_size).astype(np.float32)


def write_brick_chunk(
    level,
    domain: BrickDomain,
    bricks: np.ndarray,
    fields: Mapping[str, Any],
    *,
    voxel_size: float,
    chunk_bricks: int,
) -> tuple[int, ...]:
    """Worker: write one chunk's bricks and their field rows.

    ``bricks`` are brick indices that all lie in one chunk; each value in
    ``fields`` is ``(len(bricks), 512)`` (numpy, or a device array, which
    zarr-vectors copies off once). Presence is not stamped; the
    coordinator's :func:`finalize_brick_store` does that once.

    Returns the chunk coordinate written.
    """
    zb = _zb()
    bricks = np.asarray(bricks)
    cc = np.unique(domain.coords[bricks] // chunk_bricks, axis=0)
    if len(cc) != 1:
        raise ValueError(f"bricks span {len(cc)} chunks; write one chunk per call")
    key = tuple(int(v) for v in cc[0])
    zb.write_chunk_vertices(level, key, [brick_centres(domain, bricks, voxel_size)],
                            dtype="float32", record_presence=False)
    for name, rows in fields.items():
        rows = _host(rows)
        zb.write_chunk_attributes(level, name, key, [rows], dtype=rows.dtype,
                                  record_presence=False)
    return key


def write_brick_chunks(level, domain: BrickDomain, fields: Mapping[str, Any], *,
                       voxel_size: float, chunk_bricks: int,
                       bricks: Sequence[int] | np.ndarray | None = None) -> int:
    """Worker: write every chunk touched by ``bricks`` in one write session.

    ``fields`` values are ``(len(bricks), 512)`` rows aligned with ``bricks``
    (default: every brick of the domain, in domain order), so a worker only
    ever holds its own share. ``bricks`` must cover whole chunks: a chunk
    written by two workers keeps whichever landed last.
    Returns the number of chunks written.
    """
    zb = _zb()
    ids = np.arange(domain.n_bricks) if bricks is None else np.asarray(bricks)
    host = {k: _host(v) for k, v in fields.items()}
    cc = domain.coords[ids] // chunk_bricks
    order = np.lexsort(cc.T[::-1])
    cc = cc[order]
    starts = np.flatnonzero(np.r_[True, (np.diff(cc, axis=0) != 0).any(1)])
    bounds, chunk = store_geometry(domain, voxel_size, chunk_bricks)
    with zb.open_write_session(level, bounds=bounds, chunk_shape=chunk):
        for s, e in zip(starts, np.r_[starts[1:], len(ids)]):
            rows = order[s:e]
            write_brick_chunk(level, domain, ids[rows], {k: v[rows] for k, v in host.items()},
                              voxel_size=voxel_size, chunk_bricks=chunk_bricks)
    return len(starts)


def finalize_brick_store(level) -> None:
    """Coordinator, after every worker: rebuild presence once and record what exists."""
    zb = _zb()
    zb.rebuild_presence(level)
    zb.refresh_arrays_present(level)


def read_brick_chunks(level, chunk_coords, fields: Sequence[str], *, device: str | None = None,
                      decode: str = "auto"):
    """Read chunks' brick centres and field rows in one batched read.

    Returns the zarr-vectors ``CellBatch``: ``batch["vertex_attributes/<f>"].data``
    is ``(rows, 512)`` with CSR ``offsets`` over ``batch.chunk_coords``;
    ``device="cuda"`` decodes on the GPU where the store allows it.
    """
    zb = _zb()
    arrays = ["vertices"] + [f"vertex_attributes/{f}" for f in fields]
    return zb.read_cells(level, chunk_coords, arrays, device=device, decode=decode)


def store_metadata(path: Any) -> dict:
    """The ``zvcfd`` user-metadata namespace of a brick store."""
    import zarr_vectors as zv

    return dict(zv.open(str(path)).metadata["zvcfd"])


def _chunk_keys(level) -> np.ndarray:
    keys = _zb().list_chunk_keys(level, "vertices")
    return np.array([tuple(int(v) for v in k) for k in keys], np.int64).reshape(-1, 3)


def _brick_coords(centres: np.ndarray, meta: dict) -> np.ndarray:
    return np.round(np.asarray(centres) / (meta["brick"] * meta["voxel_size"]) - 0.5).astype(
        np.int64)


def read_domain(path: Any, *, periodic: bool = False) -> BrickDomain:
    """Rebuild the :class:`BrickDomain` from a domain store (one with ``flags``)."""
    meta = store_metadata(path)
    level = open_brick_level(path)
    batch = read_brick_chunks(level, _chunk_keys(level), ["flags"])
    return BrickDomain.from_bricks(meta["shape_zyx"], _brick_coords(batch["vertices"].data, meta),
                                   batch["vertex_attributes/flags"].data, brick=meta["brick"],
                                   periodic=periodic)


def read_snapshot(path: Any, domain: BrickDomain, fields: Sequence[str] | None = None, *,
                  device: str | None = None, decode: str = "auto") -> dict[str, Any]:
    """Every brick of a snapshot store, as ``{field: (n_bricks, 512)}`` in domain order.

    With ``device="cuda"`` the arrays are cupy, decoded on the GPU where the
    store's codec allows it.
    """
    meta = store_metadata(path)
    names = list(fields or meta["fields"])
    level = open_brick_level(path)
    batch = read_brick_chunks(level, _chunk_keys(level), names, device=device, decode=decode)
    centres = batch["vertices"].data
    if hasattr(centres, "__cuda_array_interface__"):
        import cupy as cp

        centres = cp.asnumpy(centres)
    ids = domain.brick_index(_brick_coords(centres, meta))
    if (ids < 0).any():
        raise ValueError("snapshot holds bricks the domain does not")
    out = {}
    for n in names:
        rows = batch[f"vertex_attributes/{n}"].data
        if device == "cuda":
            import cupy as cp

            full = cp.empty((domain.n_bricks,) + rows.shape[1:], rows.dtype)
            full[cp.asarray(ids)] = rows
        else:
            full = np.empty((domain.n_bricks,) + rows.shape[1:], rows.dtype)
            full[ids] = rows
        out[n] = full
    return out


def _host(a: Any) -> np.ndarray:
    if hasattr(a, "__cuda_array_interface__"):
        import cupy as cp

        return cp.asnumpy(a)
    return np.ascontiguousarray(a)
