"""Meshes as Zarr Vectors collections: import any mesh once, run both solvers from it.

A **mesh collection** ``<name>.zvmesh/`` is a Zarr v3 group whose
``attributes.ome`` is an RFC-8 collection document, like a run collection
(:mod:`zvcfd.collection`). Its nodes are two Zarr Vectors stores
(:mod:`zvcfd.io.mesh_store`, spec in ``docs/spec/mesh_store.md``):

- ``volume`` (``zvcfd:fv-mesh``): nodes and elements, for the
  finite-volume solver. Absent for a surface-only import;
- ``boundary`` (``zvcfd:fv-boundary``): the boundary triangles, one object
  per zone, with each zone's kind and name. The lattice-Boltzmann solver
  voxelises it; the finite-volume solver rebuilds its boundary zones from it.

Coordinates are metres whatever the source's unit. ``zvcfd import-mesh``
writes a collection; ``zvcfd run`` reads one, or imports a raw mesh file
into a cached collection first (:func:`cached_import`), so every run starts
from Zarr Vectors.

Sources (:func:`read_source`):

- an Ansys Fluent ``.msh`` (ASCII or binary); surface-only imports read
  just its boundary zones, which is all the voxel solver needs;
- a VTK ``.vtu`` volume mesh, or a SimVascular ``mesh-complete`` folder
  (volume mesh plus one ``.vtp`` per face);
- anything `meshio <https://github.com/nschloe/meshio>`_ reads (Gmsh,
  Abaqus, Exodus, MED, Nastran, ...), if it is installed: tetrahedra,
  pyramids, wedges and hexahedra; boundary zones from tagged triangle and
  quadrilateral blocks (Gmsh physical groups, cell sets), and any
  untagged boundary face in a zone ``wall``.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from zvcfd import collection as col
from zvcfd.mesh.core import BoundaryZone, UnstructuredMesh

SUFFIX = ".zvmesh"
KIND = f"{col.PREFIX}:mesh"
UNIT_M = {"m": 1.0, "meter": 1.0, "metre": 1.0, "cm": 1e-2, "centimeter": 1e-2,
          "mm": 1e-3, "millimeter": 1e-3, "um": 1e-6, "micrometer": 1e-6}
VOLUME, BOUNDARY = "volume.zarrvectors", "boundary.zarrvectors"


# ---------------------------------------------------------------- reading sources

def _is_fluent(path: Path) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(4096).lstrip()
    return head.startswith(b"(") and b"$MeshFormat" not in head


def _zone_kind(name: str) -> str:
    n = name.lower()
    if "inlet" in n or "inflow" in n:
        return "velocity-inlet"
    if "outlet" in n or "outflow" in n:
        return "pressure-outlet"
    return "wall"


def _scaled(mesh: UnstructuredMesh, unit: str, source: str) -> UnstructuredMesh:
    if unit not in UNIT_M:
        raise ValueError(f"unit {unit!r}; expected one of {sorted(UNIT_M)}")
    return UnstructuredMesh(mesh.nodes * UNIT_M[unit], mesh.elements, zones=mesh.zones,
                            cell_ids=mesh.cell_ids, unit="meter", source=source)


def _fluent_surface(path: Path) -> UnstructuredMesh:
    """A Fluent mesh's boundary zones only, as triangles (no elements)."""
    from zvcfd.io.fluent_msh import read_fluent_boundary

    b = read_fluent_boundary(str(path))
    tris, zone = b.triangles()
    zones = {}
    for zid, fz in sorted(b.zones.items()):
        sel = zone == zid
        if not sel.any():
            continue
        f = np.concatenate([tris[sel], -np.ones((sel.sum(), 1), np.int64)], 1)
        zones[int(zid)] = BoundaryZone(int(zid), fz.kind, fz.name, f,
                                       -np.ones(sel.sum(), np.int64))
    return UnstructuredMesh(np.asarray(b.nodes, float), {}, zones=zones)


def _meshio_mesh(path: Path) -> UnstructuredMesh:
    try:
        import meshio
    except ImportError as exc:
        raise ImportError(f"{path.name}: this format needs meshio "
                          "(pip install 'zvcfd[meshio]')") from exc
    from zvcfd.mesh.vtk import _orient

    m = meshio.read(str(path))
    nodes = np.asarray(m.points, float)
    if nodes.shape[1] == 2:
        raise ValueError(f"{path.name}: a 2-D mesh")
    volume = {"tetra": "tet", "pyramid": "pyramid", "wedge": "wedge", "hexahedron": "hex"}
    elements = {}
    for block in m.cells:
        k = volume.get(block.type)
        if k is not None:
            e = np.asarray(block.data, np.int64)
            elements[k] = np.concatenate([elements[k], _orient(nodes, k, e)]) \
                if k in elements else _orient(nodes, k, e)
    if not elements:
        raise ValueError(f"{path.name}: no tetrahedra, pyramids, wedges or hexahedra")
    mesh = UnstructuredMesh(nodes, elements, source=str(path))
    # tags of boundary faces: gmsh physical groups, or any integer cell data, or cell sets
    tag_names = {int(v[0]): k for k, v in (m.field_data or {}).items()
                 if len(v) and int(v[-1]) == 2}
    faces, tags = [], []
    for i, block in enumerate(m.cells):
        if block.type not in ("triangle", "quad"):
            continue
        f = np.asarray(block.data, np.int64)
        if block.type == "triangle":
            f = np.concatenate([f, -np.ones((len(f), 1), np.int64)], 1)
        tag = None
        for key in ("gmsh:physical", "medit:ref", "cell_tags", "CellEntityIds"):
            if key in (m.cell_data or {}):
                tag = np.asarray(m.cell_data[key][i], np.int64).reshape(-1)
                break
        if tag is None:
            tag = np.zeros(len(f), np.int64)
            for j, (name, sets) in enumerate((m.cell_sets or {}).items(), start=1):
                if i < len(sets) and sets[i] is not None and len(sets[i]):
                    tag[np.asarray(sets[i], np.int64)] = j
                    tag_names.setdefault(j, name)
        faces.append(f)
        tags.append(tag)
    names = {t: tag_names.get(t, f"zone-{t}") for t in np.unique(np.concatenate(tags))} \
        if tags else {}
    mesh.zones = _zones_by_face_match(mesh, faces, tags, names)
    return mesh


def _zones_by_face_match(mesh: UnstructuredMesh, faces: list, tags: list,
                         names: dict) -> dict[int, BoundaryZone]:
    """Zones from tagged boundary faces; untagged boundary faces form zone ``wall``."""
    from zvcfd.mesh.fluent import match_rows

    fx = mesh.faces()
    b = fx["c1"] < 0
    bf, owner = fx["faces"][b], fx["c0"][b]
    zone = np.zeros(len(bf), np.int64)                           # 0: untagged
    if faces:
        f = np.concatenate(faces)
        t = np.concatenate(tags)
        hit = match_rows(f, bf)
        zone[hit >= 0] = t[hit[hit >= 0]]
    out = {}
    ids = sorted(set(np.unique(zone).tolist()))
    for k, tag in enumerate(ids, start=1):
        sel = zone == tag
        name = "wall" if tag == 0 else str(names.get(tag, f"zone-{tag}"))
        out[k] = BoundaryZone(k, _zone_kind(name), name, bf[sel], owner[sel])
    return out


def read_source(src, *, unit: str = "m", surface_only: bool = False) -> UnstructuredMesh:
    """Any supported mesh (see the module docstring), in metres."""
    src = Path(src)
    if src.is_dir():
        from zvcfd.mesh.vtk import read_vtu_mesh

        vtu = sorted(src.glob("*.mesh.vtu")) or sorted(src.glob("*.vtu"))
        if not vtu:
            raise FileNotFoundError(f"{src}: no .vtu volume mesh")
        mesh = read_vtu_mesh(vtu[0], surfaces=src, unit=unit)
    elif src.suffix == ".vtu":
        from zvcfd.mesh.vtk import read_vtu_mesh

        mesh = read_vtu_mesh(src, unit=unit)
    elif src.suffix in (".msh", ".cas") and _is_fluent(src):
        mesh = None
        if surface_only:
            try:
                mesh = _fluent_surface(src)
            except (ValueError, IndexError, UnicodeDecodeError):
                mesh = None
        if mesh is None or not mesh.zones:           # binary sections: the full reader
            from zvcfd.mesh.fluent import read_fluent_mesh

            mesh = read_fluent_mesh(src, unit=unit)
    else:
        mesh = _meshio_mesh(src)
    return _scaled(mesh, unit, str(src))


# ---------------------------------------------------------------- collections

def _document(path: Path, attrs: dict) -> dict:
    doc = col.run_document(path, name=path.stem, unit="meter", run_id=col.new_id("mesh"))
    doc["attributes"].pop(f"{col.PREFIX}:run")
    doc["attributes"][KIND] = attrs
    return doc


def is_mesh_collection(path) -> bool:
    p = Path(path) / col.ZARR_JSON
    if not p.is_file():
        return False
    try:
        return KIND in col.read(path)["attributes"]
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def write_mesh_collection(path, mesh: UnstructuredMesh, *, chunk: float | None = None,
                          surface_only: bool = False, compressor=None) -> Path:
    """Write ``mesh`` (in metres) as a mesh collection; returns its path."""
    from zvcfd.io.mesh_store import write_boundary_store, write_mesh_store

    if mesh.unit not in ("m", "meter", "metre"):
        raise ValueError(f"mesh unit {mesh.unit!r}: scale to metres first (read_source does)")
    if not mesh.zones:
        raise ValueError("the mesh has no boundary zones")
    if mesh.unit != "meter":                      # the stores take UDUNITS names
        mesh = UnstructuredMesh(mesh.nodes, mesh.elements, zones=mesh.zones,
                                cell_ids=mesh.cell_ids, unit="meter", source=mesh.source)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    lo, hi = mesh.nodes.min(0), mesh.nodes.max(0)
    chunk = float(chunk or (hi - lo).max() / 4)
    t0 = time.time()
    doc = _document(path, {})
    info = {}
    if mesh.elements and not surface_only:
        info["volume"] = write_mesh_store(path / VOLUME, mesh, chunk=chunk,
                                          compressor=compressor)
        col.add_node(doc, col.node(f"{col.PREFIX}:fv-mesh", "volume", f"./{VOLUME}",
                                   attributes={f"{col.PREFIX}:fv-mesh": info["volume"]}))
    info["boundary"] = write_boundary_store(path / BOUNDARY, mesh, chunk=chunk,
                                            compressor=compressor)
    col.add_node(doc, col.node(f"{col.PREFIX}:fv-boundary", "boundary", f"./{BOUNDARY}"))
    doc["attributes"][KIND] = {
        "unit": "meter", "source": mesh.source, "chunk": chunk,
        "volume": bool(mesh.elements) and not surface_only,
        "nodes": mesh.n_nodes, "elements": mesh.counts() if not surface_only else {},
        "zones": [{"zone": int(z), "kind": zz.kind, "name": zz.name, "faces": int(zz.n_faces)}
                  for z, zz in sorted(mesh.zones.items())],
        "bounds_m": [lo.tolist(), hi.tolist()], "write_s": time.time() - t0}
    col.write(path, doc)
    return path


def collection_info(path) -> dict:
    return dict(col.read(path)["attributes"][KIND])


def read_mesh(path) -> UnstructuredMesh:
    """The volume mesh with its boundary zones, as the finite-volume solver needs it."""
    from zvcfd.io.mesh_store import read_boundary_store, read_mesh_store, zones_from_boundary

    path = Path(path)
    info = collection_info(path)
    if not info.get("volume"):
        raise ValueError(f"{path}: a surface-only mesh collection (re-import it with its "
                         "volume for the finite-volume solver)")
    mesh = read_mesh_store(path / VOLUME)
    mesh.zones = zones_from_boundary(mesh, read_boundary_store(path / BOUNDARY))
    return UnstructuredMesh(mesh.nodes, mesh.elements, zones=mesh.zones, unit="meter",
                            source=str(path))


def read_surface(path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """``(triangles (T, 3, 3) xyz metres, zone id per triangle, zone table)``, for voxelising."""
    import zarr_vectors as zv
    from zarr_vectors import building as zb

    from zvcfd.io.mesh_store import vertex_rows

    bpath = Path(path) / BOUNDARY
    meta = dict(zv.open(str(bpath)).metadata["zvcfd"])
    level = zb.get_resolution_level(zb.open_store(str(bpath)), 0)
    keys = [tuple(int(v) for v in k) for k in zb.list_chunk_keys(level, "vertices")]
    xyz, zone, start, total = [], [], {}, 0
    for cc in keys:
        v = np.concatenate(zb.read_chunk_vertices(level, cc, dtype="float64")).reshape(-1, 3)
        z = np.concatenate(zb.read_chunk_attributes(level, "zone", cc, dtype="int64")).reshape(-1)
        start[cc] = total
        xyz.append(v[:, ::-1])
        zone.append(z)
        total += len(v)
    xyz, zone = np.concatenate(xyz), np.concatenate(zone)
    chunks, vi = zb.read_link_arrays(level)
    v = vertex_rows(chunks, vi, start)
    return xyz[v], zone[v[:, 0]], meta["zones"]


def import_mesh(src, out=None, *, unit: str = "m", chunk: float | None = None,
                surface_only: bool = False, log=print) -> Path:
    """Read any supported mesh and write it as a mesh collection."""
    src = Path(src)
    out = Path(out) if out else src.with_name((src.stem if src.is_file() else src.name) + SUFFIX)
    t0 = time.time()
    mesh = read_source(src, unit=unit, surface_only=surface_only)
    log(f"read {src.name}: {mesh.n_nodes:,} nodes, {mesh.counts() or 'surface only'}, "
        f"{len(mesh.zones)} zones ({time.time() - t0:.1f} s)")
    t1 = time.time()
    write_mesh_collection(out, mesh, chunk=chunk, surface_only=surface_only)
    log(f"wrote {out} ({time.time() - t1:.1f} s)")
    return out


def cached_import(src, cache_dir, *, unit: str = "m", chunk: float | None = None,
                  surface_only: bool = False, log=print) -> Path:
    """A mesh collection for a raw mesh file, imported once into ``cache_dir`` and reused.

    The cache key covers the source's path, size and modification time, the unit, the
    chunk edge and whether the import is surface-only. A full import also serves
    surface-only requests.
    """
    src = Path(src).resolve()
    st = src.stat() if src.is_file() else max((p.stat() for p in src.rglob("*") if p.is_file()),
                                             key=lambda s: s.st_mtime)
    stem = src.stem if src.is_file() else src.name

    def key(surface):
        blob = json.dumps([str(src), st.st_size, int(st.st_mtime), unit, chunk, surface]).encode()
        return Path(cache_dir) / f"{stem}-{hashlib.sha256(blob).hexdigest()[:10]}{SUFFIX}"

    full = key(False)
    if is_mesh_collection(full):
        return full
    if surface_only and is_mesh_collection(key(True)):
        return key(True)
    out = key(surface_only)
    log(f"importing {src.name} into {out}")
    return import_mesh(src, out, unit=unit, chunk=chunk, surface_only=surface_only, log=log)


def resolve(src, cache_dir, *, unit: str = "m", chunk: float | None = None,
            surface_only: bool = False, log=print) -> Path:
    """The mesh collection a run reads: ``src`` itself if it is one, else a cached import."""
    if is_mesh_collection(src):
        if not surface_only and not collection_info(src).get("volume"):
            raise ValueError(f"{src}: a surface-only mesh collection; the finite-volume "
                             "solver needs the volume (re-import without --surface-only)")
        return Path(src)
    return cached_import(src, cache_dir, unit=unit, chunk=chunk, surface_only=surface_only,
                         log=log)


def voxelize_collection(path, voxel: float, *, pad: int = 2):
    """Voxelise a mesh collection's boundary store (``voxel`` in metres); see
    :func:`zvcfd.geometry.voxelize_mesh`."""
    from zvcfd.geometry import voxelize_mesh

    tri, zone, zones = read_surface(path)
    return voxelize_mesh(tri, zone, {z["zone"]: z["kind"] for z in zones}, voxel,
                         zone_names={z["zone"]: z["name"] for z in zones}, unit_scale=1.0,
                         pad=pad)


__all__ = ["cached_import", "collection_info", "import_mesh", "is_mesh_collection",
           "read_mesh", "read_source", "read_surface", "resolve", "voxelize_collection",
           "write_mesh_collection"]
