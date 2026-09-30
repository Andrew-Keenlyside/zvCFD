"""Unstructured meshes in Zarr Vectors stores: a ``zvcfd:fv-mesh`` volume store, a boundary store.

**Volume store** (custom geometry type ``zvcfd:fv-mesh``, like
``zvcfd:bricks``; spec in ``docs/spec/mesh_store.md``):

- ``vertices``: mesh nodes, float64, axes ``(z, y, x)`` as in every zvCFD
  store, chunked spatially by ``chunk_shape``;
- ``vertex_attributes/node``: each vertex's node number in the mesh (int64),
  so a reader restores the solver's node order;
- ``links/0``: element records of width 6 (the widest zarr-vectors'
  array partitioner takes). A tetrahedron or pyramid repeats its last node
  to fill the record, a wedge fills it exactly, and a hexahedron takes two
  records: nodes ``(0, 1, 2, 4, 5, 6)`` then ``(0, 2, 3, 4, 6, 7)``.
  Records whose nodes lie in several chunks are cross-chunk links, which
  zarr-vectors keeps explicitly;
- ``link_attributes/kind``: uint8 per record: 0 tet, 1 pyramid, 2 wedge,
  3 first and 4 second half of a hex;
- ``link_attributes/element``: int64 per record, the element's global index
  (types in :data:`zvcfd.mesh.core.KINDS` order). Links come back in chunk
  order, so this restores the mesh's element order and pairs hex halves.

**Boundary store**: a standard ZV ``mesh`` store of the boundary triangles,
one object per zone (zone vertices duplicated where zones meet), so any
Zarr Vectors viewer shows it. ``vertex_attributes/node`` maps back to mesh
nodes and ``vertex_attributes/zone`` gives each vertex's zone. The zone
table (id, kind, name) is kept in the ``zvcfd`` metadata
namespace. Per-wall-node fields such as wall shear stress are added as
further vertex attributes.

Node coordinates are double precision in both stores: the finite-volume
geometry is built from them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from zvcfd.mesh.core import KINDS, NODES_PER, BoundaryZone, UnstructuredMesh

GEOMETRY_TYPE = "zvcfd:fv-mesh"
AXES = [{"name": n, "type": "space"} for n in ("z", "y", "x")]
WIDTH = 6
HEX_A, HEX_B = [0, 1, 2, 4, 5, 6], [0, 2, 3, 4, 6, 7]


def _zb():
    from zarr_vectors import building

    return building


def _bounds(zyx: np.ndarray, chunk: float):
    lo = np.floor(zyx.min(0) / chunk) * chunk
    hi = np.ceil(zyx.max(0) / chunk) * chunk
    hi = np.where(hi <= zyx.max(0), hi + chunk, hi)
    return lo.tolist(), hi.tolist()


def _metadata(path, values: dict) -> None:
    import zarr_vectors as zv

    zv.open(str(path), mode="r+").metadata["zvcfd"].update(values)


def write_mesh_store(path, mesh: UnstructuredMesh, *, chunk: float,
                     compressor: Any = None) -> dict:
    """Write the volume store. ``chunk`` is the cubic chunk edge, in mesh units.

    Returns counts: nodes, elements, chunks, and elements crossing chunks.
    """
    zb = _zb()
    zyx = mesh.nodes[:, ::-1]
    bounds = _bounds(zyx, chunk)
    cshape = (float(chunk),) * 3
    root = zb.create_store(path, bounds=bounds, chunk_shape=cshape, axes=AXES,
                           geometry_types=[GEOMETRY_TYPE], vertex_dtype="float64",
                           unit=mesh.unit, compressor=compressor,
                           links_convention="explicit", cross_chunk_strategy="explicit")
    level = zb.create_resolution_level(root, 0, zb.LevelMetadata(
        level=0, vertex_count=mesh.n_nodes,
        arrays_present=["vertices", "vertex_attributes", "links", "link_attributes"]))
    assign = zb.assign_chunks(zyx, cshape)
    chunks = sorted(assign)
    vchunk, vlocal, chunks = zb.build_vertex_chunk_mapping(assign, mesh.n_nodes, chunks)
    # element records of WIDTH nodes
    recs, kind, elem = [], [], []
    off = mesh.element_offsets()
    for code, k in enumerate(KINDS):
        if k not in mesh.elements:
            continue
        e = mesh.elements[k]
        gid = off[k] + np.arange(len(e))
        if k == "hex":
            recs += [e[:, HEX_A], e[:, HEX_B]]
            kind += [np.full(len(e), 3, np.uint8), np.full(len(e), 4, np.uint8)]
            elem += [gid, gid]
            continue
        pad = np.repeat(e[:, -1:], WIDTH - NODES_PER[k], axis=1)
        recs.append(np.concatenate([e, pad], 1))
        kind.append(np.full(len(e), code, np.uint8))
        elem.append(gid)
    rec = np.concatenate(recs)
    kind = np.concatenate(kind)
    elem = np.concatenate(elem)
    link_chunks = np.asarray(chunks, np.int64)[vchunk[rec]]
    link_vi = vlocal[rec]
    with zb.open_write_session(level, compressor=compressor, bounds=bounds, chunk_shape=cshape):
        zb.create_vertices_array(level, dtype="float64")
        zb.create_attribute_array(level, "node", dtype="int64")
        for cc in chunks:
            gi = np.asarray(assign[cc], np.int64)
            zb.write_chunk_vertices(level, cc, [zyx[gi]], dtype="float64")
            zb.write_chunk_attributes(level, "node", cc, [gi], dtype="int64")
    part = zb.write_links(level, [], 3, link_width=WIDTH, _arrays=(link_chunks, link_vi))
    zb.write_link_attributes(level, "kind", kind, num_links=len(rec), partition=part)
    zb.write_link_attributes(level, "element", elem, num_links=len(rec), partition=part)
    crossing = int((~(vchunk[rec] == vchunk[rec][:, :1]).all(1)).sum())
    info = {"nodes": mesh.n_nodes, "elements": mesh.n_elements, "records": len(rec),
            "chunks": len(chunks),
            "crossing_elements": crossing}
    _metadata(path, {"kind": "fv-mesh", "unit": mesh.unit, "chunk": chunk,
                     "counts": mesh.counts(), "kinds": list(KINDS), "width": WIDTH,
                     "source": mesh.source})
    return info


def read_mesh_store(path) -> UnstructuredMesh:
    """Read a volume store back into an :class:`UnstructuredMesh` (no zones)."""
    zb = _zb()
    import zarr_vectors as zv

    meta = dict(zv.open(str(path)).metadata["zvcfd"])
    level = zb.get_resolution_level(zb.open_store(str(path)), 0)
    keys = [tuple(int(v) for v in k) for k in zb.list_chunk_keys(level, "vertices")]
    n = sum(meta["counts"].values())
    xyz, gids, start = [], [], {}
    total = 0
    for cc in keys:
        v = np.concatenate(zb.read_chunk_vertices(level, cc, dtype="float64")).reshape(-1, 3)
        g = np.concatenate(zb.read_chunk_attributes(level, "node", cc, dtype="int64")).reshape(-1)
        start[cc] = total
        xyz.append(v)
        gids.append(g)
        total += len(v)
    gids = np.concatenate(gids)
    nodes = np.zeros((total, 3))
    nodes[gids] = np.concatenate(xyz)[:, ::-1]
    chunks, vi = zb.read_link_arrays(level)
    kind = zb.read_link_attributes(level, "kind", dtype="uint8").reshape(-1)
    uc, inv = np.unique(chunks.reshape(-1, 3), axis=0, return_inverse=True)
    off = np.array([start[tuple(int(c) for c in row)] for row in uc], np.int64)
    glob = gids[off[inv.reshape(-1)] + vi.reshape(-1)].reshape(vi.shape)
    elem = zb.read_link_attributes(level, "element", dtype="int64").reshape(-1)
    order = np.lexsort((kind, elem))
    glob, kind, elem = glob[order], kind[order], elem[order]
    elements = {}
    for code, k in enumerate(KINDS[:3]):
        sel = kind == code
        if sel.any():
            elements[k] = glob[sel, :NODES_PER[k]]
    a, b = glob[kind == 3], glob[kind == 4]
    if len(a):
        elements["hex"] = np.stack([a[:, 0], a[:, 1], a[:, 2], b[:, 2], a[:, 3], a[:, 4],
                                    a[:, 5], b[:, 5]], 1)
    if sum(len(e) for e in elements.values()) != n:
        raise ValueError("element count differs from the store's metadata")
    return UnstructuredMesh(nodes, elements, unit=meta.get("unit"), source=str(path))


def write_boundary_store(path, mesh: UnstructuredMesh, *, chunk: float,
                         compressor: Any = None) -> dict:
    """Write the boundary triangles as a ZV ``mesh`` store, one object per zone."""
    from zarr_vectors.types.meshes import write_mesh

    verts, faces, obj, node, zid_v = [], [], [], [], []
    base = 0
    zones = []
    for i, (zid, z) in enumerate(sorted(mesh.zones.items())):
        tri = z.triangles()
        used, local = np.unique(tri, return_inverse=True)
        verts.append(mesh.nodes[used][:, ::-1])
        faces.append(local.reshape(-1, 3) + base)
        obj.append(np.full(len(used), i, np.int64))
        node.append(used)
        zid_v.append(np.full(len(used), zid, np.int64))
        base += len(used)
        zones.append({"object": i, "zone": int(zid), "kind": z.kind, "name": z.name,
                      "faces": int(z.n_faces)})
    zyx = np.concatenate(verts)
    bounds = _bounds(zyx, chunk)
    write_mesh(str(path), zyx, np.concatenate(faces), chunk_shape=(float(chunk),) * 3,
               bounds=bounds, dtype="float64", object_ids=np.concatenate(obj),
               vertex_attributes={"node": np.concatenate(node), "zone": np.concatenate(zid_v)},
               compressor=compressor)
    _metadata(path, {"kind": "fv-boundary", "unit": mesh.unit, "zones": zones})
    return {"vertices": int(len(zyx)), "triangles": int(sum(len(f) for f in faces)),
            "zones": len(zones)}


def read_boundary_store(path) -> dict:
    """``{"triangles": (T, 3) mesh nodes, "zone": (T,) zone ids, "zones": table}``."""
    import zarr_vectors as zv

    zb = _zb()
    meta = dict(zv.open(str(path)).metadata["zvcfd"])
    level = zb.get_resolution_level(zb.open_store(str(path)), 0)
    keys = [tuple(int(v) for v in k) for k in zb.list_chunk_keys(level, "vertices")]
    node, zone, start, total = [], [], {}, 0
    for cc in keys:
        g = np.concatenate(zb.read_chunk_attributes(level, "node", cc, dtype="int64")).reshape(-1)
        z = np.concatenate(zb.read_chunk_attributes(level, "zone", cc, dtype="int64")).reshape(-1)
        start[cc] = total
        node.append(g)
        zone.append(z)
        total += len(g)
    node, zone = np.concatenate(node), np.concatenate(zone)
    chunks, vi = zb.read_link_arrays(level)
    uc, inv = np.unique(chunks.reshape(-1, 3), axis=0, return_inverse=True)
    off = np.array([start[tuple(int(c) for c in row)] for row in uc], np.int64)
    v = off[inv.reshape(-1)].reshape(vi.shape) + vi
    return {"triangles": node[v], "zone": zone[v[:, 0]], "zones": meta["zones"]}


def zones_from_boundary(mesh: UnstructuredMesh, boundary: dict) -> dict[int, BoundaryZone]:
    """Rebuild boundary zones on ``mesh`` from :func:`read_boundary_store` output.

    Triangles of quadrilateral faces are merged back by matching the mesh's
    own boundary faces.
    """
    from zvcfd.mesh.fluent import match_rows

    fx = mesh.faces()
    b = fx["c1"] < 0
    faces, owner = fx["faces"][b], fx["c0"][b]
    pad = -np.ones((len(boundary["triangles"]), 1), np.int64)
    stored = np.concatenate([boundary["triangles"], pad], 1)
    zone = np.full(len(faces), -1)
    # a triangle matches itself; a quad matches through one of its four 3-node subsets
    for sub in ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)):
        cand = faces[:, list(sub)]
        cand = np.concatenate([cand, -np.ones((len(cand), 1), np.int64)], 1)
        if sub != (0, 1, 2):
            cand[faces[:, 3] < 0] = -2                     # triangles: first subset only
        hit = match_rows(stored, cand)
        new = (hit >= 0) & (zone < 0)
        zone[new] = boundary["zone"][hit[new]]
    if (zone < 0).any():
        raise ValueError(f"{int((zone < 0).sum())} boundary faces have no zone in the store")
    out = {}
    for z in boundary["zones"]:
        sel = zone == z["zone"]
        out[z["zone"]] = BoundaryZone(z["zone"], z["kind"], z["name"], faces[sel], owner[sel])
    return out


__all__ = ["GEOMETRY_TYPE", "read_boundary_store", "read_mesh_store", "write_boundary_store",
           "write_mesh_store", "zones_from_boundary"]
