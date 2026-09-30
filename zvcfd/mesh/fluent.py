"""Ansys Fluent ``.msh`` volume meshes: full reader and writer.

:func:`read_fluent_mesh` reads nodes, every face zone (interior faces
included) and the zone table, then rebuilds each cell's element from its
faces. Fluent stores, for each face, its nodes and the two cells it
separates, not each cell's nodes. Cells are classified by their triangle
and quadrilateral face counts:

=========  =====  =====
element    tri    quad
=========  =====  =====
tet        4      0
pyramid    4      1
wedge      2      3
hex        0      6
=========  =====  =====

The zone header's declared element type is ignored (Simpleware writes
``7``, polyhedral, for a tetrahedron–prism mesh). Nodes are then ordered
to :mod:`zvcfd.mesh.core`'s local orders with positive volumes, and
boundary faces are turned to point out of the fluid. Both are done from
the geometry, so the reader does not rely on the file's orientation
convention. It measures the convention instead, as
``mesh.meta["c0_side"]``: the fraction of interior faces whose
right-hand-rule normal points into ``c0``.

ASCII sections (``10``, ``12``, ``13``) are parsed in bulk with numpy:
hexadecimal tokens are decoded through a lookup table, in blocks of
``block`` bytes. Binary sections (``2010``/``3010`` nodes in single/double
precision, ``2012``/``3012`` cells and ``2013``/``3013`` faces as int32)
are read in place. A mixed-type binary face section is walked by pointer
doubling.

:func:`write_fluent_mesh` writes an :class:`~zvcfd.mesh.core.UnstructuredMesh`
back out (ASCII or binary), so generated validation meshes can go to other
codes. It follows the convention measured above (normal into ``c0``;
``c1 = 0`` on boundary faces).

The format is described in the Fluent User's Guide, appendix *Case and
Data File Formats* (mesh sections).
"""

from __future__ import annotations

import mmap
import re
import time
import warnings
from pathlib import Path

import numpy as np

from zvcfd.mesh.core import FLUENT_TYPE, KINDS, BoundaryZone, UnstructuredMesh

_HEADER = re.compile(rb"(?m)^\((10|12|13|39|45|2010|3010|2012|3012|2013|3013) \(([^()]*)\)")
BC_KIND = {2: "interior", 3: "wall", 4: "pressure-inlet", 5: "pressure-outlet", 7: "symmetry",
           8: "periodic-shadow", 9: "pressure-far-field", 10: "velocity-inlet", 12: "periodic",
           14: "fan", 20: "mass-flow-inlet", 24: "interface", 31: "parent", 36: "outflow",
           37: "axis"}
KIND_BC = {v: k for k, v in BC_KIND.items()}

_HEX = np.full(256, -1, np.int8)
for _i, _c in enumerate(b"0123456789"):
    _HEX[_c] = _i
for _i, _c in enumerate(b"abcdef"):
    _HEX[_c] = 10 + _i
    _HEX[_c - 32] = 10 + _i


# ---------------------------------------------------------------- tokenising

def _hex_tokens(a: np.ndarray):
    """Hex tokens of a uint8 buffer: ``(values int64, start positions int64)``."""
    d = _HEX[a]
    dig = (d >= 0).view(np.int8)
    edge = np.diff(dig, prepend=np.int8(0), append=np.int8(0))
    st = np.flatnonzero(edge == 1)
    ln = np.flatnonzero(edge == -1) - st
    vals = np.zeros(len(st), np.int64)
    if len(st) and ln.max() > 15:
        raise ValueError("hex token longer than 15 digits")
    for k in range(int(ln.max(initial=0))):
        m = ln > k
        vals[m] = vals[m] * 16 + d[st[m] + k]
    return vals, st


def _face_block(a: np.ndarray, ftype: int):
    """Parse ASCII face records in ``a`` (whole lines): ``(nodes (F, 4), c0, c1)``, 1-based."""
    vals, st = _hex_tokens(a)
    nl = np.flatnonzero(a == 10)
    line = np.searchsorted(nl, st)
    first = np.flatnonzero(np.r_[True, line[1:] != line[:-1]])
    count = np.diff(np.r_[first, len(vals)])
    if ftype in (0, 5):
        n = vals[first]
        if (count != n + 3).any():
            raise ValueError("malformed face record")
        node0 = first + 1
    else:
        n = np.full(len(first), ftype)
        if (count != ftype + 2).any():
            raise ValueError("malformed face record")
        node0 = first
    if (n > 4).any() or (n < 3).any():
        raise NotImplementedError("polygonal faces (more than 4 nodes) are not supported")
    nodes = np.zeros((len(first), 4), np.int64)
    for j in range(4):
        m = n > j
        nodes[m, j] = vals[node0[m] + j]
    c0 = vals[node0 + n]
    c1 = vals[node0 + n + 1]
    return nodes, c0, c1


def _parse_faces_ascii(mm, start: int, end: int, ftype: int, block: int):
    parts = []
    pos = start
    while pos < end:
        stop = min(pos + block, end)
        if stop < end:
            nl = mm.find(b"\n", stop, end)
            stop = end if nl < 0 else nl + 1
        a = np.frombuffer(mm, np.uint8, count=stop - pos, offset=pos)
        parts.append(_face_block(a, ftype))
        pos = stop
    nodes = np.concatenate([p[0] for p in parts])
    return nodes, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])


def _parse_faces_binary(mm, start: int, nface: int, ftype: int):
    """Binary face records; returns ``(nodes, c0, c1, bytes consumed)``."""
    avail = (len(mm) - start) // 4
    if ftype not in (0, 5):
        k = ftype + 2
        ints = np.frombuffer(mm, "<i4", count=nface * k, offset=start).astype(np.int64)
        rec = ints.reshape(nface, k)
        nodes = np.zeros((nface, 4), np.int64)
        nodes[:, :ftype] = rec[:, :ftype]
        return nodes, rec[:, ftype], rec[:, ftype + 1], nface * k * 4
    ints = np.frombuffer(mm, "<i4", count=min(avail, nface * 7), offset=start).astype(np.int64)
    L = len(ints)
    nxt = np.minimum(np.arange(L) + ints + 3, L)          # next record if a record starts here
    pos = np.zeros(1, np.int64)
    jump = np.append(nxt, L)
    while len(pos) < nface:                                # pointer doubling
        new = jump[pos]
        pos = np.concatenate([pos, new])
        jump = jump[jump]
    pos = pos[:nface]
    if (pos >= L).any():
        raise ValueError("binary face section shorter than declared")
    n = ints[pos]
    if (n > 4).any() or (n < 3).any():
        raise NotImplementedError("polygonal faces (more than 4 nodes) are not supported")
    nodes = np.zeros((nface, 4), np.int64)
    for j in range(4):
        m = n > j
        nodes[m, j] = ints[pos[m] + 1 + j]
    c0, c1 = ints[pos + 1 + n], ints[pos + 2 + n]
    used = int(pos[-1] + n[-1] + 3)
    return nodes, c0, c1, used * 4


def _body(mm, after_header: int):
    """Position of a section's body ``(`` after its header, or None if it has none."""
    i = after_header
    while i < len(mm) and mm[i:i + 1] in b" \t\r\n":
        i += 1
    return i if mm[i:i + 1] == b"(" else None


# ---------------------------------------------------------------- reader

def read_fluent_mesh(path, *, unit: str | None = None, block: int = 1 << 26,
                     log=None) -> UnstructuredMesh:
    """Read a 3-D Fluent mesh into an :class:`UnstructuredMesh`.

    Args:
        path: ``.msh`` file (ASCII or binary sections, or a mix).
        unit: coordinate unit to record (Fluent meshes do not carry one).
        block: bytes per ASCII parsing block (memory ~ 20× this).
        log: optional ``print``-like callable for progress.
    """
    t0 = time.time()
    say = log or (lambda *a: None)
    path = str(path)
    nodes_parts: list[tuple[int, np.ndarray]] = []
    n_nodes = n_cells = n_faces = 0
    face_zones: list[dict] = []
    names: dict[int, tuple[str, str]] = {}
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        pos = 0
        while True:
            m = _HEADER.search(mm, pos)
            if m is None:
                break
            sec = int(m.group(1))
            head = m.group(2).split()
            pos = m.end()
            if sec in (39, 45):
                names[int(head[0])] = (head[1].decode(), head[2].decode() if len(head) > 2 else "")
                continue
            vals = [int(h, 16) for h in head]
            zone, first, last = vals[0], vals[1], vals[2]
            count = last - first + 1
            body = _body(mm, pos)
            binary = sec >= 2000
            base = sec % 1000 if binary else sec
            if zone == 0 or body is None:
                if zone == 0:
                    if base == 10:
                        n_nodes = count
                    elif base == 12:
                        n_cells = count
                    elif base == 13:
                        n_faces = count
                continue
            start = body + 1
            if base == 10:
                nd = vals[4] if len(vals) > 4 else 3
                if nd != 3:
                    raise ValueError("only 3-D meshes are supported")
                if binary:
                    dt = "<f8" if sec >= 3000 else "<f4"
                    xyz = np.frombuffer(mm, dt, count=count * 3, offset=start).astype(np.float64)
                    pos = start + count * 3 * np.dtype(dt).itemsize
                else:
                    end = mm.find(b")", start)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        xyz = np.fromstring(mm[start:end], sep=" ")
                    pos = end + 1
                nodes_parts.append((first, xyz.reshape(count, 3)))
            elif base == 12:
                if binary:
                    pos = start + count * 4
                else:
                    pos = mm.find(b")", start) + 1
            elif base == 13:
                bc, ftype = vals[3], (vals[4] if len(vals) > 4 else 0)
                tz = time.time()
                if binary:
                    nodes, c0, c1, used = _parse_faces_binary(mm, start, count, ftype)
                    pos = start + used
                else:
                    end = mm.find(b")", start)
                    nodes, c0, c1 = _parse_faces_ascii(mm, start, end, ftype, block)
                    pos = end + 1
                if len(nodes) != count:
                    raise ValueError(f"face zone {zone}: {len(nodes)} faces, header says {count}")
                face_zones.append({"zone": zone, "first": first, "bc": bc, "nodes": nodes,
                                   "c0": c0, "c1": c1})
                say(f"  face zone {zone}: {count:,} faces ({time.time() - tz:.1f} s)")
    t_parse = time.time() - t0

    # nodes, 0-based
    if not n_nodes:
        n_nodes = max(f + len(x) - 1 for f, x in nodes_parts)
    xyz = np.zeros((n_nodes, 3))
    for f, x in nodes_parts:
        xyz[f - 1:f - 1 + len(x)] = x
    # all faces, 0-based, in file order
    face_zones.sort(key=lambda z: z["first"])
    fnodes = np.concatenate([z["nodes"] for z in face_zones]) - 1       # pads become -1
    c0 = np.concatenate([z["c0"] for z in face_zones]) - 1
    c1 = np.concatenate([z["c1"] for z in face_zones]) - 1
    fzone = np.concatenate([np.full(len(z["nodes"]), z["zone"]) for z in face_zones])
    if n_faces and len(fnodes) != n_faces:
        raise ValueError(f"read {len(fnodes)} faces, the file declares {n_faces}")
    if not n_cells:
        n_cells = int(max(c0.max(), c1.max()) + 1)

    t1 = time.time()
    elements, cell_ids, cell_elem = _rebuild_elements(xyz, fnodes, c0, c1, n_cells)
    t_elem = time.time() - t1
    mesh = UnstructuredMesh(xyz, elements, cell_ids=cell_ids, unit=unit, source=path)

    # orientation convention of the file, measured on interior faces
    cent = mesh.all_centroids()
    interior = (c0 >= 0) & (c1 >= 0)
    sample = np.flatnonzero(interior)[:: max(1, int(interior.sum()) // 200_000)]
    nrm, fc = _normals(xyz, fnodes[sample])
    c0_side = float(np.mean(np.einsum("ij,ij->i", nrm, cent[cell_elem[c0[sample]]] - fc) > 0))

    # boundary zones, outward
    for z in face_zones:
        if z["bc"] == 2:
            continue
        sel = fzone == z["zone"]
        f = fnodes[sel]
        own = np.where(c0[sel] >= 0, c0[sel], c1[sel])
        elem = cell_elem[own]
        nrm, fc = _normals(xyz, f)
        flip = np.einsum("ij,ij->i", nrm, fc - cent[elem]) < 0
        f[flip] = _reverse(f[flip])
        kind, name = names.get(z["zone"], (BC_KIND.get(z["bc"], str(z["bc"])), ""))
        mesh.zones[z["zone"]] = BoundaryZone(z["zone"], kind, name, f, elem)
    mesh.meta.update({"c0_side": c0_side, "faces": int(len(fnodes)),
                      "interior_faces": int(interior.sum()), "declared_cells": n_cells,
                      "parse_s": t_parse, "elements_s": t_elem,
                      "total_s": time.time() - t0})
    return mesh


def _normals(xyz: np.ndarray, f: np.ndarray):
    """Area vectors (right-hand rule) and centroids of padded polygons ``(F, 4)``."""
    tri = f[:, 3] < 0
    p = xyz[np.where(f < 0, f[:, :1], f)]
    n = np.where(tri[:, None], 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                 0.5 * np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]))
    c = np.where(tri[:, None], p[:, :3].mean(1), p.mean(1))
    return n, c


def _reverse(f: np.ndarray) -> np.ndarray:
    out = f.copy()
    tri = f[:, 3] < 0
    out[tri, :3] = f[tri][:, [0, 2, 1]]
    out[~tri] = f[~tri][:, [0, 3, 2, 1]]
    return out


def _signed(xyz, a, b, c, d):
    pa = xyz[a]
    return np.einsum("ij,ij->i", np.cross(xyz[b] - pa, xyz[c] - pa), xyz[d] - pa)


def _partners(bottom: np.ndarray, polys: np.ndarray) -> np.ndarray:
    """For each bottom node, its partner across an edge of ``polys`` (quads) leaving the bottom."""
    eu = polys.reshape(len(polys), -1)
    ev = np.roll(polys, -1, axis=2).reshape(len(polys), -1)
    inb_u = (eu[:, :, None] == bottom[:, None, :]).any(2)
    inb_v = (ev[:, :, None] == bottom[:, None, :]).any(2)
    out = np.empty_like(bottom)
    for s in range(bottom.shape[1]):
        b = bottom[:, s:s + 1]
        cv = np.where((eu == b) & ~inb_v, ev, -1).max(1)
        cu = np.where((ev == b) & ~inb_u, eu, -1).max(1)
        out[:, s] = np.maximum(cv, cu)
    if (out < 0).any():
        raise ValueError("could not match the two caps of a wedge or hexahedron")
    return out


def _rebuild_elements(xyz, fnodes, c0, c1, n_cells):
    """Elements from face–cell incidence: ``(elements, cell_ids, cell -> global element)``."""
    nv = np.where(fnodes[:, 3] < 0, 3, 4)
    fid = np.arange(len(fnodes))
    cell = np.concatenate([c0[c0 >= 0], c1[c1 >= 0]])
    face = np.concatenate([fid[c0 >= 0], fid[c1 >= 0]])
    order = np.lexsort((nv[face], cell))
    cell, face = cell[order], face[order]
    start = np.searchsorted(cell, np.arange(n_cells))
    ntri = np.bincount(cell, weights=(nv[face] == 3), minlength=n_cells).astype(int)
    nquad = np.bincount(cell, weights=(nv[face] == 4), minlength=n_cells).astype(int)
    kind_of = {(4, 0): "tet", (4, 1): "pyramid", (2, 3): "wedge", (0, 6): "hex"}
    code = np.full(n_cells, -1)
    for k, (t, q) in enumerate(kind_of):
        code[(ntri == t) & (nquad == q)] = k
    if (code < 0).any():
        bad = np.unique(np.stack([ntri[code < 0], nquad[code < 0]], 1), axis=0)
        raise NotImplementedError(f"{int((code < 0).sum())} cells are not tetrahedra, pyramids, "
                                  f"wedges or hexahedra (tri, quad face counts: {bad.tolist()})")
    elements, cell_ids = {}, {}
    for k, kind in enumerate(kind_of.values()):
        cells = np.flatnonzero(code == k)
        if not len(cells):
            continue
        nf = {"tet": 4, "pyramid": 5, "wedge": 5, "hex": 6}[kind]
        F = face[start[cells][:, None] + np.arange(nf)]
        P = fnodes[F]
        if kind == "tet":
            base = P[:, 0, :3]
            other = P[:, 1, :3]
            apex = np.where((other[:, :, None] != base[:, None, :]).all(2), other, -1).max(1)
            e = np.concatenate([base, apex[:, None]], 1)
            neg = _signed(xyz, e[:, 0], e[:, 1], e[:, 2], e[:, 3]) < 0
            e[neg] = e[neg][:, [0, 2, 1, 3]]
        elif kind == "pyramid":
            base = P[:, 4, :4]
            tri = P[:, 0, :3]
            apex = np.where((tri[:, :, None] != base[:, None, :]).all(2), tri, -1).max(1)
            e = np.concatenate([base, apex[:, None]], 1)
            neg = _signed(xyz, e[:, 0], e[:, 1], e[:, 2], e[:, 4]) < 0
            e[neg] = e[neg][:, [0, 3, 2, 1, 4]]
        elif kind == "wedge":
            bottom = P[:, 0, :3]
            top = _partners(bottom, P[:, 2:5, :4])
            neg = _signed(xyz, bottom[:, 0], bottom[:, 1], bottom[:, 2], top[:, 0]) < 0
            bottom[neg] = bottom[neg][:, [0, 2, 1]]
            top[neg] = top[neg][:, [0, 2, 1]]
            e = np.concatenate([bottom, top], 1)
        else:
            bottom = P[:, 0, :4]
            top = _partners(bottom, P[:, 1:6, :4])
            neg = _signed(xyz, bottom[:, 0], bottom[:, 1], bottom[:, 3], top[:, 0]) < 0
            bottom[neg] = bottom[neg][:, [0, 3, 2, 1]]
            top[neg] = top[neg][:, [0, 3, 2, 1]]
            e = np.concatenate([bottom, top], 1)
        elements[kind] = e
        cell_ids[kind] = cells
    # global element index of each cell (KINDS order, matching UnstructuredMesh)
    cell_elem = np.empty(n_cells, np.int64)
    off = 0
    for kind in KINDS:
        if kind in cell_ids:
            cell_elem[cell_ids[kind]] = off + np.arange(len(cell_ids[kind]))
            off += len(cell_ids[kind])
    return elements, cell_ids, cell_elem


# ---------------------------------------------------------------- writer

def write_fluent_mesh(mesh: UnstructuredMesh, path, *, binary: bool = False,
                      cell_zone: str = "fluid") -> dict:
    """Write ``mesh`` as a Fluent ``.msh`` (3-D, one cell zone).

    Interior faces go to zone 2; boundary zones keep their ids (renumbered
    from 3 if they clash) and their kinds. Faces are written with the
    right-hand-rule normal pointing into ``c0``; boundary faces have
    ``c1 = 0``. Returns the zone ids used.
    """
    fx = mesh.faces()
    faces, c0, c1 = fx["faces"], fx["c0"], fx["c1"]
    # faces() gives normals out of c0: flip so they point into c0, as Fluent files do
    inner = c1 >= 0
    f_int = _reverse(faces[inner])
    fc0, fc1 = c0[inner], c1[inner]
    bnd = np.flatnonzero(~inner)
    zone_ids, zone_faces, used = {}, [], set()
    nxt = 3
    found = np.zeros(len(bnd), bool)
    for zid, z in mesh.zones.items():
        hit = match_rows(faces[bnd], z.faces)
        if (hit < 0).any():
            raise ValueError(f"zone {zid}: {int((hit < 0).sum())} faces are not on the boundary")
        found[hit] = True
        new = zid if zid >= 3 and zid not in used else None
        while new is None or new in used:
            new, nxt = nxt, nxt + 1
        used.add(new)
        zone_ids[zid] = new
        zone_faces.append((new, z.kind, z.name or z.kind, bnd[hit]))
    if not found.all():
        raise ValueError(f"{int((~found).sum())} boundary faces belong to no zone")

    ctype = np.concatenate([np.full(len(mesh.elements[k]), FLUENT_TYPE[k])
                            for k in KINDS if k in mesh.elements])
    n_face = len(faces)
    with open(path, "wb") as out:
        w = out.write
        w(b'(0 "zvCFD Fluent mesh")\n(2 3)\n')
        w(f"(10 (0 1 {mesh.n_nodes:x} 0 3))\n".encode())
        w(f"(12 (0 1 {mesh.n_elements:x} 0))\n".encode())
        w(f"(13 (0 1 {n_face:x} 0))\n".encode())
        if binary:
            w(f"(3010 (1 1 {mesh.n_nodes:x} 1 3)(".encode())
            w(mesh.nodes.astype("<f8").tobytes())
            w(b")\nEnd of Binary Section 3010)\n")
            w(f"(2012 (1 1 {mesh.n_elements:x} 1 0)(".encode())
            w(ctype.astype("<i4").tobytes())
            w(b")\nEnd of Binary Section 2012)\n")
        else:
            w(f"(10 (1 1 {mesh.n_nodes:x} 1 3)(\n".encode())
            np.savetxt(out, mesh.nodes, fmt="%.16e")
            w(b"))\n")
            w(f"(12 (1 1 {mesh.n_elements:x} 1 0)(\n".encode())
            _write_hex_lines(out, ctype[:, None])
            w(b"))\n")
        first = 1
        sections = [(2, "interior", f_int, fc0, fc1)]
        for new, kind, name, idx in zone_faces:
            sections.append((new, kind, _reverse(faces[idx]), c0[idx], -np.ones(len(idx), int)))
        for zid, kind, f, a, b in sections:
            last = first + len(f) - 1
            bc = KIND_BC.get(kind, 3)
            nv = np.where(f[:, 3] < 0, 3, 4)
            rows = [nv, f[:, 0] + 1, f[:, 1] + 1, f[:, 2] + 1, f[:, 3] + 1, a + 1, b + 1]
            if binary:
                w(f"(2013 ({zid:x} {first:x} {last:x} {bc:x} 0)(".encode())
                tri = nv == 3
                rec = np.stack(rows, 1).astype("<i4")
                rec3 = np.delete(rec, 4, axis=1)                  # drop the pad for triangles
                buf = np.empty(int(tri.sum()) * 6 + int((~tri).sum()) * 7, "<i4")
                size = np.where(tri, 6, 7)
                offs = np.r_[0, np.cumsum(size)[:-1]]
                for j in range(6):
                    buf[offs[tri] + j] = rec3[tri, j]
                for j in range(7):
                    buf[offs[~tri] + j] = rec[~tri, j]
                w(buf.tobytes())
                w(b")\nEnd of Binary Section 2013)\n")
            else:
                w(f"(13 ({zid:x} {first:x} {last:x} {bc:x} 0)(\n".encode())
                rec = np.stack(rows, 1)
                tri = nv == 3
                lines = np.empty(len(f), object)
                for sel, cols in ((tri, [0, 1, 2, 3, 5, 6]), (~tri, [0, 1, 2, 3, 4, 5, 6])):
                    if sel.any():
                        lines[sel] = _hex_rows(rec[sel][:, cols])
                w(("\n".join(lines) + "\n").encode())
                w(b"))\n")
            first = last + 1
        w(f"(45 (1 fluid {cell_zone})())\n".encode())
        w(b"(45 (2 interior interior-1)())\n")
        for new, kind, name, _ in zone_faces:
            w(f"(45 ({new} {kind} {name})())\n".encode())
    return {"zones": zone_ids, "faces": n_face}


def match_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each padded polygon of ``b``, the index of the polygon of ``a`` with the same
    nodes (any order), or -1."""
    ka, kb = np.sort(a, axis=1), np.sort(b, axis=1)
    _, inv = np.unique(np.concatenate([ka, kb]), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    where = -np.ones(inv.max() + 1, np.int64)
    where[inv[:len(a)]] = np.arange(len(a))
    return where[inv[len(a):]]


def _hex_rows(a: np.ndarray) -> list[str]:
    return [" ".join(format(int(v), "x") for v in row) for row in a]


def _write_hex_lines(out, a: np.ndarray) -> None:
    out.write(("\n".join(_hex_rows(a)) + "\n").encode())


def mesh_info(path) -> dict:
    """Read a Fluent mesh and summarise it (counts, volume, zones, timings)."""
    mesh = read_fluent_mesh(path)
    s = mesh.summary()
    s["meta"] = mesh.meta
    s["file_bytes"] = Path(path).stat().st_size
    return s


__all__ = ["BC_KIND", "match_rows", "mesh_info", "read_fluent_mesh", "write_fluent_mesh"]
