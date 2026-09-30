"""Uniform conforming refinement of unstructured meshes, for grid-convergence studies.

Every element is split into eight (a pyramid into six pyramids and four
tetrahedra), halving every edge. New nodes sit at edge midpoints,
quadrilateral-face centroids and hexahedron centroids. They come from
global edge and face tables, so neighbouring elements share them and the
refined mesh is conforming. Boundary faces are split the same way and keep
their zones and outward orientation.

The refined mesh keeps the parent's geometry exactly: new boundary nodes
lie on the parent's (straight-edged) faces. That is the right thing for a
grid study of a *given* mesh, such as the coronary tree, whose faceted
surface is the geometry. A curved analytic wall would need projection.

- tetrahedron: four corner tetrahedra and the octahedron cut into four
  along one of its diagonals (Bey 1995);
- wedge: each triangle into four, and the prism into two layers;
- hexahedron: the 27-point subdivision;
- pyramid: four corner pyramids, a top pyramid, an inverted pyramid on
  the base centroid, and four tetrahedra between them.

Reference: J. Bey, *Tetrahedral grid refinement*, Computing 55, 355 (1995).
"""

from __future__ import annotations

import numpy as np

from zvcfd.mesh.core import BoundaryZone, UnstructuredMesh


class _Table:
    """New nodes keyed by the sorted set of parent nodes they average."""

    def __init__(self, nodes: np.ndarray):
        self.nodes = [nodes]
        self.next = len(nodes)
        self.maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def ids(self, groups: np.ndarray) -> np.ndarray:
        """Node ids for ``groups (M, k)`` of parent nodes (created on first use)."""
        k = groups.shape[1]
        key = np.sort(groups, axis=1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        if k not in self.maps:
            ids = np.arange(self.next, self.next + len(uniq))
            self.next += len(uniq)
            self.nodes.append(self.nodes[0][uniq].mean(1))
            self.maps[k] = (uniq, ids)
            return ids[inv]
        old, oid = self.maps[k]
        allk = np.concatenate([old, uniq])
        u2, i2 = np.unique(allk, axis=0, return_inverse=True)
        i2 = i2.reshape(-1)
        where = -np.ones(len(u2), np.int64)
        where[i2[:len(old)]] = oid
        new = where[i2[len(old):]] < 0
        fresh = np.arange(self.next, self.next + int(new.sum()))
        self.next += len(fresh)
        self.nodes.append(self.nodes[0][uniq[new]].mean(1))
        where[i2[len(old):][new]] = fresh
        self.maps[k] = (np.concatenate([old, uniq[new]]), np.concatenate([oid, fresh]))
        return where[i2[len(old):]][inv]

    def coordinates(self) -> np.ndarray:
        return np.concatenate(self.nodes)


def _tri4(a, b, c, ab, bc, ca):
    return [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]


def refine(mesh: UnstructuredMesh) -> UnstructuredMesh:
    """Split every element in eight (pyramids in ten), conformingly."""
    from zvcfd.mesh.generate import _orient

    T = _Table(mesh.nodes)

    def mid(e, i, j):
        return T.ids(np.stack([e[:, i], e[:, j]], 1))

    def quad(e, *c):
        return T.ids(np.stack([e[:, k] for k in c], 1))

    out: dict[str, list] = {"tet": [], "pyramid": [], "wedge": [], "hex": []}
    if "tet" in mesh.elements:
        e = mesh.elements["tet"]
        m = {(i, j): mid(e, i, j) for i in range(4) for j in range(i + 1, 4)}
        n = [e[:, i] for i in range(4)]
        g = lambda i, j: m[(min(i, j), max(i, j))]              # noqa: E731
        tets = [(n[0], g(0, 1), g(0, 2), g(0, 3)), (g(0, 1), n[1], g(1, 2), g(1, 3)),
                (g(0, 2), g(1, 2), n[2], g(2, 3)), (g(0, 3), g(1, 3), g(2, 3), n[3])]
        # octahedron cut along the m02 - m13 diagonal; equator m01 -> m03 -> m23 -> m12
        eq = [g(0, 1), g(0, 3), g(2, 3), g(1, 2)]
        for k in range(4):
            tets.append((g(0, 2), g(1, 3), eq[k], eq[(k + 1) % 4]))
        out["tet"] += [np.stack(t, 1) for t in tets]
    if "wedge" in mesh.elements:
        e = mesh.elements["wedge"]
        n = [e[:, i] for i in range(6)]
        bot = [n[0], n[1], n[2], mid(e, 0, 1), mid(e, 1, 2), mid(e, 2, 0)]
        top = [n[3], n[4], n[5], mid(e, 3, 4), mid(e, 4, 5), mid(e, 5, 3)]
        midl = [mid(e, 0, 3), mid(e, 1, 4), mid(e, 2, 5), quad(e, 0, 1, 4, 3),
                quad(e, 1, 2, 5, 4), quad(e, 2, 0, 3, 5)]
        for lo, hi in ((bot, midl), (midl, top)):
            for tri in _tri4(*range(6)[:3], 3, 4, 5):
                out["wedge"].append(np.stack([lo[i] for i in tri] + [hi[i] for i in tri], 1))
    if "hex" in mesh.elements:
        e = mesh.elements["hex"]
        corner = {(0, 0, 0): 0, (2, 0, 0): 1, (2, 2, 0): 2, (0, 2, 0): 3,
                  (0, 0, 2): 4, (2, 0, 2): 5, (2, 2, 2): 6, (0, 2, 2): 7}
        grid = {}
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    cs = [corner[(a, b, c)] for a in ({0, 2} if i == 1 else {i})
                          for b in ({0, 2} if j == 1 else {j}) for c in ({0, 2} if k == 1 else {k})]
                    if len(cs) == 1:
                        grid[(i, j, k)] = e[:, cs[0]]
                    elif len(cs) == 8:
                        grid[(i, j, k)] = None                        # body centre, below
                    else:
                        grid[(i, j, k)] = T.ids(np.stack([e[:, c] for c in cs], 1))
        body = np.arange(T.next, T.next + len(e))
        T.next += len(e)
        T.nodes.append(mesh.nodes[e].mean(1))
        grid[(1, 1, 1)] = body
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    loc = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1),
                           (1, 1, 1), (0, 1, 1)]
                    out["hex"].append(np.stack([grid[(a + p, b + q, c + r)] for p, q, r in loc],
                                               1))
    if "pyramid" in mesh.elements:
        e = mesh.elements["pyramid"]
        b = [e[:, i] for i in range(4)]
        apex = e[:, 4]
        mb = [mid(e, i, (i + 1) % 4) for i in range(4)]           # base edge midpoints
        ma = [mid(e, i, 4) for i in range(4)]                     # apex edge midpoints
        cb = quad(e, 0, 1, 2, 3)
        pyr, tet = [], []
        for i in range(4):
            pyr.append((b[i], mb[i], cb, mb[(i - 1) % 4], ma[i]))
            tet.append((mb[i], ma[i], ma[(i + 1) % 4], cb))
        pyr.append((ma[0], ma[1], ma[2], ma[3], apex))
        pyr.append((ma[0], ma[3], ma[2], ma[1], cb))
        out["pyramid"] += [np.stack(p, 1) for p in pyr]
        out["tet"] += [np.stack(t, 1) for t in tet]
    nodes = T.coordinates()
    elements = {}
    for k, parts in out.items():
        if parts:
            elements[k] = _orient_all(nodes, k, np.concatenate(parts), _orient)
    zones = {}
    for zid, z in mesh.zones.items():
        f = z.faces
        tri = f[:, 3] < 0
        sub = []
        t3 = f[tri, :3]
        if len(t3):
            ab, bc, ca = (T.ids(t3[:, [0, 1]]), T.ids(t3[:, [1, 2]]), T.ids(t3[:, [2, 0]]))
            for s in _tri4(t3[:, 0], t3[:, 1], t3[:, 2], ab, bc, ca):
                sub.append(np.concatenate([np.stack(s, 1), -np.ones((len(t3), 1), np.int64)], 1))
        q4 = f[~tri]
        if len(q4):
            a, bq, c, d = (q4[:, i] for i in range(4))
            ab, bc, cd, da = (T.ids(q4[:, [0, 1]]), T.ids(q4[:, [1, 2]]), T.ids(q4[:, [2, 3]]),
                              T.ids(q4[:, [3, 0]]))
            qc = T.ids(q4)
            for s in ((a, ab, qc, da), (ab, bq, bc, qc), (qc, bc, c, cd), (da, qc, cd, d)):
                sub.append(np.stack(s, 1))
        zones[zid] = BoundaryZone(z.zone, z.kind, z.name, np.concatenate(sub),
                                  np.zeros(0, np.int64))
    if len(T.nodes) and len(nodes) != T.next:
        raise RuntimeError("refinement node bookkeeping failed")
    fine = UnstructuredMesh(nodes, elements, zones, unit=mesh.unit, source=mesh.source,
                            meta={"refined_from": mesh.n_nodes})
    _owners(fine)
    return fine


def _orient_all(nodes, kind, e, orient):
    if kind == "pyramid":
        x = nodes[e]
        s = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 4] - x[:, 0])
        e[s < 0] = e[s < 0][:, [0, 3, 2, 1, 4]]
        return e
    return orient(nodes, kind, e)


def _owners(mesh: UnstructuredMesh) -> None:
    """Fill each boundary face's owning element from the mesh's own face table."""
    from zvcfd.mesh.fluent import match_rows

    fx = mesh.faces()
    b = fx["c1"] < 0
    faces, own = fx["faces"][b], fx["c0"][b]
    for z in mesh.zones.values():
        hit = match_rows(faces, z.faces)
        if (hit < 0).any():
            raise RuntimeError(f"zone {z.name}: refined faces are not on the boundary")
        z.cells = own[hit]


__all__ = ["refine"]
