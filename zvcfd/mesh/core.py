"""Unstructured meshes: nodes, elements by type, boundary zones.

Conventions (used by every reader, generator and the finite-volume geometry):

- Coordinates are ``(x, y, z)`` in the file's length unit (``mesh.unit``).
  Voxel code uses ``(z, y, x)``; conversion happens where the two meet.
- Elements are stored per type, as node-index arrays in the local order
  below. Every element has **positive volume**: readers and generators
  reorder nodes to make it so.
- ``FACES[kind]`` lists each element's faces with node order giving an
  **outward** normal by the right-hand rule. ``EDGES[kind]`` lists its
  edges. Boundary-zone faces are stored with outward normals too
  (pointing out of the fluid).

Local node order::

    tet      0 1 2 3                  (x1-x0) x (x2-x0) . (x3-x0) > 0
    pyramid  0 1 2 3 base, 4 apex     base counter-clockwise seen from the apex
    wedge    0 1 2 bottom, 3 4 5 top  3, 4, 5 above 0, 1, 2; bottom counter-clockwise from above
    hex      0 1 2 3 bottom, 4 5 6 7  4..7 above 0..3; bottom counter-clockwise from above

These are VTK's orders for all four types (checked against VTK's own cell
volumes in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KINDS = ("tet", "pyramid", "wedge", "hex")
NODES_PER = {"tet": 4, "pyramid": 5, "wedge": 6, "hex": 8}

FACES: dict[str, tuple[tuple[int, ...], ...]] = {
    "tet": ((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)),
    "pyramid": ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
    "wedge": ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    "hex": ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6),
            (3, 0, 4, 7)),
}
EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "tet": ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
    "pyramid": ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)),
    "wedge": ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)),
    "hex": ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
            (2, 6), (3, 7)),
}
# Fluent / VTK cell-type codes
FLUENT_TYPE = {"tet": 2, "hex": 4, "pyramid": 5, "wedge": 6}
VTK_TYPE = {"tet": 10, "hex": 12, "wedge": 13, "pyramid": 14}


@dataclass
class BoundaryZone:
    """One boundary zone: its faces (outward), and the element each face bounds.

    ``faces`` is ``(F, 4)`` node indices with ``-1`` in the last column of
    triangles. ``cells`` is the owning element's global index (see
    :meth:`UnstructuredMesh.element_offsets`).
    """

    zone: int
    kind: str
    name: str
    faces: np.ndarray
    cells: np.ndarray

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def triangles(self) -> np.ndarray:
        """Triangles ``(T, 3)``: quads split along their 0–2 diagonal."""
        tri = self.faces[:, 3] < 0
        q = self.faces[~tri]
        return np.concatenate([self.faces[tri, :3], q[:, [0, 1, 2]], q[:, [0, 2, 3]]])


@dataclass
class UnstructuredMesh:
    """Nodes, elements by type and boundary zones of a volume mesh.

    Attributes:
        nodes: ``(N, 3)`` float64, ``(x, y, z)`` in ``unit``.
        elements: ``{kind: (E, n) int64}`` in the local orders of this module.
        zones: ``{zone id: BoundaryZone}``.
        cell_ids: ``{kind: (E,) int64}`` the source file's 0-based cell index of
            each element (a reader fills it; generators leave the global order).
        unit: coordinate length unit (``"m"``, ``"mm"``, ...), if known.
    """

    nodes: np.ndarray
    elements: dict[str, np.ndarray]
    zones: dict[int, BoundaryZone] = field(default_factory=dict)
    cell_ids: dict[str, np.ndarray] = field(default_factory=dict)
    unit: str | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)       # reader diagnostics, timings

    def __post_init__(self):
        self.nodes = np.ascontiguousarray(self.nodes, np.float64)
        self.elements = {k: np.ascontiguousarray(self.elements[k], np.int64)
                         for k in KINDS if k in self.elements and len(self.elements[k])}
        for k in self.elements:
            if self.cell_ids.get(k) is None:
                off = self.element_offsets()[k]
                self.cell_ids[k] = np.arange(off, off + len(self.elements[k]))

    # ------------------------------------------------------------ counts

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return sum(len(e) for e in self.elements.values())

    def counts(self) -> dict[str, int]:
        return {k: len(self.elements[k]) for k in KINDS if k in self.elements}

    def element_offsets(self) -> dict[str, int]:
        """Global element index of each type's first element (types in ``KINDS`` order)."""
        off, out = 0, {}
        for k in KINDS:
            out[k] = off
            off += len(self.elements.get(k, ()))
        return out

    # ------------------------------------------------------------ geometry

    def element_volumes(self, kind: str) -> np.ndarray:
        """Volumes of one element type, from its faces fanned about their centroids.

        Quadrilateral faces are split into four triangles about the mean of
        their nodes, the same surface the control-volume geometry uses, so
        control volumes add up to these exactly.
        """
        return _fan_volumes(self.nodes, self.elements[kind], FACES[kind])

    def volume(self) -> float:
        return float(sum(self.element_volumes(k).sum() for k in self.elements))

    def edges(self) -> np.ndarray:
        """Unique mesh edges ``(M, 2)``, ``i < j``."""
        pairs = [e[:, list(EDGES[k])].reshape(-1, 2) for k, e in self.elements.items()]
        p = np.sort(np.concatenate(pairs), axis=1)
        key = np.unique(p[:, 0] * self.n_nodes + p[:, 1])
        return np.stack([key // self.n_nodes, key % self.n_nodes], 1)

    def faces(self) -> dict[str, np.ndarray]:
        """Every face once: interior faces shared by two elements, boundary faces by one.

        Returns ``faces (F, 4)`` (outward from ``c0``, ``-1`` pads triangles),
        ``c0`` and ``c1`` (global element indices, ``c1 = -1`` on the boundary).
        Raises if a face is shared by more than two elements (non-manifold) or if
        two elements share a face with the same orientation (inverted element).
        """
        polys, owner = [], []
        off = self.element_offsets()
        for k, e in self.elements.items():
            for f in FACES[k]:
                p = e[:, list(f)]
                if len(f) == 3:
                    p = np.concatenate([p, -np.ones((len(e), 1), np.int64)], 1)
                polys.append(p)
                owner.append(off[k] + np.arange(len(e)))
        polys = np.concatenate(polys)
        owner = np.concatenate(owner)
        key = np.sort(polys, axis=1)
        order = np.lexsort(key.T[::-1])
        ks = key[order]
        new = np.r_[True, (ks[1:] != ks[:-1]).any(1)]
        start = np.flatnonzero(new)
        size = np.diff(np.r_[start, len(ks)])
        if size.max(initial=0) > 2:
            raise ValueError(f"{int((size > 2).sum())} faces are shared by more than two elements")
        first = order[start]
        second = np.where(size == 2, order[np.minimum(start + 1, len(order) - 1)], -1)
        pair = second >= 0
        # a shared face must appear once in each orientation
        a, b = polys[first[pair]], polys[second[pair]]
        if len(a) and not _opposite(a, b).all():
            raise ValueError("elements sharing a face disagree on its orientation "
                             "(an inverted element)")
        return {"faces": polys[first], "c0": owner[first],
                "c1": np.where(pair, owner[second.clip(0)], -1)}

    def element_centroids(self, kind: str) -> np.ndarray:
        return self.nodes[self.elements[kind]].mean(1)

    def all_centroids(self) -> np.ndarray:
        """Centroids of every element, in global element order."""
        return np.concatenate([self.element_centroids(k) for k in KINDS if k in self.elements])

    def transformed(self, R: np.ndarray, t=(0.0, 0.0, 0.0)) -> UnstructuredMesh:
        """The mesh moved to ``x' = R x + t`` (``R`` orthogonal, rotation or reflection).

        A reflection turns every element inside out, so element node orders and
        boundary faces are mirrored back to positive volumes and outward normals.
        """
        R = np.asarray(R, float)
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-12):
            raise ValueError("R must be orthogonal")
        nodes = self.nodes @ R.T + np.asarray(t, float)
        flip = np.linalg.det(R) < 0
        mirror = {"tet": [0, 2, 1, 3], "pyramid": [0, 3, 2, 1, 4], "wedge": [0, 2, 1, 3, 5, 4],
                  "hex": [0, 3, 2, 1, 4, 7, 6, 5]}
        elements = {k: (e[:, mirror[k]] if flip else e.copy()) for k, e in self.elements.items()}
        zones = {}
        for zid, z in self.zones.items():
            f = z.faces.copy()
            if flip:
                tri = f[:, 3] < 0
                f[tri, :3] = f[tri][:, [0, 2, 1]]
                f[~tri] = f[~tri][:, [0, 3, 2, 1]]
            zones[zid] = BoundaryZone(z.zone, z.kind, z.name, f, z.cells)
        return UnstructuredMesh(nodes, elements, zones, dict(self.cell_ids), self.unit,
                                self.source, dict(self.meta))

    def renumbered(self, order: np.ndarray) -> UnstructuredMesh:
        """The same mesh with nodes reordered: new node ``k`` is old node ``order[k]``."""
        order = np.asarray(order, np.int64)
        new = np.empty(self.n_nodes, np.int64)
        new[order] = np.arange(self.n_nodes)
        zones = {z: BoundaryZone(b.zone, b.kind, b.name,
                                 np.where(b.faces >= 0, new[np.maximum(b.faces, 0)], -1), b.cells)
                 for z, b in self.zones.items()}
        return UnstructuredMesh(self.nodes[order], {k: new[e] for k, e in self.elements.items()},
                                zones, dict(self.cell_ids), self.unit, self.source,
                                dict(self.meta, node_order="renumbered"))

    # ------------------------------------------------------------ reports

    def quality(self) -> dict:
        """Volumes, edge-length ratios and counts per type."""
        out = {}
        for k, e in self.elements.items():
            v = self.element_volumes(k)
            x = self.nodes[e]
            ed = np.array(EDGES[k])
            ln = np.linalg.norm(x[:, ed[:, 1]] - x[:, ed[:, 0]], axis=2)
            ratio = ln.max(1) / np.maximum(ln.min(1), 1e-300)
            out[k] = {"elements": len(e), "volume_min": float(v.min()),
                      "volume_max": float(v.max()), "non_positive": int((v <= 0).sum()),
                      "edge_ratio_max": float(ratio.max()),
                      "edge_ratio_p99": float(np.percentile(ratio, 99))}
        return out

    def summary(self) -> dict:
        used = np.zeros(self.n_nodes, bool)
        for e in self.elements.values():
            used[e.reshape(-1)] = True
        lo, hi = self.nodes.min(0), self.nodes.max(0)
        return {"source": self.source, "unit": self.unit, "nodes": self.n_nodes,
                "unused_nodes": int((~used).sum()), "elements": self.counts(),
                "volume": self.volume(), "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
                "zones": [{"zone": z.zone, "kind": z.kind, "name": z.name,
                           "faces": z.n_faces} for z in self.zones.values()]}


def _opposite(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise: is polygon ``b`` polygon ``a`` traversed the other way (any rotation)?"""
    tri = a[:, 3] < 0
    out = np.zeros(len(a), bool)
    for n, sel in ((3, tri), (4, ~tri)):
        if not sel.any():
            continue
        aa, bb = a[sel, :n], b[sel, :n][:, ::-1]
        ok = np.zeros(sel.sum(), bool)
        for r in range(n):
            ok |= (np.roll(bb, r, axis=1) == aa).all(1)
        out[sel] = ok
    return out


def _fan_volumes(nodes: np.ndarray, elem: np.ndarray, faces, batch: int = 1_000_000):
    out = np.empty(len(elem))
    for s in range(0, len(elem), batch):
        x = nodes[elem[s:s + batch]]
        c = x.mean(1)
        v = np.zeros(len(x))
        for f in faces:
            p = x[:, list(f)] - c[:, None]
            if len(f) == 3:
                v += np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2]))
            else:
                fc = p.mean(1)
                for i in range(4):
                    v += np.einsum("ij,ij->i", fc, np.cross(p[:, i], p[:, (i + 1) % 4]))
        out[s:s + batch] = v / 6.0
    return out


def to_vtk(mesh: UnstructuredMesh):
    """``(cells, celltypes)`` in VTK's layout (for pyvista.UnstructuredGrid)."""
    cells, types = [], []
    for k in KINDS:
        if k not in mesh.elements:
            continue
        e = mesh.elements[k]
        n = e.shape[1]
        cells.append(np.concatenate([np.full((len(e), 1), n), e], 1).reshape(-1))
        types.append(np.full(len(e), VTK_TYPE[k], np.uint8))
    return np.concatenate(cells), np.concatenate(types)


__all__ = ["EDGES", "FACES", "FLUENT_TYPE", "KINDS", "NODES_PER", "VTK_TYPE", "BoundaryZone",
           "UnstructuredMesh", "to_vtk"]
