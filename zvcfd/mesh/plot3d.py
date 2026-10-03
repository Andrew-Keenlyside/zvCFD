"""Two-dimensional PLOT3D grids, extruded one cell deep for the three-dimensional solver.

NASA's Turbulence Modeling Resource publishes its verification and
validation grids (flat plate, NACA 0012, ...) as formatted 2-D PLOT3D: a
block count, ``idim jdim``, then all ``x`` and all ``y`` with ``i`` fastest.
:func:`read_plot3d_2d` reads one block; :func:`extrude_2d` merges points
that coincide in the plane (a C-grid's wake cut), extrudes the quadrilaterals
along ``z`` into one layer of hexahedra, and names the boundary faces with a
classifier, as :mod:`zvcfd.mesh.generate` does. The two ``z`` planes are
usually symmetry zones: the flow is then the 2-D one.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from zvcfd.mesh.core import UnstructuredMesh


def read_plot3d_2d(path) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)``, each ``(jdim, idim)``, from a formatted 2-D PLOT3D file (``.gz`` too)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        v = np.array(fh.read().split(), float)
    if int(v[0]) != 1:
        raise ValueError(f"{path.name}: {int(v[0])} blocks; one is supported")
    ni, nj = int(v[1]), int(v[2])
    xy = v[3:]
    if len(xy) != 2 * ni * nj:
        raise ValueError(f"{path.name}: {len(xy)} values for a {ni} x {nj} grid")
    return xy[:ni * nj].reshape(nj, ni), xy[ni * nj:].reshape(nj, ni)


def _merge(points: np.ndarray, tol: float) -> np.ndarray:
    """Representative index of each point after merging points within ``tol`` of each other."""
    from scipy.spatial import cKDTree

    parent = np.arange(len(points))
    pairs = cKDTree(points).query_pairs(tol, output_type="ndarray")

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(len(points))])


def extrude_2d(x: np.ndarray, y: np.ndarray, span: float, classify, *,
               merge_tol: float = 1e-12, unit: str | None = None) -> UnstructuredMesh:
    """One layer of hexahedra, ``z`` from 0 to ``span``, from a structured 2-D grid.

    Args:
        x, y: ``(jdim, idim)`` node coordinates.
        span: the layer's thickness.
        classify: ``f(centroids (F, 3), normals (F, 3)) -> {"which": (F,), "names":
            [(name, kind), ...]}`` naming every boundary face (as in
            :mod:`zvcfd.mesh.generate`).
        merge_tol: points closer than this in the plane are one node (relative to the
            grid's extent).
    """
    from zvcfd.mesh.generate import _orient, _zones_from_faces

    nj, ni = x.shape
    p2 = np.c_[x.ravel(), y.ravel()]
    scale = float(np.ptp(p2, axis=0).max())
    rep = _merge(p2, merge_tol * scale)
    keep, local = np.unique(rep, return_inverse=True)
    n2 = len(keep)
    pts = p2[keep]
    nodes = np.concatenate([np.c_[pts, np.zeros(n2)], np.c_[pts, np.full(n2, span)]])
    idx = local.reshape(nj, ni)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    quad = np.stack([a, b, c, d], 1)
    # quadrilaterals collapsed by the merge (none in a C-grid, but be safe) are dropped
    ok = np.array([len(set(q)) == 4 for q in quad])
    quad = quad[ok]
    hexes = np.concatenate([quad, quad + n2], 1)
    hexes = _orient(nodes, "hex", hexes)
    mesh = UnstructuredMesh(nodes, {"hex": hexes}, unit=unit)
    mesh.zones = _zones_from_faces(mesh, classify)
    return mesh


__all__ = ["extrude_2d", "read_plot3d_2d"]
