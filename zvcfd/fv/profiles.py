"""Inlet velocity profiles on a boundary zone of an unstructured mesh.

A flow-rate inlet takes a *shape* and scales it so that the discrete inflow
(the sub-face fluxes the solvers use) equals the prescribed flow exactly:

``"plug"``
    uniform normal velocity.
``"poiseuille"`` (alias ``"parabolic"``)
    the fully developed laminar profile of the zone's own cross-section:
    ``−Δu = 1`` on the (planar) face with ``u = 0`` on its rim, solved with
    linear triangles (quadrilaterals split in two). On a circle this is the
    parabola ``1 − (r/R)²``; on an ellipse, a rectangle or a segmented vessel
    it is the exact developed shape rather than a parabola fitted by radius.

The direction is the zone's inward unit normal (area-weighted mean).

Reference: F. M. White, *Viscous Fluid Flow*, 3rd ed., §3-3 (fully
developed duct flow, Poisson problem for the axial velocity).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def _triangles(faces: np.ndarray) -> np.ndarray:
    tri = [faces[:, [0, 1, 2]]]
    quad = faces[:, 3] >= 0
    if quad.any():
        tri.append(faces[quad][:, [0, 2, 3]])
    return np.concatenate(tri)


def zone_normal(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit normal of a face set (the orientation of its faces)."""
    t = _triangles(faces)
    a = np.cross(nodes[t[:, 1]] - nodes[t[:, 0]], nodes[t[:, 2]] - nodes[t[:, 0]]).sum(0)
    return a / np.linalg.norm(a)


def rim_nodes(faces: np.ndarray) -> np.ndarray:
    """Nodes on edges used by exactly one face of the set."""
    edges = []
    for f in faces:
        v = f[f >= 0]
        edges += [tuple(sorted((v[i], v[(i + 1) % len(v)]))) for i in range(len(v))]
    e, n = np.unique(np.array(edges), axis=0, return_counts=True)
    return np.unique(e[n == 1])


def poiseuille(nodes: np.ndarray, faces: np.ndarray) -> dict[int, float]:
    """Fully developed profile on a face set, ``{node: u}`` with ``max u = 1``.

    Args:
        nodes: ``(N, 3)`` mesh coordinates.
        faces: ``(F, 4)`` face node indices (``-1`` for the fourth of a triangle).
    """
    zn = np.unique(faces[faces >= 0])
    loc = {int(g): i for i, g in enumerate(zn)}
    n = zone_normal(nodes, faces)
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    xy = np.stack([nodes[zn] @ e1, nodes[zn] @ e2], 1)
    tri = np.vectorize(loc.get)(_triangles(faces))
    p = xy[tri]                                               # (T, 3, 2)
    d = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], 1)   # (T, 2, 2) rows = edges
    area = 0.5 * np.abs(np.linalg.det(d))
    # gradients of the barycentric functions: G (T, 3, 2)
    inv = np.linalg.inv(d)                                    # columns: grad of λ1, λ2
    g12 = np.swapaxes(inv, 1, 2)
    G = np.concatenate([-(g12[:, :1] + g12[:, 1:]), g12], 1)
    K = np.einsum("tik,tjk,t->tij", G, G, area)
    rows = np.repeat(tri, 3, 1).reshape(-1)
    cols = np.tile(tri, (1, 3)).reshape(-1)
    M = sp.csr_matrix((K.reshape(-1), (rows, cols)), shape=(len(zn),) * 2)
    f = np.bincount(tri.reshape(-1), np.repeat(area / 3, 3), len(zn))
    rim = np.array([loc[int(g)] for g in rim_nodes(faces)])
    free = np.setdiff1d(np.arange(len(zn)), rim)
    u = np.zeros(len(zn))
    u[free] = spl.spsolve(M[free][:, free].tocsc(), f[free])
    u /= u.max()
    return {int(g): float(v) for g, v in zip(zn, u)}


def shape(nodes: np.ndarray, faces: np.ndarray, profile) -> tuple[np.ndarray, np.ndarray]:
    """``(zone nodes, velocity shape (n, 3))`` along the inward normal, before scaling."""
    zn = np.unique(faces[faces >= 0])
    inward = -zone_normal(nodes, faces)
    if callable(profile):
        s = np.asarray(profile(nodes[zn]), float).reshape(-1)
    elif profile == "plug":
        s = np.ones(len(zn))
    elif profile in ("poiseuille", "parabolic"):
        u = poiseuille(nodes, faces)
        s = np.array([u[int(g)] for g in zn])
    else:
        raise ValueError(f"profile {profile!r}: plug, poiseuille (parabolic) or a callable")
    return zn, s[:, None] * inward


__all__ = ["poiseuille", "rim_nodes", "shape", "zone_normal"]
