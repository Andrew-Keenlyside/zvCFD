"""Element-based finite-volume geometry: shape functions and median-dual control volumes.

Each element is split into one sub-control-volume (SCV) per node. The
surfaces between them run through edge midpoints, face centroids (the
mean of the face's nodes) and the element centroid (the mean of its
nodes). The piece of surface between the SCVs of nodes ``a`` and ``b``,
for element edge ``(a, b)``, is the **integration-point face** of that
edge: the loop ``(m_ab, f_1, c, f_2)`` through the edge midpoint, the
centroids of the two element faces sharing the edge, and the element
centroid. A tetrahedron has 6 integration points, a pyramid 8, a wedge 9
and a hexahedron 12, one per edge (Schneider & Raw 1987), each carrying
two flux points.

Two properties are exact to round-off, and :func:`dual_geometry` checks
both:

- node volumes add up to the element volumes of
  :meth:`~zvcfd.mesh.core.UnstructuredMesh.element_volumes`, because both
  fan every face about its centroid;
- every control volume is closed: its outward area vectors, integration-
  point faces plus boundary sub-faces, sum to zero (the geometric
  conservation law).

Each ip face is taken as its two planar triangles ``(m, f_1, c)`` and
``(m, c, f_2)``, and fluxes are evaluated at the centroid of each: two
**flux points** per element edge. On an affine element the flux of a
linear field through each triangle is then exact, so it is exact through
every control volume, boundary ones included. With one point per
(non-planar) ip face it is exact only where the ring of elements around an
edge is closed. Volumes come from the same triangles.

Shape functions (isoparametric, linear-complete):

- tet: ``1 − ξ − η − ζ, ξ, η, ζ`` on the unit tetrahedron;
- wedge: area coordinates ``(1 − ξ − η, ξ, η)`` × ``(1 ∓ ζ)/2``;
- hex: trilinear on ``[−1, 1]³``;
- pyramid: a hexahedron with its top face collapsed to the apex; the base
  nodes carry ``⅛ (1 ± ξ)(1 ± η)(1 − ζ)`` and the apex ``(1 + ζ)/2``.

Each flux point sits at its triangle's centroid on the reference element,
mapped back to parameters. For every type but the pyramid the reference
map is the identity. On affine elements (every tetrahedron, and
undistorted others), the discrete gradient and divergence of linear
fields are then exact on every control volume.

Reference: G. E. Schneider, M. J. Raw, *Control volume finite-element
method for heat transfer and fluid flow using colocated variables*,
Numer. Heat Transfer 11, 363 (1987).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import numpy as np

from zvcfd.mesh.core import EDGES, FACES, KINDS, UnstructuredMesh

XI_NODES = {
    "tet": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
    "wedge": np.array([[0, 0, -1], [1, 0, -1], [0, 1, -1], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
                      float),
    "hex": np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float),
    "pyramid": np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [0, 0, 1]], float),
}


def shape(kind: str, xi: np.ndarray):
    """Shape functions and their parametric derivatives at points ``xi (..., 3)``.

    Returns ``N (..., n)`` and ``dN (..., n, 3)``.
    """
    xi = np.asarray(xi, float)
    x, y, z = xi[..., 0], xi[..., 1], xi[..., 2]
    one, zero = np.ones_like(x), np.zeros_like(x)
    if kind == "tet":
        N = np.stack([1 - x - y - z, x, y, z], -1)
        d = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        dN = np.broadcast_to(d, x.shape + d.shape).copy()
    elif kind == "wedge":
        L = [1 - x - y, x, y]
        dL = [(-one, -one), (one, zero), (zero, one)]
        lo, hi = (1 - z) / 2, (1 + z) / 2
        N = np.stack([L[i] * lo for i in range(3)] + [L[i] * hi for i in range(3)], -1)
        dN = np.stack([np.stack([dL[i][0] * lo, dL[i][1] * lo, -L[i] / 2], -1) for i in range(3)]
                      + [np.stack([dL[i][0] * hi, dL[i][1] * hi, L[i] / 2], -1)
                         for i in range(3)], -2)
    elif kind == "hex":
        s = XI_NODES["hex"]
        fx = 1 + x[..., None] * s[:, 0]
        fy = 1 + y[..., None] * s[:, 1]
        fz = 1 + z[..., None] * s[:, 2]
        N = fx * fy * fz / 8
        dN = np.stack([s[:, 0] * fy * fz, fx * s[:, 1] * fz, fx * fy * s[:, 2]], -1) / 8
    elif kind == "pyramid":
        s = XI_NODES["pyramid"][:4]
        fx = 1 + x[..., None] * s[:, 0]
        fy = 1 + y[..., None] * s[:, 1]
        fz = (1 - z)[..., None]
        base = fx * fy * fz / 8
        dbase = np.stack([s[:, 0] * fy * fz, fx * s[:, 1] * fz, -fx * fy], -1) / 8
        N = np.concatenate([base, ((1 + z) / 2)[..., None]], -1)
        dap = np.stack([zero, zero, one / 2], -1)[..., None, :]
        dN = np.concatenate([dbase, dap], -2)
    else:
        raise ValueError(kind)
    return N, dN


@dataclass(frozen=True)
class Topology:
    """Local dual-mesh topology of one element type.

    Per-element points are numbered: nodes ``0..n-1``, edge midpoints
    ``n..n+ne-1``, face centroids, then the centroid. ``loops[e]`` gives the
    four points ``(m, f_1, c, f_2)`` of element edge ``e``'s ip face, ordered
    so that its area vector points from the edge's first node to its second.

    Fluxes are evaluated at **two flux points per ip face**, one on each of
    its planar halves ``(m, f_1, c)`` and ``(m, c, f_2)`` (``tris``). Arrays
    indexed by flux point: ``edges`` (the edge's two local nodes, each edge
    listed twice), ``tris``, ``xi_ip``, ``N_ip``, ``dN_ip``.
    """

    kind: str
    n: int
    edges: np.ndarray          # (n_ip, 2) local nodes of each flux point's edge
    faces: tuple
    loops: np.ndarray          # (ne, 4) point indices of each ip face
    tris: np.ndarray           # (n_ip, 3) point indices of each flux point's triangle
    xi_ip: np.ndarray          # (n_ip, 3)
    N_ip: np.ndarray           # (n_ip, n)
    dN_ip: np.ndarray          # (n_ip, n, 3)
    point_weights: np.ndarray  # (n_points, n): each point as a combination of the nodes

    @property
    def n_ip(self) -> int:
        return len(self.edges)


@cache
def topology(kind: str) -> Topology:
    n = XI_NODES[kind].shape[0]
    el_edges = np.array(EDGES[kind])
    faces = FACES[kind]
    ne, nf = len(el_edges), len(faces)
    W = np.zeros((n + ne + nf + 1, n))
    W[:n] = np.eye(n)
    for e, (a, b) in enumerate(el_edges):
        W[n + e, [a, b]] = 0.5
    for f, fn in enumerate(faces):
        W[n + ne + f, list(fn)] = 1.0 / len(fn)
    W[-1] = 1.0 / n
    ref = W @ XI_NODES[kind]              # the points on the reference element (as geometry)
    loops = np.zeros((ne, 4), int)
    for e, (a, b) in enumerate(el_edges):
        fs = [f for f, fn in enumerate(faces) if a in fn and b in fn]
        assert len(fs) == 2, (kind, e, fs)
        m, c = n + e, n + ne + nf
        f1, f2 = n + ne + fs[0], n + ne + fs[1]
        area = 0.5 * np.cross(ref[c] - ref[m], ref[f2] - ref[f1])
        if area @ (ref[b] - ref[a]) < 0:
            f1, f2 = f2, f1
        loops[e] = (m, f1, c, f2)
    tris = np.concatenate([loops[:, [0, 1, 2]], loops[:, [0, 2, 3]]], 1).reshape(-1, 3)
    edges = np.repeat(el_edges, 2, axis=0)
    # each flux point at its triangle's centroid on the reference element, mapped back
    xi_ip = _reference_inverse(kind, ref[tris].mean(1))
    N, dN = shape(kind, xi_ip)
    return Topology(kind, n, edges, faces, loops, tris, xi_ip, N, dN, W)


def _reference_inverse(kind: str, x: np.ndarray) -> np.ndarray:
    """Parameters of points ``x`` of the reference element (whose nodes are ``XI_NODES``).

    The tetrahedron, wedge and hexahedron maps are the identity there. The
    collapsed-hexahedron pyramid maps ``(ξ, η, ζ)`` to
    ``(ξ (1 − ζ)/2, η (1 − ζ)/2, ζ)``.
    """
    if kind != "pyramid":
        return x.copy()
    z = x[:, 2]
    return np.stack([2 * x[:, 0] / (1 - z), 2 * x[:, 1] / (1 - z), z], 1)


def element_points(x: np.ndarray, topo: Topology) -> np.ndarray:
    """All dual points ``(E, n_points, 3)`` of elements with node coordinates ``x (E, n, 3)``."""
    return np.einsum("pn,enk->epk", topo.point_weights, x)


def ip_areas(P: np.ndarray, topo: Topology) -> np.ndarray:
    """Flux-point area vectors ``(E, n_ip, 3)``, pointing from edge node 0 to node 1."""
    T = P[:, topo.tris]                                    # (E, n_ip, 3, 3)
    return 0.5 * np.cross(T[:, :, 1] - T[:, :, 0], T[:, :, 2] - T[:, :, 0])


def ip_points(P: np.ndarray, topo: Topology) -> np.ndarray:
    """Flux-point positions ``(E, n_ip, 3)``: the centroids of their triangles."""
    return P[:, topo.tris].mean(2)


def scv_volumes(P: np.ndarray, topo: Topology) -> np.ndarray:
    """Sub-control-volume of each node ``(E, n)``: the divergence theorem over its ip faces.

    Relative to the node itself, the element-face pieces of the SCV surface
    (fans about the face centroid) contribute nothing: each of their
    triangles contains the node. Only the ip-face triangles enter, and
    those are planar, so the volumes are exact.
    """
    T = P[:, topo.tris]                                    # (E, n_ip, 3, 3)
    A = 0.5 * np.cross(T[:, :, 1] - T[:, :, 0], T[:, :, 2] - T[:, :, 0])
    cen = T.mean(2)
    x = P[:, :topo.n]
    a, b = topo.edges[:, 0], topo.edges[:, 1]
    ca = np.einsum("eik,eik->ei", cen - x[:, a], A)
    cb = -np.einsum("eik,eik->ei", cen - x[:, b], A)
    Ma = np.zeros((topo.n_ip, topo.n))
    Mb = np.zeros((topo.n_ip, topo.n))
    Ma[np.arange(topo.n_ip), a] = 1.0
    Mb[np.arange(topo.n_ip), b] = 1.0
    return (ca @ Ma + cb @ Mb) / 3.0


def gradients(x: np.ndarray, topo: Topology):
    """Shape-function gradients at the integration points ``(E, ne, n, 3)`` and ``det J``."""
    J = np.einsum("enk,ind->eikd", x, topo.dN_ip)          # dx_k / dxi_d
    Jinv = np.linalg.inv(J)
    G = np.einsum("ind,eidk->eink", topo.dN_ip, Jinv)
    return G, np.linalg.det(J)


def boundary_subfaces(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area vector of each node's share of each boundary face ``(F, 4, 3)`` (0 on pads).

    A node's share runs node -> next edge midpoint -> face centroid ->
    previous edge midpoint, with the face's (outward) orientation.
    """
    tri = faces[:, 3] < 0
    out = np.zeros(faces.shape + (3,))
    for n, sel in ((3, tri), (4, ~tri)):
        if not sel.any():
            continue
        p = nodes[faces[sel, :n]]
        fc = p.mean(1)
        for i in range(n):
            nxt = 0.5 * (p[:, i] + p[:, (i + 1) % n])
            prv = 0.5 * (p[:, i] + p[:, (i - 1) % n])
            out[np.flatnonzero(sel), i] = 0.5 * np.cross(fc - p[:, i], prv - nxt)
    return out


def boundary_subface_weights(faces: np.ndarray) -> np.ndarray:
    """Interpolation weights ``(F, 4, 4)`` from a face's nodes to each sub-face's area centroid.

    ``w[f, i, j]`` weights node ``j`` of face ``f`` for the sub-face of node
    ``i``. A triangle's median dual cuts it into six triangles of equal area,
    so a sub-face's centroid is exactly ``11/18`` of its node and ``7/36`` of
    each other node: flows of linear fields are exact. A quadrilateral's
    sub-face centroid is the bilinear point ``(½, ½)`` from its node:
    ``9/16``, ``3/16``, ``3/16``, ``1/16``, exact on parallelograms.
    """
    tri = faces[:, 3] < 0
    w = np.zeros(faces.shape + (4,))
    t = np.full((3, 3), 7 / 36)
    np.fill_diagonal(t, 11 / 18)
    q = np.array([[9, 3, 1, 3], [3, 9, 3, 1], [1, 3, 9, 3], [3, 1, 3, 9]]) / 16
    w[tri, :3, :3] = t
    w[~tri] = q
    return w


@dataclass
class DualGeometry:
    """Node control volumes and integration-point areas of a whole mesh.

    Attributes:
        node_volume: ``(N,)`` control-volume volumes.
        areas: ``{kind: (E, ne, 3)}`` ip area vectors (if kept).
        closure: ``(N, 3)`` sum of each control volume's outward area vectors.
        boundary: ``{zone: (F, 4, 3)}`` boundary sub-face area vectors.
        report: the two checks and timings.
    """

    node_volume: np.ndarray
    areas: dict[str, np.ndarray] = field(default_factory=dict)
    closure: np.ndarray | None = None
    boundary: dict[int, np.ndarray] = field(default_factory=dict)
    report: dict = field(default_factory=dict)


def dual_geometry(mesh: UnstructuredMesh, *, keep_areas: bool = True,
                  batch: int = 500_000) -> DualGeometry:
    """Build the median dual of ``mesh`` and check volume and closure."""
    import time

    t0 = time.time()
    N = mesh.n_nodes
    vol = np.zeros(N)
    close = np.zeros((N, 3))
    areas: dict[str, list] = {}
    elem_vol = 0.0
    ortho_min = np.inf
    for kind in KINDS:
        if kind not in mesh.elements:
            continue
        topo = topology(kind)
        E = mesh.elements[kind]
        parts = []
        for s in range(0, len(E), batch):
            e = E[s:s + batch]
            x = mesh.nodes[e]
            P = element_points(x, topo)
            A = ip_areas(P, topo)
            v = scv_volumes(P, topo)
            vol += np.bincount(e.ravel(), v.ravel(), N)
            a, b = e[:, topo.edges[:, 0]], e[:, topo.edges[:, 1]]
            for k in range(3):
                close[:, k] += np.bincount(a.ravel(), A[..., k].ravel(), N)
                close[:, k] -= np.bincount(b.ravel(), A[..., k].ravel(), N)
            d = x[:, topo.edges[:, 1]] - x[:, topo.edges[:, 0]]
            cosang = np.einsum("eik,eik->ei", A, d) / (
                np.linalg.norm(A, axis=2) * np.linalg.norm(d, axis=2))
            ortho_min = min(ortho_min, float(np.degrees(np.arcsin(np.clip(cosang, -1, 1))).min()))
            if keep_areas:
                parts.append(A)
        elem_vol += float(mesh.element_volumes(kind).sum())
        if keep_areas:
            areas[kind] = np.concatenate(parts)
    bnd = {}
    for zid, z in mesh.zones.items():
        S = boundary_subfaces(mesh.nodes, z.faces)
        bnd[zid] = S
        f = np.where(z.faces < 0, 0, z.faces)
        for k in range(3):
            close[:, k] += np.bincount(f.ravel(), S[..., k].ravel(), N)
    # scale for the closure check: the typical control-volume face area
    h2 = np.cbrt(np.maximum(vol, 1e-300)) ** 2
    rel = np.linalg.norm(close, axis=1) / h2
    report = {"volume_nodes": float(vol.sum()), "volume_elements": elem_vol,
              "volume_rel_diff": abs(vol.sum() - elem_vol) / abs(elem_vol),
              "closure_max_rel": float(rel.max()), "min_node_volume": float(vol.min()),
              "orthogonality_min_deg": ortho_min, "seconds": time.time() - t0}
    return DualGeometry(vol, {k: v for k, v in areas.items()}, close, bnd, report)


__all__ = ["DualGeometry", "Topology", "XI_NODES", "boundary_subface_weights",
           "boundary_subfaces", "dual_geometry", "ip_points",
           "element_points", "gradients", "ip_areas", "scv_volumes", "shape", "topology"]
