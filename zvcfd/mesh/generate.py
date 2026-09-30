"""Validation meshes: boxes and circular tubes, in hexahedra, wedges, tetrahedra or a mix.

Every mesh is conforming: neighbouring elements share whole faces. Where
a quadrilateral has to be split into triangles, its diagonal is chosen
from its **global** node numbers, through the vertex with the smallest
index. Each element then agrees with its neighbours without any
bookkeeping. A wedge splits into three tetrahedra by the same rule, which
always has a valid decomposition (Dompierre et al. 1999).

- :func:`box`: a structured box, the classic manufactured-solution and
  channel domain. Zones ``xmin`` … ``zmax``. ``warp`` bends it smoothly, so
  distortion shrinks with refinement (for convergence studies), while
  ``perturb`` moves nodes at random by a fixed fraction of the edge.
- :func:`delaunay_box`, :func:`delaunay_tube`: *unstructured* tetrahedra,
  the Delaunay triangulation (scipy/Qhull) of a jittered lattice, with no
  preferred diagonal direction.
- :func:`tube`: a circular pipe along z on a butterfly O-grid (a square
  core and four blocks out to the wall). Zones ``inlet`` (z = 0),
  ``outlet`` (z = L) and ``wall``. With ``kind="mixed"``, the outer
  ``layers`` rings are wedges with triangles parallel to the wall
  (inflation layers, as in the Simpleware coronary mesh), and the core is
  tetrahedra.

Reference: J. Dompierre, P. Labbé, M.-G. Vallet, R. Camarero, *How to
subdivide pyramids, prisms and hexahedra into tetrahedra*, Proc. 8th
International Meshing Roundtable (1999).
"""

from __future__ import annotations

import numpy as np

from zvcfd.mesh.core import BoundaryZone, UnstructuredMesh

# Dompierre et al. (1999), table 2: vertex permutations taking a wedge's
# smallest-index vertex to position 0 while keeping the wedge's topology
_WEDGE_PERM = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3], [2, 0, 1, 5, 3, 4],
                        [3, 5, 4, 0, 2, 1], [4, 3, 5, 1, 0, 2], [5, 4, 3, 2, 1, 0]])


def wedges_to_tets(w: np.ndarray) -> np.ndarray:
    """Split wedges ``(E, 6)`` into tetrahedra ``(3E, 4)``, conformingly (Dompierre)."""
    w = np.asarray(w, np.int64)
    p = w[np.arange(len(w))[:, None], _WEDGE_PERM[np.argmin(w, axis=1)]]
    v0, v1, v2, v3, v4, v5 = p.T
    a = np.minimum(v1, v5) < np.minimum(v2, v4)
    t = np.empty((len(w), 3, 4), np.int64)
    t[:, 0] = np.where(a[:, None], np.stack([v0, v1, v2, v5], 1), np.stack([v0, v1, v2, v4], 1))
    t[:, 1] = np.where(a[:, None], np.stack([v0, v1, v5, v4], 1), np.stack([v0, v4, v2, v5], 1))
    t[:, 2] = np.stack([v0, v4, v5, v3], 1)
    return t.reshape(-1, 4)


def hex_to_wedges(h: np.ndarray) -> np.ndarray:
    """Split hexahedra ``(E, 8)`` into two wedges each, triangles on the bottom and top faces.

    The bottom quad is cut along the diagonal through its smallest-index
    node. The top quad is cut along the matching diagonal. Hexahedra stacked
    in a column therefore agree only if their shared quads choose the same
    diagonal, which holds when node numbers increase in the same pattern
    layer by layer, as in the extrusions here.
    """
    h = np.asarray(h, np.int64)
    b = h[:, :4]
    r = np.argmin(np.minimum(b[:, [0, 1, 2, 3]], b[:, [2, 3, 0, 1]]), axis=1) % 2
    # r = 0: diagonal 0-2; r = 1: diagonal 1-3
    i = np.where(r[:, None] == 0, np.array([[0, 1, 2, 0, 2, 3]]), np.array([[0, 1, 3, 1, 2, 3]]))
    rows = np.arange(len(h))[:, None]
    bot = b[rows, i]
    top = h[:, 4:][rows, i]
    w1 = np.concatenate([bot[:, :3], top[:, :3]], 1)
    w2 = np.concatenate([bot[:, 3:], top[:, 3:]], 1)
    return np.stack([w1, w2], 1).reshape(-1, 6)


def hex_to_pyramids(nodes: np.ndarray, h: np.ndarray):
    """Split each hexahedron into six pyramids about a new centre node.

    Returns ``(nodes with the centres appended, pyramids (6E, 5))``; each
    pyramid's base is a hex face, turned to face the centre.
    """
    h = np.asarray(h, np.int64)
    centre = np.arange(len(nodes), len(nodes) + len(h))
    nodes = np.concatenate([nodes, nodes[h].mean(1)])
    from zvcfd.mesh.core import FACES

    pyr = [np.concatenate([h[:, list(f)][:, ::-1], centre[:, None]], 1) for f in FACES["hex"]]
    return nodes, np.stack(pyr, 1).reshape(-1, 5)


def _orient(nodes, kind, e):
    """Make every element's volume positive by mirroring its local order."""
    x = nodes[e]
    if kind == "tet":
        s = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])
        e[s < 0] = e[s < 0][:, [0, 2, 1, 3]]
    elif kind == "wedge":
        s = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])
        e[s < 0] = e[s < 0][:, [0, 2, 1, 3, 5, 4]]
    elif kind == "hex":
        s = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 3] - x[:, 0]), x[:, 4] - x[:, 0])
        e[s < 0] = e[s < 0][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    return e


def _zones_from_faces(mesh: UnstructuredMesh, classify) -> dict[int, BoundaryZone]:
    """Boundary zones from the mesh's boundary faces, by ``classify(centroids, normals)``."""
    fx = mesh.faces()
    b = fx["c1"] < 0
    f, own = fx["faces"][b], fx["c0"][b]
    p = mesh.nodes[np.where(f < 0, f[:, :1], f)]
    tri = f[:, 3] < 0
    c = np.where(tri[:, None], p[:, :3].mean(1), p.mean(1))
    n = np.where(tri[:, None], np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                 np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    labels = classify(c, n)
    zones = {}
    for zid, (name, kind) in enumerate(labels["names"], start=3):
        sel = labels["which"] == zid - 3
        zones[zid] = BoundaryZone(zid, kind, name, f[sel], own[sel])
    if sum(z.n_faces for z in zones.values()) != len(f):
        raise ValueError("some boundary faces were not classified")
    return zones


def _finish(nodes, kind, hexes, zones_rule, perturb=0.0, seed=0, lengths=None):
    rng = np.random.default_rng(seed)
    nodes = nodes.copy()
    if kind == "hex":
        elements = {"hex": hexes}
    elif kind == "wedge":
        elements = {"wedge": hex_to_wedges(hexes)}
    elif kind == "tet":
        elements = {"tet": wedges_to_tets(hex_to_wedges(hexes))}
    elif kind == "pyramid":
        nodes, pyr = hex_to_pyramids(nodes, hexes)
        elements = {"pyramid": pyr}
    else:
        raise ValueError(f"kind {kind!r}")
    mesh = UnstructuredMesh(nodes, {k: _orient(nodes, k, v) for k, v in elements.items()})
    mesh.zones = _zones_from_faces(mesh, zones_rule)
    if perturb:
        _perturb(mesh, perturb, rng)
    return mesh


def _perturb(mesh: UnstructuredMesh, amount: float, rng) -> None:
    """Move interior nodes by up to ``amount`` × the local shortest edge, at random."""
    on_bnd = np.zeros(mesh.n_nodes, bool)
    for z in mesh.zones.values():
        f = z.faces
        on_bnd[f[f >= 0]] = True
    e = mesh.edges()
    ln = np.linalg.norm(mesh.nodes[e[:, 1]] - mesh.nodes[e[:, 0]], axis=1)
    h = np.full(mesh.n_nodes, np.inf)
    np.minimum.at(h, e[:, 0], ln)
    np.minimum.at(h, e[:, 1], ln)
    d = rng.uniform(-1, 1, (mesh.n_nodes, 3)) * (amount * h)[:, None]
    d[on_bnd] = 0.0
    mesh.nodes += d
    for k in mesh.elements:
        if (mesh.element_volumes(k) <= 0).any():
            raise ValueError("perturbation inverted an element; use a smaller amount")


# ---------------------------------------------------------------- box

def box(n=(8, 8, 8), lengths=(1.0, 1.0, 1.0), *, kind: str = "hex", origin=(0.0, 0.0, 0.0),
        perturb: float = 0.0, warp: float = 0.0, seed: int = 0) -> UnstructuredMesh:
    """A structured box of ``n = (nx, ny, nz)`` cells.

    Args:
        kind: ``"hex"``, ``"wedge"`` (each hex cut into two, triangles normal to z),
            ``"tet"`` (each wedge cut into three) or ``"pyramid"`` (each hex into six
            about its centre).
        perturb: move interior nodes randomly by up to this fraction of the local edge.
        warp: displace nodes by ``warp * L * sin 2πξ sin 2πη sin 2πζ`` along each axis
            (ξ, η, ζ in [0, 1] across the box): smooth, zero on the faces, the
            same mapping at every resolution.
    Zones 3–8: ``xmin, xmax, ymin, ymax, zmin, zmax`` (kind ``wall``).
    """
    nx, ny, nz = n
    xs = [np.linspace(o, o + L, k + 1) for o, L, k in zip(origin, lengths, n)]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    if warp:
        xi = (nodes - np.asarray(origin, float)) / np.asarray(lengths, float)
        bump = np.prod(np.sin(2 * np.pi * xi), axis=1)
        nodes = nodes + warp * bump[:, None] * np.asarray(lengths, float)[None, :]
    idx = np.arange(len(nodes)).reshape(nx + 1, ny + 1, nz + 1)
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    hexes = np.stack([idx[i, j, k], idx[i + 1, j, k], idx[i + 1, j + 1, k], idx[i, j + 1, k],
                      idx[i, j, k + 1], idx[i + 1, j, k + 1], idx[i + 1, j + 1, k + 1],
                      idx[i, j + 1, k + 1]], 1)
    lo = np.asarray(origin, float)
    hi = lo + np.asarray(lengths, float)
    tol = 1e-9 * max(lengths)

    def rule(c, nrm):
        which = np.full(len(c), -1)
        for s, (ax, side) in enumerate([(0, lo), (0, hi), (1, lo), (1, hi), (2, lo), (2, hi)]):
            which[np.abs(c[:, ax] - side[ax]) < tol] = s
        names = [(nm, "wall") for nm in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")]
        return {"which": which, "names": names}

    return _finish(nodes, kind, hexes, rule, perturb, seed)


# ---------------------------------------------------------------- tube

def _section(radius: float, n_core: int, n_ring: int, core: float, growth: float):
    """The butterfly O-grid cross-section: ``(xy (P, 2), quads (Q, 4), ring (Q,), column (Q,))``.

    ``ring`` is the quad's radial layer counted from the wall (0 = at the
    wall), or -1 inside the square core; ``column`` numbers the radial
    columns of the four outer blocks (-1 in the core).
    """
    s = core * radius / np.sqrt(2.0)                  # half-width of the core square
    t = np.linspace(-1.0, 1.0, n_core + 1)
    # radial node positions between the square side (0) and the wall (1), cells shrinking by growth
    w = growth ** np.arange(n_ring)[::-1]
    rho = np.r_[0.0, np.cumsum(w) / w.sum()]
    pts, quads, ring, column = [], [], [], []
    cx, cy = np.meshgrid(t, t, indexing="ij")
    core_xy = np.stack([s * cx.ravel(), s * cy.ravel()], 1)
    pts.append(core_xy)
    ci = np.arange((n_core + 1) ** 2).reshape(n_core + 1, n_core + 1)
    a, b = np.meshgrid(np.arange(n_core), np.arange(n_core), indexing="ij")
    a, b = a.ravel(), b.ravel()
    quads.append(np.stack([ci[a, b], ci[a + 1, b], ci[a + 1, b + 1], ci[a, b + 1]], 1))
    ring.append(np.full(len(a), -1))
    column.append(np.full(len(a), -1))
    base = len(core_xy)
    for q in range(4):
        rot = np.array([[np.cos(q * np.pi / 2), -np.sin(q * np.pi / 2)],
                        [np.sin(q * np.pi / 2), np.cos(q * np.pi / 2)]])
        side = np.stack([np.full_like(t, s), s * t], 1)                    # x = s side
        th = t * np.pi / 4
        arc = radius * np.stack([np.cos(th), np.sin(th)], 1)
        blk = (1 - rho[:, None, None]) * side[None] + rho[:, None, None] * arc[None]
        pts.append((blk.reshape(-1, 2)) @ rot.T)
        bi = base + np.arange((n_ring + 1) * (n_core + 1)).reshape(n_ring + 1, n_core + 1)
        a, b = np.meshgrid(np.arange(n_ring), np.arange(n_core), indexing="ij")
        a, b = a.ravel(), b.ravel()
        quads.append(np.stack([bi[a, b], bi[a + 1, b], bi[a + 1, b + 1], bi[a, b + 1]], 1))
        ring.append(n_ring - 1 - a)
        column.append(q * n_core + b)
        base += bi.size
    xy = np.concatenate(pts)
    quads = np.concatenate(quads)
    ring = np.concatenate(ring)
    column = np.concatenate(column)
    # merge coincident nodes (block edges), keeping first occurrences in order
    key = np.round(xy / (radius * 1e-9)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    keep = np.sort(first)
    renum = np.empty(len(first), np.int64)
    renum[np.argsort(first)] = np.arange(len(first))
    quads = renum[inv[quads]]
    return xy[keep], quads, ring, column


def tube(radius: float = 0.5, length: float = 4.0, *, n_core: int = 4, n_ring: int = 4,
         n_axial: int = 16, kind: str = "hex", layers: int = 2, growth: float = 1.0,
         core: float = 0.5, perturb: float = 0.0, seed: int = 0) -> UnstructuredMesh:
    """A circular pipe along z, ``radius`` and ``length``, on a butterfly O-grid.

    Args:
        n_core: cells along each side of the square core, and per quarter circle.
        n_ring: radial cells between the core square and the wall.
        n_axial: cells along the pipe.
        kind: ``"hex"``, ``"wedge"`` (triangles normal to the axis), ``"tet"``, or
            ``"mixed"``: the outer ``layers`` rings are wedges with triangles
            parallel to the wall, the rest tetrahedra.
        growth: ratio of successive radial cell sizes towards the wall (< 1 refines it).
        core: core square half-diagonal as a fraction of the radius.
        perturb: random interior-node displacement, as a fraction of the local edge.
    Zones: 3 ``inlet`` (velocity-inlet, z = 0), 4 ``outlet`` (pressure-outlet,
    z = L), 5 ``wall``.
    """
    xy, quads, ring, column = _section(radius, n_core, n_ring, core, growth)
    npt = len(xy)
    z = np.linspace(0.0, length, n_axial + 1)
    nodes = np.concatenate([np.column_stack([xy, np.full(npt, zk)]) for zk in z])
    k = np.arange(n_axial)
    q0 = quads[None, :, :] + (k[:, None, None] * npt)
    hexes = np.concatenate([q0, q0 + npt], axis=2).reshape(-1, 8)
    hring = np.tile(ring, n_axial)
    hcol = (np.arange(n_axial)[:, None] * (4 * n_core) + column[None, :]).reshape(-1)
    tol = 1e-9 * max(radius, length)

    def rule(c, nrm):
        which = np.full(len(c), 2)
        which[np.abs(c[:, 2]) < tol] = 0
        which[np.abs(c[:, 2] - length) < tol] = 1
        return {"which": which, "names": [("inlet", "velocity-inlet"),
                                           ("outlet", "pressure-outlet"), ("wall", "wall")]}

    if kind != "mixed":
        return _finish(nodes, kind, hexes, rule, perturb, seed)
    # mixed: wall-ring hexes -> wedges with triangles on the constant-radius faces; core -> tets
    wall = (hring >= 0) & (hring < layers)
    core_w = hex_to_wedges(hexes[~wall])
    tets = wedges_to_tets(core_w)
    h = hexes[wall]
    # radial direction in a ring quad is local 0->1 (a increases outwards): the inner face is
    # (0, 3, 7, 4) and the outer (1, 2, 6, 5); cut both along the diagonal the core chooses for
    # the inner quad, so the innermost ring meets the core's triangles
    inner = h[:, [0, 3, 7, 4]]
    outer = h[:, [1, 2, 6, 5]]
    col = np.argmin(np.minimum(inner[:, [0, 1, 2, 3]], inner[:, [2, 3, 0, 1]]), axis=1) % 2
    # every ring layer of a column must cut the same way as the innermost: take the choice of
    # the column's innermost quad (same angular/axial position, smallest ring index)
    key = hring[wall]
    pos = hcol[wall]
    order = np.lexsort((-key, pos))
    first = np.r_[True, pos[order][1:] != pos[order][:-1]]
    grp = np.cumsum(first) - 1
    choice = col[order][first][grp]
    col[order] = choice
    i = np.where(col[:, None] == 0, np.array([[0, 1, 2, 0, 2, 3]]), np.array([[0, 1, 3, 1, 2, 3]]))
    rows = np.arange(len(h))[:, None]
    bi, bo = inner[rows, i], outer[rows, i]
    w = np.concatenate([np.concatenate([bi[:, :3], bo[:, :3]], 1),
                        np.concatenate([bi[:, 3:], bo[:, 3:]], 1)])
    elements = {"tet": _orient(nodes, "tet", tets), "wedge": _orient(nodes, "wedge", w)}
    mesh = UnstructuredMesh(nodes, elements)
    mesh.zones = _zones_from_faces(mesh, rule)
    if perturb:
        _perturb(mesh, perturb, np.random.default_rng(seed))
    return mesh


# ---------------------------------------------------------------- FDA benchmark nozzle

def fda_nozzle(*, d_inlet: float = 0.012, d_throat: float = 0.004, cone_length: float = 0.022685,
               throat_length: float = 0.04, inlet_length: float = 0.05,
               outlet_length: float = 0.12, n_core: int = 4, n_ring: int = 4,
               n_annulus: int = 8, h_axial: float | None = None, growth: float = 1.03,
               kind: str = "hex") -> UnstructuredMesh:
    """The FDA benchmark nozzle, sudden-expansion orientation: inlet pipe, 20° cone, throat, step.

    Default dimensions are the FDA's (Hariharan et al. 2011; Stewart et al.
    2012): 12 mm pipe, a conical contraction 22.685 mm long to a 4 mm throat
    40 mm long, and a sudden expansion back to 12 mm. The inlet and outlet
    lengths are free parameters. The flow runs along +z, with z = 0 at the
    sudden expansion.

    The cross-section is a butterfly O-grid (``n_core``, ``n_ring``) scaled
    to the local radius through the inlet pipe, cone and throat. Past the
    expansion a polar annulus of ``n_annulus`` radial cells (graded towards
    the jet's shear layer) joins it to the pipe wall, and the annulus's
    upstream face is the step wall. Axial cells are ``h_axial`` long (default:
    the throat's radial spacing) in the throat, and grow by ``growth``
    downstream. Zones: 3 ``inlet``, 4 ``outlet``, 5 ``wall`` (including the step).
    """
    R0, Rt = d_inlet / 2, d_throat / 2
    xy, quads, _, _ = _section(1.0, n_core, n_ring, 0.5, 1.0)       # unit disk
    npt_in = len(xy)
    rim = np.flatnonzero(np.isclose(np.hypot(xy[:, 0], xy[:, 1]), 1.0))
    ang = np.arctan2(xy[rim, 1], xy[rim, 0])
    rim = rim[np.argsort(ang)]
    ang = np.sort(ang)
    # annulus rings from Rt to R0, fine near Rt (the jet's shear layer)
    rr = Rt + (R0 - Rt) * _graded(n_annulus, 1.15)
    ann_xy = np.concatenate([np.stack([r * np.cos(ang), r * np.sin(ang)], 1) for r in rr[1:]])
    nrim = len(rim)
    h = h_axial or (Rt / (n_core + n_ring))
    # axial stations
    z_step = 0.0
    z_throat0 = -throat_length
    z_cone0 = z_throat0 - cone_length
    z_in = z_cone0 - inlet_length
    zs_in = np.linspace(z_in, z_cone0, max(2, int(round(inlet_length / (3 * h)))) + 1)
    zs_cone = np.linspace(z_cone0, z_throat0, max(2, int(round(cone_length / (1.5 * h)))) + 1)
    zs_thr = np.linspace(z_throat0, z_step, max(2, int(round(throat_length / h))) + 1)
    zs_out = [z_step]
    dz = h
    while zs_out[-1] < outlet_length - 1e-12:
        zs_out.append(min(zs_out[-1] + dz, outlet_length))
        dz *= growth
    zs_out = np.array(zs_out)

    def radius(z):
        return np.where(z <= z_cone0, R0, np.where(z >= z_throat0, Rt,
                                                   R0 + (Rt - R0) * (z - z_cone0) / cone_length))

    up = np.unique(np.concatenate([zs_in, zs_cone, zs_thr]))
    nodes, hexes = [], []
    # upstream of (and at) the expansion: the scaled disk only
    for k, z in enumerate(up):
        nodes.append(np.column_stack([xy * radius(z), np.full(npt_in, z)]))
    n_up = len(up)
    for k in range(n_up - 1):
        a, b2 = k * npt_in, (k + 1) * npt_in
        hexes.append(np.concatenate([quads + a, quads + b2], 1))
    # downstream: disk (radius Rt) plus annulus, per station; the step station shares the disk
    step_disk = (n_up - 1) * npt_in                     # disk nodes at z = 0
    offset = n_up * npt_in
    ring_q = []
    for j in range(n_annulus):
        inner = rim if j == 0 else npt_in + (j - 1) * nrim + np.arange(nrim)
        outer = npt_in + j * nrim + np.arange(nrim)
        nxt = np.roll(np.arange(nrim), -1)
        ring_q.append(np.stack([inner, outer, outer[nxt], inner[nxt]], 1))
    ring_q = np.concatenate(ring_q)
    sec_q = np.concatenate([quads, ring_q])
    per = npt_in + n_annulus * nrim
    sec_xy = np.concatenate([xy * Rt, ann_xy])
    station = []
    for k, z in enumerate(zs_out):
        if k == 0:
            ids = np.concatenate([step_disk + np.arange(npt_in),
                                  offset + np.arange(n_annulus * nrim)])
            nodes.append(np.column_stack([ann_xy, np.full(len(ann_xy), z)]))
            offset += n_annulus * nrim
        else:
            ids = offset + np.arange(per)
            nodes.append(np.column_stack([sec_xy, np.full(per, z)]))
            offset += per
        station.append(ids)
    for k in range(len(zs_out) - 1):
        a, b2 = station[k], station[k + 1]
        hexes.append(np.concatenate([a[sec_q], b2[sec_q]], 1))
    nodes = np.concatenate(nodes)
    hexes = np.concatenate(hexes)
    tol = 1e-9 * d_inlet

    def rule(c, nrm):
        which = np.full(len(c), 2)
        which[np.abs(c[:, 2] - z_in) < tol] = 0
        which[np.abs(c[:, 2] - outlet_length) < tol] = 1
        return {"which": which, "names": [("inlet", "velocity-inlet"),
                                           ("outlet", "pressure-outlet"), ("wall", "wall")]}

    mesh = _finish(nodes, kind, hexes, rule)
    mesh.meta.update({"z_inlet": z_in, "z_cone": z_cone0, "z_throat": z_throat0,
                      "z_step": z_step, "z_outlet": outlet_length})
    return mesh


# ---------------------------------------------------------------- cylinder in a channel

def _graded(n: int, ratio: float) -> np.ndarray:
    """``n + 1`` points in [0, 1] whose cell sizes grow geometrically by ``ratio``."""
    w = ratio ** np.arange(n)
    return np.r_[0.0, np.cumsum(w) / w.sum()]


def cylinder_channel(m: int = 8, *, length: float = 2.2, height: float = 0.41,
                     centre=(0.2, 0.2), radius: float = 0.05, half: float = 0.1,
                     ring: int | None = None, ring_ratio: float = 1.15,
                     downstream_ratio: float = 1.04, depth: float | None = None,
                     kind: str = "hex") -> UnstructuredMesh:
    """The DFG benchmark channel with a cylinder, one cell deep (quasi-2-D).

    A butterfly O-grid joins the cylinder to the square ``centre ± half``
    (``m`` cells per quarter circle, ``ring`` radial cells graded towards the
    cylinder); eight rectangular blocks fill the rest of ``[0, length] ×
    [0, height]``, graded downstream. ``m`` is the resolution knob: cells
    double in each direction when it doubles.
    Zones: 3 ``inlet`` (x = 0), 4 ``outlet``, 5 ``walls`` (y = 0, height),
    6 ``cylinder``, 7 ``front`` / 8 ``back`` (z faces, for symmetry).
    """
    cx, cy = centre
    ring = ring or m
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    h = 2 * half / m                                    # cell size along the square
    nl = max(1, int(round(x0 / h)))
    nb = max(1, int(round(y0 / h)))
    nt = max(1, int(round((height - y1) / h)))
    # downstream: first cell h, growing by downstream_ratio
    L = length - x1
    nr = 1
    while h * (downstream_ratio ** nr - 1) / (downstream_ratio - 1) < L:
        nr += 1
    xs_r = x1 + L * _graded(nr, downstream_ratio)
    xs = [np.linspace(0, x0, nl + 1), np.linspace(x0, x1, m + 1), xs_r]
    ys = [np.linspace(0, y0, nb + 1), np.linspace(y0, y1, m + 1),
          np.linspace(y1, height, nt + 1)]
    pts, quads = [], []
    base = 0

    def add_block(P):                                    # P: (ni + 1, nj + 1, 2), i then j
        nonlocal base
        ni, nj = P.shape[0] - 1, P.shape[1] - 1
        idx = base + np.arange((ni + 1) * (nj + 1)).reshape(ni + 1, nj + 1)
        a, b2 = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
        a, b2 = a.ravel(), b2.ravel()
        q = np.stack([idx[a, b2], idx[a + 1, b2], idx[a + 1, b2 + 1], idx[a, b2 + 1]], 1)
        # counter-clockwise
        v = P.reshape(-1, 2)[q - base]
        cross = (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 3, 1] - v[:, 0, 1]) - \
            (v[:, 1, 1] - v[:, 0, 1]) * (v[:, 3, 0] - v[:, 0, 0])
        q[cross < 0] = q[cross < 0][:, ::-1]
        pts.append(P.reshape(-1, 2))
        quads.append(q)
        base += P.shape[0] * P.shape[1]

    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            X, Y = np.meshgrid(xs[i], ys[j], indexing="ij")
            add_block(np.stack([X, Y], -1))
    # O-grid: four blocks from the square's sides to quarter arcs
    t = np.linspace(-1.0, 1.0, m + 1)
    rho = _graded(ring, ring_ratio)                     # 0 at the cylinder
    for q in range(4):
        ang = q * np.pi / 2
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        side = np.stack([np.full_like(t, half), half * t], 1) @ rot.T
        th = t * np.pi / 4
        arc = radius * np.stack([np.cos(th), np.sin(th)], 1) @ rot.T
        P = (1 - rho[:, None, None]) * arc[None] + rho[:, None, None] * side[None]
        add_block(P + np.array([cx, cy]))
    xy = np.concatenate(pts)
    quads = np.concatenate(quads)
    key = np.round(xy / (h * 1e-6)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(first)
    renum = np.empty(len(first), np.int64)
    renum[order] = np.arange(len(first))
    xy = xy[first[order]]
    quads = renum[inv[quads]]
    dz = depth or h
    npt = len(xy)
    nodes = np.concatenate([np.column_stack([xy, np.zeros(npt)]),
                            np.column_stack([xy, np.full(npt, dz)])])
    hexes = np.concatenate([quads, quads + npt], 1)
    tol = 1e-6 * h

    def rule(c, nrm):
        which = np.full(len(c), -1)
        which[np.abs(c[:, 0]) < tol] = 0
        which[np.abs(c[:, 0] - length) < tol] = 1
        which[(np.abs(c[:, 1]) < tol) | (np.abs(c[:, 1] - height) < tol)] = 2
        which[np.hypot(c[:, 0] - cx, c[:, 1] - cy) < radius + 0.5 * h] = 3
        which[np.abs(c[:, 2]) < tol] = 4
        which[np.abs(c[:, 2] - dz) < tol] = 5
        names = [("inlet", "velocity-inlet"), ("outlet", "pressure-outlet"), ("walls", "wall"),
                 ("cylinder", "wall"), ("front", "symmetry"), ("back", "symmetry")]
        return {"which": which, "names": names}

    return _finish(nodes, kind, hexes, rule)


# ---------------------------------------------------------------- unstructured tetrahedra

def _delaunay(points: np.ndarray, rule, min_quality: float = 1e-6) -> UnstructuredMesh:
    """Tetrahedra of the Delaunay triangulation of ``points``; flat hull slivers dropped."""
    from scipy.spatial import Delaunay

    tri = Delaunay(points).simplices.astype(np.int64)
    x = points[tri]
    vol = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]),
                    x[:, 3] - x[:, 0]) / 6.0
    e = np.linalg.norm(x[:, [1, 2, 3, 2, 3, 3]] - x[:, [0, 0, 0, 1, 1, 2]], axis=2).max(1)
    keep = np.abs(vol) > min_quality * e ** 3
    tets = tri[keep]
    used = np.unique(tets)
    renum = -np.ones(len(points), np.int64)
    renum[used] = np.arange(len(used))
    nodes = points[used]
    mesh = UnstructuredMesh(nodes, {"tet": _orient(nodes, "tet", renum[tets])})
    mesh.zones = _zones_from_faces(mesh, rule)
    return mesh


def delaunay_box(n=(8, 8, 8), lengths=(1.0, 1.0, 1.0), *, jitter: float = 0.3, seed: int = 0,
                 origin=(0.0, 0.0, 0.0)) -> UnstructuredMesh:
    """Unstructured tetrahedra filling a box: Delaunay of a lattice jittered by ``jitter`` × h.

    Boundary points stay on their faces (jittered within them), so the box
    faces are exact. Zones as :func:`box`.
    """
    rng = np.random.default_rng(seed)
    xs = [np.linspace(o, o + L, k + 1) for o, L, k in zip(origin, lengths, n)]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    lo = np.asarray(origin, float)
    hi = lo + np.asarray(lengths, float)
    h = np.asarray(lengths, float) / np.asarray(n, float)
    tol = 1e-9 * max(lengths)
    on_face = (np.abs(pts - lo) < tol) | (np.abs(pts - hi) < tol)
    d = rng.uniform(-jitter, jitter, pts.shape) * h
    d[on_face] = 0.0                                    # stay on the faces (and edges, corners)
    pts = pts + d

    def rule(c, nrm):
        which = np.full(len(c), -1)
        for s, (ax, side) in enumerate([(0, lo), (0, hi), (1, lo), (1, hi), (2, lo), (2, hi)]):
            which[np.abs(c[:, ax] - side[ax]) < tol] = s
        return {"which": which,
                "names": [(nm, "wall") for nm in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")]}

    return _delaunay(pts, rule)


def delaunay_tube(radius: float = 0.5, length: float = 4.0, *, h: float = 0.1,
                  jitter: float = 0.3, seed: int = 0) -> UnstructuredMesh:
    """Unstructured tetrahedra filling a circular pipe along z (zones as :func:`tube`).

    Points: concentric rings of spacing ``h`` in each cross-section, on
    planes ``h`` apart, jittered inside. Wall points lie on the circle and
    end points on the end planes, so the wall is a polygon of side ``≈ h``.
    """
    rng = np.random.default_rng(seed)
    nr = max(1, int(round(radius / h)))
    nz = max(1, int(round(length / h)))
    sec = [np.zeros((1, 2))]
    for i in range(1, nr + 1):
        r = radius * i / nr
        m = max(6, int(round(2 * np.pi * r / h)))
        th = (np.arange(m) + 0.5 * (i % 2)) * 2 * np.pi / m
        sec.append(np.stack([r * np.cos(th), r * np.sin(th)], 1))
    sec = np.concatenate(sec)
    wall = np.isclose(np.hypot(sec[:, 0], sec[:, 1]), radius)
    zs = np.linspace(0.0, length, nz + 1)
    pts, fixed = [], []
    for k, z in enumerate(zs):
        pts.append(np.column_stack([sec, np.full(len(sec), z)]))
        fixed.append(np.stack([wall, wall, np.full(len(sec), k in (0, nz))], 1))
    pts = np.concatenate(pts)
    fixed = np.concatenate(fixed)
    d = rng.uniform(-jitter, jitter, pts.shape) * h
    d[fixed] = 0.0
    # interior points must stay inside the wall polygon
    pts = pts + d
    r = np.hypot(pts[:, 0], pts[:, 1])
    inner = ~np.concatenate([wall] * (nz + 1))
    shrink = np.where(inner & (r > radius * np.cos(np.pi / 6) - 0.5 * h),
                      (radius * np.cos(np.pi / 6) - 0.5 * h) / np.maximum(r, 1e-300), 1.0)
    pts[:, :2] *= shrink[:, None]
    tol = 1e-9 * max(radius, length)

    def rule(c, nrm):
        which = np.full(len(c), 2)
        which[np.abs(c[:, 2]) < tol] = 0
        which[np.abs(c[:, 2] - length) < tol] = 1
        return {"which": which, "names": [("inlet", "velocity-inlet"),
                                           ("outlet", "pressure-outlet"), ("wall", "wall")]}

    return _delaunay(pts, rule)


__all__ = ["box", "cylinder_channel", "delaunay_box", "delaunay_tube", "fda_nozzle",
           "hex_to_pyramids", "hex_to_wedges", "tube", "wedges_to_tets"]
