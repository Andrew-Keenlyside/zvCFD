"""Momentum entering through pressure boundaries, beyond ``ṁ_b u_node``.

Inside the domain the advected value at a flux point is the upwind node's
value plus the deferred correction ``β ∇u_up · (x_ip − x_up)``, so every
interior face carries momentum to second order. Through a pressure boundary
the momentum crossing a node's control volume is ``ṁ_b u_node``, first
order. Where the flow enters, the node's outgoing corrections then lack
their counterpart.

The solvers' default (``GPUSolver._own_inflow``) keeps those corrections out
of the inflow node's own row and lets its boundary flux carry them. That is
conservative, and consistent where the faces leaving the node mirror its
boundary faces (prism and hexahedral inflow rows). Where they do not (a
tetrahedral inflow row), :func:`inflow_offsets` and :func:`inflow_cross_stream`
add the cross-stream part of the difference, a geometric term. The part along
the boundary normal is left out: the faces leaving the node lie half a cell
downstream, and with the node's one-sided gradient that half-cell would be
differenced centrally, which is unstable at cell Péclet numbers above about 2.

:func:`boundary_moments` is the full second-order reconstruction
(``boundary_reconstruction = True``, off by default). Each boundary sub-face
carries its share of the node's boundary mass flow, in proportion to ``ρ u·S``
at its centroid, at ``u_node + β_sf ∇u_node · (x_sf − x_node)``, while the node's
outgoing corrections stay in its row. That brings the boundary rows'
truncation error to the interior level (0.00102 at the inlet against 0.00103 in
the next row, on a 9.5 k-node pipe in developed flow), and where it converges it
is accurate: within 0.5 % of Stokes on a tetrahedral pipe, 0.03 % on an
extruded one. But it keeps the streamwise part, so on coarse meshes it is
unstable: a 6-cell-across extruded pipe (cell Péclet number about 8) diverges
with exact or inexact linear solves, explicit or implicit, limited or not, and
with false time steps down to 0.03. It is verified (the implicit form equals the
explicit one to 10⁻¹⁶; the GPU and CPU solvers agree).

The functions take the array module (NumPy or CuPy) where they run per
iteration, so the CPU reference and the GPU solver evaluate the same terms.
"""

from __future__ import annotations

import numpy as np

from zvcfd.fv.geometry import boundary_subface_weights


def boundary_advection_tables(nodes: np.ndarray, sub: dict, bcs: dict) -> dict | None:
    """Per-sub-face arrays of every pressure zone, flattened (``None`` if there is none).

    ``nodes``: node coordinates ``(N, 3)``; ``sub``: ``{zone: (faces (F, 4), S (F, 4, 3))}``
    with ``-1`` padding; ``bcs``: the solver's boundary conditions.

    Returns ``{"node" (M,), "fn" (M, 4), "W" (M, 4), "S" (M, 3), "dx" (M, 3),
    "outflow_only" (M,)}``: the sub-face's node, its face's nodes and the
    weights interpolating to the sub-face's area centroid, its area vector,
    the centroid's offset from the node, and whether only outflow carries
    momentum there (``backflow="outflow"``).
    """
    parts = []
    for zid, (f, S) in sub.items():
        spec = bcs.get(zid, {})
        if spec.get("kind") != "pressure":
            continue
        ok = f >= 0
        W = boundary_subface_weights(f)                          # (F, 4, 4)
        fz = np.where(ok, f, 0)
        xc = np.einsum("fij,fjk->fik", W, nodes[fz])            # sub-face centroids
        fi, ki = np.nonzero(ok)
        if not len(fi):                       # a partition may hold none of the zone's faces
            continue
        parts.append({
            "node": f[fi, ki], "fn": fz[fi], "W": W[fi, ki] * ok[fi],
            "S": S[fi, ki], "dx": xc[fi, ki] - nodes[f[fi, ki]],
            "outflow_only": np.full(len(fi), spec.get("backflow", "consistent") != "consistent")})
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def neighbour_minmax(xp, indptr, indices, U):
    """``(lo, hi)`` ``(n, 3)``: the extremes of ``U`` over each node and its neighbours (the
    pattern ``indptr``, ``indices`` includes the node itself)."""
    n = len(indptr) - 1
    rows = xp.repeat(xp.arange(n), xp.diff(indptr).astype(int)) if xp is np else \
        xp.searchsorted(indptr, xp.arange(indices.size), side="right") - 1
    lo, hi = U.copy(), U.copy()
    vals = U[indices]
    if xp is np:
        for k in range(U.shape[1]):
            np.minimum.at(lo[:, k], rows, vals[:, k])
            np.maximum.at(hi[:, k], rows, vals[:, k])
    else:
        import cupyx

        for k in range(U.shape[1]):
            cupyx.scatter_min(lo[:, k], rows, vals[:, k])
            cupyx.scatter_max(hi[:, k], rows, vals[:, k])
    return lo, hi


def inflow_offsets(mesh, T: dict | None, sym=None) -> np.ndarray:
    """``Δd (N, 3)``: at each pressure-boundary node, the cross-stream offset between the
    flux points of the faces leaving it into the domain and its boundary sub-faces.

    The default inflow treatment (``GPUSolver._own_inflow``) keeps the deferred
    corrections of the faces an inflow node is upwind of out of its own row, and
    lets its boundary flux carry them. That boundary flux is consistent where those
    faces mirror the node's boundary faces. In general they do not, and
    ``|ṁ_b| β ∇u_node · Δd`` restores the cross-stream part of the difference. Both
    means are geometric: the faces leaving the node weighted by their area along
    the inward normal ``n``, the sub-faces by area. Only the part normal to ``n``
    is kept. Along ``n``, the half-cell would be differenced centrally from the
    node's one-sided gradient, which is unstable at cell Péclet numbers above
    about 2 (the full second-order reconstruction, :func:`boundary_moments`, does
    that).

    ``Δd`` is zero on extruded meshes (prism and hexahedral inflow rows), where the
    faces leaving a node mirror its boundary faces. ``sym`` ``(N, 3)`` bool drops the
    offsets normal to axis-aligned symmetry planes, as in :func:`boundary_moments`.
    """
    from zvcfd.fv.geometry import element_points, ip_areas, ip_points, topology

    N = mesh.n_nodes
    d = np.zeros((N, 3))
    if T is None or not len(T["node"]):
        return d
    node, S, dx = T["node"], T["S"], T["dx"]
    nout = np.zeros((N, 3))
    np.add.at(nout, node, S)
    nrm = np.linalg.norm(nout, axis=1)
    isb = nrm > 0
    nin = np.zeros((N, 3))
    nin[isb] = -nout[isb] / nrm[isb, None]
    a = np.linalg.norm(S, axis=1)
    wb = np.bincount(node, a, N)
    db = np.zeros((N, 3))
    np.add.at(db, node, a[:, None] * dx)
    wo = np.zeros(N)
    do = np.zeros((N, 3))
    X = mesh.nodes
    for kind, e in mesh.elements.items():
        e = e[isb[e].any(1)]                      # only elements touching the boundary nodes
        if not len(e):
            continue
        topo = topology(kind)
        P = element_points(X[e], topo)
        A, xip = ip_areas(P, topo), ip_points(P, topo)
        for s, (la, lb) in enumerate(topo.edges):
            for n_, sign in ((e[:, la], 1.0), (e[:, lb], -1.0)):   # area out of the node
                w = np.maximum(sign * np.einsum("ek,ek->e", A[:, s], nin[n_]), 0.0)
                np.add.at(wo, n_, w)
                np.add.at(do, n_, w[:, None] * (xip[:, s] - X[n_]))
    ok = isb & (wo > 0) & (wb > 0)
    d[ok] = do[ok] / wo[ok, None] - db[ok] / wb[ok, None]
    d -= np.einsum("nk,nk->n", d, nin)[:, None] * nin
    if sym is not None:
        d = np.where(np.asarray(sym, bool), 0.0, d)
    return d


def inflow_cross_stream(xp, d, mb, gradU, beta):
    """``(N, 3)``: ``|ṁ_b| β ∇u_k · Δd`` per node and component (``mb < 0`` where flow
    enters; zero elsewhere). Subtract it from the momentum right-hand side at the
    nodes of ``GPUSolver._own_inflow``."""
    bt = beta if xp.ndim(beta) else xp.full(d.shape, float(beta))
    return xp.maximum(-mb, 0.0)[:, None] * bt * xp.einsum("nkj,nj->nk", gradU, d)


def boundary_moments(xp, T: dict, U, mb, gradU, beta, lo, hi, rho: float, n: int, sym=None,
                     bsf=None):
    """``D (n, 3, 3)``: ``D[i, k] = Σ_sf ṁ_sf β_sf,k (x_sf − x_i)``, so that the correction of
    velocity component ``k`` at node ``i`` is ``∇u_k,i · D[i, k]``.

    ``T``: :func:`boundary_advection_tables` as ``xp`` arrays; ``U`` ``(n, 3)``;
    ``mb`` ``(n,)`` the boundary outflow per node (inflow negative); ``gradU``
    ``(n, 3, 3)`` with ``du_j/dx_k`` at ``[:, j, k]``; ``beta`` the scheme's
    blend, a number or ``(n, 3)``; ``lo``, ``hi`` from :func:`neighbour_minmax`.
    ``β_sf,k`` is the smaller of ``β`` and the Barth–Jespersen factor of the
    extrapolation to the sub-face: the reconstructed boundary value stays
    within its neighbours' range. Unlimited, the reconstruction of momentum
    entering through an open boundary can feed itself (a specified blend of 1
    on the benchmark pipe ran away).

    ``sym`` ``(n, 3)`` bool marks nodes on axis-aligned symmetry planes. There
    the boundary patch is half of the mirrored one, so its offsets normal to
    the plane are dropped: the mirrored half would cancel them.

    Returns ``(D, bsf)``: ``bsf`` ``(M, 3)`` are the sub-faces' factors. Pass them
    back as ``bsf`` to freeze them, as the solvers freeze High Resolution's
    limiter: recomputed every outer iteration, the non-smooth limiter keeps
    the iteration in a limit cycle (``lo``, ``hi`` are then unused).
    """
    node = T["node"]
    if node.size == 0:
        return xp.zeros((n, 3, 3)), xp.zeros((0, 3))
    us = xp.einsum("mj,mjk->mk", T["W"], U[T["fn"]])
    q = rho * xp.einsum("mk,mk->m", us, T["S"])
    area = xp.sqrt(xp.einsum("mk,mk->m", T["S"], T["S"]))
    qsum = xp.bincount(node, weights=q, minlength=n)[node]
    asum = xp.bincount(node, weights=area, minlength=n)[node]
    qabs = xp.bincount(node, weights=xp.abs(q), minlength=n)[node]
    # the node's boundary flow shared as rho u.S; by area where that sum vanishes
    good = xp.abs(qsum) > 1e-12 * xp.maximum(qabs, 1e-300)
    w = xp.where(good, q / xp.where(good, qsum, 1.0), area / xp.maximum(asum, 1e-300))
    msf = mb[node] * w
    msf = xp.where(T["outflow_only"], xp.maximum(msf, 0.0), msf)
    dx = T["dx"]
    if bsf is None:
        delta = xp.einsum("mkj,mj->mk", gradU[node], dx)
        u = U[node]
        tiny = 1e-300
        r = xp.where(delta > tiny, (hi[node] - u) / xp.where(delta > tiny, delta, 1.0),
                     xp.where(delta < -tiny,
                              (lo[node] - u) / xp.where(delta < -tiny, delta, 1.0), 1.0))
        bnode = beta[node] if xp.ndim(beta) else xp.full((len(node), 3), float(beta))
        bsf = xp.minimum(xp.clip(r, 0.0, 1.0), bnode)
    D = xp.zeros((n, 3, 3))
    for k in range(3):
        for j in range(3):
            D[:, k, j] = xp.bincount(node, weights=msf * bsf[:, k] * dx[:, j], minlength=n)
    if sym is not None:
        for ax in range(3):
            D[:, :, ax] = xp.where(sym[:, ax][:, None], 0.0, D[:, :, ax])
    return D, bsf


def boundary_advection_correction(xp, D, gradU):
    """The deferred part ``Σ_sf ṁ_sf β_sf ∇u_node · (x_sf − x_node)`` per node ``(n, 3)``
    from :func:`boundary_moments`. Momentum leaving the node's control volume is
    positive: subtract it from the momentum right-hand side (explicit form). The
    solvers add it to the matrix instead, through the nodal-gradient operator: the
    term is linear in the nodal velocities."""
    return xp.einsum("nkj,nkj->nk", gradU, D)


__all__ = ["boundary_advection_correction", "boundary_advection_tables", "boundary_moments",
           "inflow_cross_stream", "inflow_offsets", "neighbour_minmax"]
