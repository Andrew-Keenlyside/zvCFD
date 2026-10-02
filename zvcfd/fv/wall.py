"""Wall shear stress as Ansys CFX evaluates it: velocity gradients at the wall integration points.

Each wall face is split among its nodes into sub-faces (the median dual of
:func:`zvcfd.fv.geometry.boundary_subfaces`), and each sub-face carries one
boundary integration point at its area centroid. There, the element that
owns the face gives the velocity gradient through its shape functions,

    ∇u_ip = Σ_n u_n ⊗ ∇N_n(ξ_ip),

and the wall shear stress is the tangential part of the viscous traction,

    t = μ (∇u + ∇uᵀ) · n,      τ = −(t − (t · n) n),

with ``n`` the sub-face's outward unit normal, so ``τ`` is the force per
area of the fluid on the wall (along the flow next to it). ``μ`` is the
viscosity at the integration point's shear rate for a shear-thinning fluid.
A wall node's value is the area-weighted mean over its wall sub-faces.

On linear tetrahedra the shape-function gradient is constant in the element,
so this is the element (P1) gradient, first order in the wall-normal size.
On prisms and hexahedra it varies within the element and is evaluated at the
integration point itself. The method is CFX's for laminar walls; its accuracy
rests on near-wall resolution (inflation layers), and
``benchmarks/fv/wall_shear_estimators.py`` measures it against exact solutions.
"""

from __future__ import annotations

import numpy as np

from zvcfd.fv.geometry import (
    XI_NODES,
    _reference_inverse,
    boundary_subface_weights,
    boundary_subfaces,
    shape,
)
from zvcfd.mesh.core import FACES, UnstructuredMesh


def _owners(mesh: UnstructuredMesh, faces: np.ndarray):
    """The element owning each boundary face: ``(kind per face, element row in that kind)``."""
    from zvcfd.mesh.fluent import match_rows

    on = np.zeros(mesh.n_nodes, bool)
    on[faces[faces >= 0]] = True
    rows, kinds, elems = [], [], []
    for k, e in mesh.elements.items():
        cand = np.flatnonzero(on[e].sum(1) >= 3)
        if not len(cand):
            continue
        for fl in FACES[k]:
            f = e[cand][:, list(fl)]
            if f.shape[1] == 3:
                f = np.concatenate([f, -np.ones((len(f), 1), np.int64)], 1)
            rows.append(f)
            kinds.append(np.full(len(f), list(mesh.elements).index(k)))
            elems.append(cand)
    rows, kinds, elems = np.concatenate(rows), np.concatenate(kinds), np.concatenate(elems)
    hit = match_rows(rows, faces)
    if (hit < 0).any():
        raise ValueError(f"{int((hit < 0).sum())} wall faces belong to no element")
    return kinds[hit], elems[hit]


class WallGradientShear:
    """The wall-ip gradient operator of a mesh's wall zones, ready to apply to velocities.

    Args:
        mesh: the (local) mesh.
        zone_ids: the wall zones.
        xp: ``numpy`` or ``cupy``: where the operator lives and evaluates.
    """

    def __init__(self, mesh: UnstructuredMesh, zone_ids, xp=np):
        self.xp, self.N = xp, mesh.n_nodes
        faces = np.concatenate([mesh.zones[z].faces for z in zone_ids])
        kind_of, elem = _owners(mesh, faces)
        kinds = list(mesh.elements)
        W = boundary_subface_weights(faces)                     # (F, 4 sub, 4 face nodes)
        S = boundary_subfaces(mesh.nodes, faces)                # (F, 4, 3), outward
        self.parts = []
        for ki, k in enumerate(kinds):
            fsel = np.flatnonzero(kind_of == ki)
            if not len(fsel):
                continue
            e = mesh.elements[k][elem[fsel]]                    # (Fk, n)
            f = faces[fsel]
            ok = f >= 0
            # local position of each face node in its element
            loc = np.argmax(e[:, None, :] == np.where(ok, f, -2)[:, :, None], axis=2)
            ref = XI_NODES[k][loc]                              # (Fk, 4, 3) reference geometry
            ref = np.where(ok[..., None], ref, 0.0)
            pts = np.einsum("fij,fjd->fid", W[fsel], ref)       # sub-face ips, reference geometry
            sub_ok = ok
            fi, si = np.nonzero(sub_ok)
            xi = _reference_inverse(k, pts[fi, si])
            _, dN = shape(k, xi)                                # (M, n, 3)
            x = mesh.nodes[e[fi]]                               # (M, n, 3)
            J = np.einsum("mnk,mnd->mkd", x, dN)
            G = np.einsum("mnd,mdk->mnk", dN, np.linalg.inv(J))
            a = S[fsel][fi, si]
            area = np.linalg.norm(a, axis=1)
            self.parts.append({
                "elem": xp.asarray(e[fi]), "G": xp.asarray(G),
                "n": xp.asarray(a / area[:, None]), "area": xp.asarray(area),
                "node": xp.asarray(f[fi, si])})
        wsum = np.zeros(self.N)
        for p in self.parts:
            np.add.at(wsum, _host(p["node"]), _host(p["area"]))
        self.area = xp.asarray(wsum)

    def __call__(self, U, viscosity=None, mu: float = 1.0):
        """Wall shear stress ``(N, 3)`` at every node (zero off the wall).

        ``viscosity``: a :class:`zvcfd.rheology.CarreauYasuda` (or any
        ``f(shear_rate)``), else the constant ``mu``.
        """
        xp = self.xp
        U = xp.asarray(U)
        acc = xp.zeros((self.N, 3))
        for p in self.parts:
            g = xp.einsum("mni,mnj->mij", U[p["elem"]], p["G"])    # du_i/dx_j at the ips
            D = g + xp.swapaxes(g, 1, 2)
            if viscosity is None:
                m = mu
            else:
                rate = xp.sqrt(0.5 * xp.einsum("mij,mij->m", D, D))
                m = _viscosity(viscosity, rate, xp)[:, None]
            t = m * xp.einsum("mij,mj->mi", D, p["n"])
            t = t - xp.einsum("mi,mi->m", t, p["n"])[:, None] * p["n"]
            w = -t * p["area"][:, None]
            for k in range(3):
                acc[:, k] += xp.bincount(p["node"], weights=w[:, k], minlength=self.N)
        return acc / xp.maximum(self.area, 1e-300)[:, None]


def _viscosity(v, rate, xp):
    if all(hasattr(v, a) for a in ("mu_0", "mu_inf", "lam", "a", "n")):
        return v.mu_inf + (v.mu_0 - v.mu_inf) * (1 + (v.lam * rate) ** v.a) ** ((v.n - 1) / v.a)
    return xp.asarray(v(_host(rate)))


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


__all__ = ["WallGradientShear"]
