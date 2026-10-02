"""The SimVascular case for the finite-volume solver: SimVascular's own mesh, cropped.

The finite-volume solver runs on SimVascular's tetrahedral mesh
(``mesh-complete.mesh.vtu``), cropped to the two coronary trees, so the
two codes share the mesh as well as the geometry. The crop cuts the mesh
as :func:`vmrcase.cut_tree` cuts the surface: at each cut it removes the
tetrahedra whose centroid lies in the slab just upstream of the plane
(``cut.slab`` thick, within ``cut.disc`` of the path), which separates
the tree from the aorta, and keeps the face-connected pieces that carry a
coronary outlet cap. The faces exposed at the planes form the cut zones;
every one lies within one tetrahedron of its plane (checked). SimVascular's
own solution supplies the boundary data node by node: its velocity on the
cut zones (velocity inlets) and its pressure on the 24 outlet caps
(pressure outlets), every 5 ms, interpolated linearly in
time, and its solution at the start of the cycle as the initial state.

Model coordinates are centimetres; the finite-volume mesh is built in
metres. With ``store=`` the cropped mesh is written once as a Zarr Vectors
mesh collection (:mod:`zvcfd.io.mesh_collection`) and the solver's mesh is
the one read back from it. ``node_sv`` maps each finite-volume node to SimVascular's result
node (the result file numbers nodes differently from the mesh file).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmrcase as vc  # noqa: E402

MESH_DIR = vc.SIM / "mesh-complete"
TET_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


@dataclass
class FVCase:
    mesh: object                  # zvcfd.mesh.core.UnstructuredMesh, metres
    node_sv: np.ndarray           # (N,) SimVascular result node of each mesh node
    kept: np.ndarray              # (E_full,) which of the full mesh's tetrahedra are kept
    cut_zones: dict               # zone id -> "velocity" or "pressure"
    outlet_zones: dict            # zone id -> cap name


def _cut_trees(full, cent_cm: np.ndarray, log) -> np.ndarray:
    """Tetrahedra kept: downstream of both cuts, face-connected to a coronary outlet cap."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    slab = np.zeros(len(cent_cm), bool)
    for cut in vc.CUTS:
        c, n = vc.path_frame(cut.path, cut.arc)
        d = cent_cm - c
        sd = d @ n
        lateral = np.sqrt(np.maximum((d * d).sum(1) - sd * sd, 0.0))
        slab |= (sd < 0) & (sd > -cut.slab) & (lateral < cut.disc)
    cand = np.flatnonzero(~slab)
    tets = full.elements["tet"][cand]
    N = np.int64(len(full.nodes))
    if N >= 1 << 21:
        raise ValueError("node ids too large for 63-bit face keys")
    f = np.sort(tets[:, TET_FACES].reshape(-1, 3), 1).astype(np.int64)
    key = (f[:, 0] * N + f[:, 1]) * N + f[:, 2]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = np.flatnonzero(ks[1:] == ks[:-1])
    owner = order // 4
    a, b = owner[same], owner[same + 1]
    E = len(cand)
    ncomp, lab = connected_components(sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(E, E)),
                                      directed=False)
    pos = -np.ones(len(cent_cm), np.int64)
    pos[cand] = np.arange(E)
    names = vc.face_names()
    outlet_cells = [z.cells for z in full.zones.values()
                    if z.kind != "wall" and z.name in names.values()
                    and z.name not in ("inflow", "aorta")]
    oc = pos[np.concatenate(outlet_cells)]
    keep_comp = np.unique(lab[oc[oc >= 0]])
    kept = np.zeros(len(cent_cm), bool)
    kept[cand[np.isin(lab, keep_comp)]] = True
    log(f"slab cut: {slab.sum():,} tetrahedra in the slabs; {ncomp} pieces, "
        f"{len(keep_comp)} with coronary outlets kept")
    return kept


def build(region=None, cut_kind=None, log=print, store=None) -> FVCase:
    """Crop SimVascular's mesh to the cut coronary trees (or a sub-region of them).

    Args:
        region: optional ``f(centroids_cm) -> bool mask`` restricting the kept
            tetrahedra further (a short segment for tests).
        cut_kind: optional ``f(face_centroids_cm) -> list of (name, kind)`` naming
            each crop face's zone. Default: the two inlet cuts, split by the
            nearer cut centre, both velocity zones.
        store: optional ``.zvmesh`` path: the mesh is written there (unless a
            collection is already there) and read back, so the solver runs on
            the Zarr Vectors store.
    """
    from scipy.spatial import cKDTree
    from svref import SVVolume

    from zvcfd.mesh.core import BoundaryZone, UnstructuredMesh
    from zvcfd.mesh.vtk import read_vtu_mesh

    full = read_vtu_mesh(MESH_DIR / "mesh-complete.mesh.vtu", MESH_DIR, unit="cm")
    if set(full.elements) != {"tet"}:
        raise ValueError(f"expected tetrahedra only, got {sorted(full.elements)}")
    tets = full.elements["tet"]
    cent = full.nodes[tets].mean(1)
    kept = _cut_trees(full, cent, log)
    if region is not None:
        kept &= region(cent)
    log(f"kept {kept.sum():,} of {len(tets):,} tetrahedra")
    new_of_old_e = -np.ones(len(tets), np.int64)
    new_of_old_e[kept] = np.arange(kept.sum())
    kt = tets[kept]
    used = np.unique(kt)
    new_of_old_n = -np.ones(len(full.nodes), np.int64)
    new_of_old_n[used] = np.arange(len(used))
    nodes_cm = full.nodes[used]
    elem = new_of_old_n[kt]

    zones, names = {}, vc.face_names()
    outlet_zones = {}
    orig_keys = []
    for zid, z in full.zones.items():
        keep = new_of_old_e[z.cells] >= 0
        if not keep.any():
            continue
        f = z.faces[keep].copy()
        f[:, :3] = new_of_old_n[f[:, :3]]
        if (f[:, :3] < 0).any():
            raise ValueError(f"zone {z.name}: a kept face uses a dropped node")
        zones[zid] = BoundaryZone(zid, z.kind, z.name, f, new_of_old_e[z.cells[keep]])
        orig_keys.append(np.sort(f[:, :3], 1))
        if z.name in names.values() and z.kind != "wall":
            outlet_zones[zid] = z.name
    # boundary faces of the kept set that are not already zone faces: the crop
    allf = elem[:, TET_FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(len(elem)), 4)
    key = np.sort(allf, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = cnt[inv.reshape(-1)] == 1
    ok = np.concatenate(orig_keys) if orig_keys else np.zeros((0, 3), np.int64)
    from zvcfd.mesh.fluent import match_rows

    on_zone = match_rows(ok, key[bnd]) >= 0
    cf = allf[bnd][~on_zone]
    co = owner[bnd][~on_zone]
    p = nodes_cm[cf]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    flip = np.einsum("ij,ij->i", nrm, p.mean(1) - nodes_cm[elem[co]].mean(1)) < 0
    cf[flip] = cf[flip][:, [0, 2, 1]]
    fc = p.mean(1)
    if cut_kind is None:
        centres = {c.name: np.array(vc.path_frame(c.path, c.arc)[0]) for c in vc.CUTS}
        labels = [min(centres, key=lambda k: np.linalg.norm(x - centres[k])) for x in fc]
        if region is None:                         # every crop face lies on its cut plane
            for cut in vc.CUTS:
                c, nrm = vc.path_frame(cut.path, cut.arc)
                sel = np.array([lab == cut.name for lab in labels])
                d = np.abs((fc[sel] - c) @ nrm)
                log(f"  {cut.name}: crop faces within {d.max() * 10:.2f} mm of the plane")
                if d.max() > 0.1:
                    raise ValueError(f"{cut.name}: crop faces {d.max():.2f} cm from the cut plane")
        labels = [(n, "velocity") for n in labels]
    else:
        labels = cut_kind(fc)
    cut_zones = {}
    next_id = max(full.zones) + 1
    for name, kind in sorted(set(labels)):
        sel = np.array([lab == (name, kind) for lab in labels])
        fk = np.concatenate([cf[sel], -np.ones((sel.sum(), 1), np.int64)], 1)
        zones[next_id] = BoundaryZone(next_id, "velocity-inlet" if kind == "velocity"
                                      else "pressure-outlet", name, fk, co[sel])
        cut_zones[next_id] = kind
        log(f"  crop zone {name} ({kind}): {sel.sum():,} faces")
        next_id += 1
    mesh = UnstructuredMesh(nodes_cm * vc.UNIT_SCALE, {"tet": elem}, zones=zones, unit="m",
                            source=f"{MESH_DIR} (cropped)")
    if store is not None:
        mesh = _through_store(mesh, Path(store), log)
    sv = SVVolume()
    d, node_sv = cKDTree(sv.points).query(nodes_cm)
    if d.max() > 5e-4:
        raise ValueError(f"mesh node {d.argmax()} is {d.max():.2e} cm from any result node")
    log(f"{mesh.n_nodes:,} nodes, {len(elem):,} tetrahedra, {len(zones)} zones")
    return FVCase(mesh, node_sv, kept, cut_zones, outlet_zones)


def _through_store(mesh, store: Path, log):
    """``mesh`` written to (or found at) a mesh collection, and read back from it."""
    import time

    from zvcfd.io.mesh_collection import is_mesh_collection, read_mesh, write_mesh_collection

    if not is_mesh_collection(store):
        t = time.time()
        write_mesh_collection(store, mesh)
        log(f"wrote {store} ({time.time() - t:.1f} s)")
    t = time.time()
    back = read_mesh(store)
    same = (np.array_equal(back.nodes, mesh.nodes)
            and all(np.array_equal(back.elements[k], mesh.elements[k]) for k in mesh.elements)
            and sorted(back.zones) == sorted(mesh.zones)
            and all(back.zones[z].n_faces == mesh.zones[z].n_faces for z in mesh.zones))
    if not same:
        raise ValueError(f"{store} does not hold this case's mesh (delete it to rewrite)")
    log(f"read the solver's mesh back from {store.name} ({time.time() - t:.1f} s)")
    return back


class SVBoundaryData:
    """SimVascular's solution at the mesh's nodes, at any time (linear between saved steps)."""

    def __init__(self, case: FVCase):
        from scipy.spatial import cKDTree
        from svref import SVVolume

        self.sv = SVVolume()
        self.case = case
        self.tree = cKDTree(case.mesh.nodes)
        self.dt_saved = float(self.sv.t[1] - self.sv.t[0])
        self._cache = {}
        self._weights = {}

    def _step(self, i: int) -> np.ndarray:
        i %= len(self.sv.t) - 1                      # the last saved step closes the period
        if i not in self._cache:
            if len(self._cache) > 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = np.asarray(self.sv.up[i])[self.case.node_sv]   # (N, 4)
        return self._cache[i]

    def at_nodes(self, t: float) -> np.ndarray:
        ph = (t % vc.PERIOD) / self.dt_saved
        i0 = int(np.floor(ph))
        w = ph - i0
        return (1 - w) * self._step(i0) + w * self._step(i0 + 1)

    def weights_of(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mesh nodes and linear weights ``(M, 4)`` of boundary points.

        The solver asks for boundary values at the zones' nodes and at the
        centroids of each face's sub-faces, which it forms from the face's
        nodes with fixed weights (:func:`zvcfd.fv.geometry.boundary_subface_weights`).
        Both are matched exactly, so the values are the linear interpolation
        of SimVascular's nodal solution on the face.
        """
        from scipy.spatial import cKDTree

        x = np.asarray(x, float).reshape(-1, 3)
        key = x.tobytes()
        if key in self._weights:
            return self._weights[key]
        if getattr(self, "_subtree", None) is None:
            from zvcfd.fv.geometry import boundary_subface_weights

            P = self.case.mesh.nodes
            nodes = [np.repeat(np.arange(len(P))[:, None], 4, 1)]
            weights = [np.tile([1.0, 0.0, 0.0, 0.0], (len(P), 1))]
            for z in self.case.mesh.zones.values():
                f = z.faces
                ok = f >= 0
                W = boundary_subface_weights(f)                       # (F, 4 sub, 4 nodes)
                fz = np.where(ok, f, f[:, :1])
                nodes.append(np.repeat(fz[:, None, :], 4, 1)[ok])
                weights.append(W[ok])
            self._subn = np.concatenate(nodes)
            self._subw = np.concatenate(weights)
            self._subtree = cKDTree((P[self._subn] * self._subw[..., None]).sum(1))
        d, i = self._subtree.query(x)
        if d.max() > 1e-12:
            raise ValueError(f"boundary point {d.max():.2e} m from every node and sub-face")
        out = (self._subn[i], self._subw[i])
        self._weights[key] = out
        return out

    def _interp(self, x, t, cols):
        nodes, w = self.weights_of(x)
        v = self.at_nodes(t)[:, cols].astype(float)
        return (v[nodes] * w[..., None]).sum(1) if v.ndim == 2 else (v[nodes] * w).sum(1)

    def velocity(self, x, t):
        return self._interp(x, t, slice(0, 3))

    def pressure(self, x, t):
        return self._interp(x, t, 3)


def boundary_conditions(case: FVCase, data: SVBoundaryData, *, backflow: float | None = None):
    """The ``{zone id: spec}`` table: walls, SimVascular velocity on velocity cuts,
    SimVascular pressure on the outlets and on pressure cuts."""
    bcs = {}
    for zid, z in case.mesh.zones.items():
        if z.kind == "wall":
            bcs[zid] = {"kind": "wall"}
        elif case.cut_zones.get(zid) == "velocity":
            bcs[zid] = {"kind": "velocity", "value": data.velocity}
        else:
            bcs[zid] = {"kind": "pressure", "value": data.pressure}
            if backflow:
                bcs[zid]["backflow_stabilisation"] = backflow
    return bcs
