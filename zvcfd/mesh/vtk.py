"""VTK volume meshes, as SimVascular writes them (``mesh-complete``), and VTK export.

SimVascular's ``mesh-complete/`` folder holds the volume mesh
(``mesh-complete.mesh.vtu``: tetrahedra with a 1-based ``GlobalNodeID`` per
point and ``GlobalElementID`` per cell) and one surface per face
(``mesh-surfaces/*.vtp``, plus ``walls_combined.vtp``). Each surface
triangle carries its nodes' ``GlobalNodeID`` and its owning element's
``GlobalElementID``, so boundary zones are rebuilt exactly, with no
geometric matching. Needs pyvista (``pip install "zvcfd[simvascular]"``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zvcfd.mesh.core import KINDS, VTK_TYPE, BoundaryZone, UnstructuredMesh, to_vtk

_KIND_OF_VTK = {v: k for k, v in VTK_TYPE.items()}


def _orient(nodes, kind, e):
    from zvcfd.mesh.generate import _orient as orient

    if kind == "pyramid":
        x = nodes[e]
        s = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 4] - x[:, 0])
        e[s < 0] = e[s < 0][:, [0, 3, 2, 1, 4]]
        return e
    return orient(nodes, kind, e)


def read_vtu_mesh(vtu, surfaces: dict[str, str | Path] | str | Path | None = None, *,
                  kinds: dict[str, str] | None = None, unit: str | None = None,
                  wall: str = "wall") -> UnstructuredMesh:
    """Read a VTK volume mesh and, optionally, its boundary surfaces as zones.

    Args:
        vtu: the volume mesh (``.vtu``).
        surfaces: ``{zone name: .vtp path}``, or a SimVascular ``mesh-complete``
            folder (its ``walls_combined.vtp`` becomes zone ``wall`` and each
            ``mesh-surfaces/*.vtp`` that is not a wall becomes a cap zone).
        kinds: zone name -> kind (``wall``, ``velocity-inlet``, ``pressure-outlet``);
            unnamed caps default to ``pressure-outlet``, and a cap called
            ``inflow`` or ``inlet`` to ``velocity-inlet``.
        unit: coordinate unit to record (SimVascular models are usually ``cm``).
    """
    import pyvista as pv

    g = pv.read(str(vtu))
    pts = np.asarray(g.points, np.float64)
    gid = np.asarray(g.point_data["GlobalNodeID"], np.int64) - 1 \
        if "GlobalNodeID" in g.point_data else np.arange(g.n_points)
    nodes = np.zeros((gid.max() + 1, 3))
    nodes[gid] = pts
    conn = np.asarray(g.cell_connectivity, np.int64)
    offs = np.asarray(g.cell_offsets if hasattr(g, "cell_offsets") else g.offset, np.int64)
    types = np.asarray(g.celltypes)
    elements, cell_ids = {}, {}
    for t in np.unique(types):
        if int(t) not in _KIND_OF_VTK:
            raise NotImplementedError(f"VTK cell type {int(t)} is not supported")
        kind = _KIND_OF_VTK[int(t)]
        sel = np.flatnonzero(types == t)
        n = offs[sel[0] + 1] - offs[sel[0]]
        e = gid[conn[offs[sel][:, None] + np.arange(n)]]
        elements[kind] = _orient(nodes, kind, e)
        cell_ids[kind] = sel
    mesh = UnstructuredMesh(nodes, elements, cell_ids=cell_ids, unit=unit, source=str(vtu))
    if surfaces is None:
        return mesh
    # file cell index -> global element index
    elem_of_cell = np.empty(g.n_cells, np.int64)
    off = mesh.element_offsets()
    for kind in KINDS:
        if kind in cell_ids:
            elem_of_cell[cell_ids[kind]] = off[kind] + np.arange(len(cell_ids[kind]))
    cell_of_eid = None
    if "GlobalElementID" in g.cell_data:
        eid = np.asarray(g.cell_data["GlobalElementID"], np.int64)
        cell_of_eid = np.full(eid.max() + 1, -1, np.int64)
        cell_of_eid[eid] = np.arange(g.n_cells)
    if not isinstance(surfaces, dict):
        folder = Path(surfaces)
        surfaces = {wall: folder / "walls_combined.vtp"}
        for f in sorted((folder / "mesh-surfaces").glob("*.vtp")):
            if not f.stem.startswith("wall"):
                surfaces[f.stem] = f
    kinds = dict(kinds or {})
    cent = mesh.all_centroids()
    for zid, (name, path) in enumerate(surfaces.items(), start=1):
        s = pv.read(str(path)).triangulate()
        tri = np.asarray(s.faces, np.int64).reshape(-1, 4)[:, 1:]
        sg = np.asarray(s.point_data["GlobalNodeID"], np.int64) - 1
        f = np.concatenate([sg[tri], -np.ones((len(tri), 1), np.int64)], 1)
        if cell_of_eid is not None and "GlobalElementID" in s.cell_data:
            own = elem_of_cell[cell_of_eid[np.asarray(s.cell_data["GlobalElementID"], np.int64)]]
        else:
            raise ValueError(f"surface {name}: no GlobalElementID to find owning elements")
        p = nodes[f[:, :3]]
        nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        flip = np.einsum("ij,ij->i", nrm, p.mean(1) - cent[own]) < 0
        f[flip, :3] = f[flip][:, [0, 2, 1]]
        default = "wall" if name == wall else (
            "velocity-inlet" if name.lower() in ("inflow", "inlet") else "pressure-outlet")
        mesh.zones[zid] = BoundaryZone(zid, kinds.get(name, default), name, f, own)
    return mesh


def write_vtu(mesh: UnstructuredMesh, path, point_data: dict | None = None) -> None:
    """Write the volume mesh (and node fields) as ``.vtu``."""
    import pyvista as pv

    cells, types = to_vtk(mesh)
    g = pv.UnstructuredGrid(cells, types, mesh.nodes)
    for k, v in (point_data or {}).items():
        g.point_data[k] = np.asarray(v)
    g.save(str(path))


__all__ = ["read_vtu_mesh", "write_vtu"]
