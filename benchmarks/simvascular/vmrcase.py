"""The SimVascular comparison case: VMR model 0066_H_CORO_H, its coronary trees cut from the aorta.

The Vascular Model Repository's model 0066_H_CORO_H (healthy coronary
tree, CT) comes with a SimVascular (svSolver) rigid-wall simulation:
Newtonian blood (rho = 1.06 g/cm^3, mu = 0.04 P), a plug inflow at the
aortic root, an RCR outlet on the aorta and open-loop coronary outlet
models (intramyocardial pressure) on the 24 coronary outlets; 11 cardiac
cycles of 1 s at dt = 1 ms. zvCFD has no coronary outlet model yet, so
the comparison is a *submodel*: the left and right coronary trees are cut
from the aorta a short way past their ostia, and both codes' solutions
are compared inside the cut trees, with zvCFD driven by SimVascular's own
flow through each cut and pressure at each outlet.

Files (Stanford Digital Repository, druid dh173bw0673, public):

    <VMR_DIR>/0066_H_CORO_H/...                     the SimVascular project (unzipped)
    <VMR_DIR>/0066_H_CORO_H_3D_RIGID.vtp            surface results, time-resolved
    <VMR_DIR>/0066_H_CORO_H_3D_RIGID.vtu            volume results, time-resolved

``VMR_DIR`` defaults to ``/hdd/data/zvcfd_vmr``. Model coordinates are in
centimetres; everything this module returns is in centimetres too, and
``UNIT_SCALE`` converts to metres.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VMR_DIR = Path(os.environ.get("VMR_DIR", "/hdd/data/zvcfd_vmr"))
MODEL = "0066_H_CORO_H"
PROJECT = VMR_DIR / MODEL
SIM = PROJECT / "Simulations" / "0002_0001"
SURFACE = SIM / "mesh-complete" / "mesh-complete.exterior.vtp"
RESULTS_VTP = VMR_DIR / f"{MODEL}_3D_RIGID.vtp"
RESULTS_VTU = VMR_DIR / f"{MODEL}_3D_RIGID.vtu"
DOWNLOAD = "https://stacks.stanford.edu/file/druid:dh173bw0673"

UNIT_SCALE = 1e-2            # metres per model unit (cm)
RHO = 1060.0                 # kg/m^3 (1.06 g/cm^3)
MU = 0.004                   # Pa s (0.04 P)
NU = MU / RHO                # 3.77e-6 m^2/s
PERIOD = 1.0                 # s

WALL_ID = 1
INLET_IDS = {"LCA_inlet": 101, "RCA_inlet": 102}


@dataclass(frozen=True)
class Cut:
    """A plane across a coronary artery: the tree downstream of it is kept."""

    name: str                # inlet patch name
    path: str                # centreline path that runs through the ostium
    arc: float               # cm along the path from its first point
    disc: float              # cm: radius of the cut disc, between the vessel and the next surface
    tree: str                # "LCA" or "RCA": which outlets belong downstream
    slab: float = 0.3        # cm removed upstream of the plane; > the surface edges (<= 1.3 mm)

    @property
    def face_id(self) -> int:
        return INLET_IDS[self.name]


# Where each plane cuts only its own vessel (slices of the surface along the
# paths): the left main 1.8 cm along the LAD path, past where it separates
# from the aortic sinus wall (1.5 cm) and ~6 mm before the LAD/LCX
# bifurcation (2.34 cm); its section spans 1.5-3.4 mm from the path point
# and the next surface in the plane is 5.8 mm away. The proximal RCA 0.95 cm
# along its path, ~6.5 mm before its first branch: 1.3-2.1 mm, next surface
# 5.7 mm.
CUTS = (Cut("LCA_inlet", "LAD", 1.80, 0.45, "LCA"), Cut("RCA_inlet", "RCA", 0.95, 0.38, "RCA"))


def face_names() -> dict[int, str]:
    """ModelFaceID -> face name, from the SimVascular model file."""
    text = (PROJECT / "Models" / "0002_0001.mdl").read_text()
    return {int(m[1]): m[2] for m in re.finditer(r'face id="(\d+)" name="([^"]+)"', text)}


def outlet_tree(name: str) -> str:
    return "RCA" if name.startswith("RCA") else "LCA"


def read_path(name: str) -> np.ndarray:
    """Centreline path points ``(N, 3)`` (x, y, z, cm)."""
    text = (PROJECT / "Paths" / f"{name}.pth").read_text()
    return np.array([[float(m[1]), float(m[2]), float(m[3])] for m in
                     re.finditer(r'<pos x="([^"]+)" y="([^"]+)" z="([^"]+)"', text)])


def path_frame(name: str, arc: float) -> tuple[np.ndarray, np.ndarray]:
    """Point and unit tangent (downstream) of a path at arc length ``arc`` (cm)."""
    p = read_path(name)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    at = np.array([np.interp(arc, s, p[:, k]) for k in range(3)])
    a0 = np.array([np.interp(arc - 0.05, s, p[:, k]) for k in range(3)])
    a1 = np.array([np.interp(arc + 0.05, s, p[:, k]) for k in range(3)])
    t = a1 - a0
    return at, t / np.linalg.norm(t)


def _order_loop(edges: np.ndarray) -> np.ndarray:
    """Point ids of one closed loop from its (unordered) edges."""
    nxt: dict[int, list[int]] = {}
    for a, b in edges:
        nxt.setdefault(int(a), []).append(int(b))
        nxt.setdefault(int(b), []).append(int(a))
    start = int(edges[0, 0])
    loop, prev, cur = [start], -1, start
    while True:
        cand = [v for v in nxt[cur] if v != prev]
        if not cand:
            break
        prev, cur = cur, cand[0]
        if cur == start:
            break
        loop.append(cur)
    if len(loop) != len(nxt):
        raise ValueError(f"cut boundary is not one loop ({len(loop)} of {len(nxt)} points)")
    return np.array(loop)


def cut_tree(surface, cut: Cut, names: dict[int, str]):
    """The coronary tree downstream of ``cut`` as a closed, zoned surface (pyvista PolyData).

    A thin slab (``cut.slab``) on the upstream side of the plane, within
    ``cut.disc`` of the path point, is clipped away (interpolated on edges,
    so the downstream cut is planar). That separates the tree from the
    aorta; the piece connected to the tree's outlets is kept, and its planar
    hole is capped by a fan around its centroid with
    ``ModelFaceID = cut.face_id``.
    """
    import pyvista as pv

    c, n = path_frame(cut.path, cut.arc)
    pts = np.asarray(surface.points)
    s = (pts - c) @ n
    lateral = np.sqrt(np.maximum(((pts - c) ** 2).sum(1) - s * s, 0.0))
    f = np.maximum.reduce([s, -(s + cut.slab), lateral - cut.disc])
    work = surface.copy()
    work.point_data["cutf"] = f
    kept = work.clip_scalar(scalars="cutf", value=0.0, invert=False).extract_surface(
        algorithm="dataset_surface")
    fid = np.asarray(kept.cell_data["ModelFaceID"])
    outlets = [k for k, v in names.items() if k != WALL_ID and outlet_tree(v) == cut.tree
               and v not in ("inflow", "aorta")]
    seed = kept.extract_cells(np.flatnonzero(fid == outlets[0])).center
    tree = kept.connectivity("closest", closest_point=seed).extract_surface(
        algorithm="dataset_surface")
    tree = tree.triangulate().clean()
    tfid = np.asarray(tree.cell_data["ModelFaceID"])
    present = set(np.unique(tfid).tolist())
    missing = [names[k] for k in outlets if k not in present]
    stray = [names.get(k, str(k)) for k in present if k not in outlets and k != WALL_ID]
    if missing or stray:
        raise ValueError(f"{cut.name}: outlets missing {missing}, foreign faces {stray}")
    # the cut boundary: edges used by exactly one triangle
    faces = tree.faces.reshape(-1, 4)[:, 1:]
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    loop = _order_loop(ue[cnt == 1])
    lp = np.asarray(tree.points)[loop]
    off = np.abs((lp - c) @ n).max()
    if off > 1e-6:
        raise ValueError(f"{cut.name}: cut boundary leaves the plane by {off:g} cm")
    centre = lp.mean(0)
    k0 = tree.n_points
    cap_pts = np.vstack([tree.points, centre[None]])
    fan = np.stack([np.full(len(loop), k0), loop, np.roll(loop, -1)], 1)
    # orient the cap so its normal points out of the fluid (upstream, -n)
    a, b, cc = cap_pts[fan[:, 0]], cap_pts[fan[:, 1]], cap_pts[fan[:, 2]]
    if (np.cross(b - a, cc - a).sum(0) @ n) > 0:
        fan = fan[:, [0, 2, 1]]
    all_faces = np.vstack([faces, fan])
    out = pv.PolyData(cap_pts, np.hstack([np.full((len(all_faces), 1), 3), all_faces]).ravel())
    out.cell_data["ModelFaceID"] = np.r_[tfid, np.full(len(fan), cut.face_id)].astype(np.int32)
    info = {"centre_cm": c.tolist(), "normal": n.tolist(), "loop_points": int(len(loop)),
            "cap_area_cm2": float(0.5 * np.linalg.norm(np.cross(b - a, cc - a), axis=1).sum()),
            "outlets": [names[k] for k in outlets]}
    return out, info


def coronary_surface():
    """Both coronary trees, cut and capped: ``(PolyData, names, info)``.

    ``names`` maps every ModelFaceID in the result to its name (the two new
    inlets included).
    """
    import pyvista as pv

    surface = pv.read(SURFACE)
    names = face_names()
    parts, info = [], {}
    for cut in CUTS:
        part, inf = cut_tree(surface, cut, names)
        parts.append(part)
        info[cut.name] = inf
    merged = parts[0].merge(parts[1:], merge_points=False).extract_surface(
        algorithm="dataset_surface")
    names = dict(names) | {v: k for k, v in INLET_IDS.items()}
    return merged, names, info


def zone_kinds(names: dict[int, str], present) -> dict[int, str]:
    """Fluent-style kinds for :func:`zvcfd.geometry.voxelize_mesh`."""
    out = {}
    for k in present:
        nm = names[int(k)]
        out[int(k)] = ("wall" if k == WALL_ID else
                       "velocity-inlet" if nm in INLET_IDS else "pressure-outlet")
    return out


def triangles(poly) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 3, 3)`` triangle coordinates (x, y, z, cm) and each triangle's face id."""
    faces = poly.faces.reshape(-1, 4)
    if not (faces[:, 0] == 3).all():
        raise ValueError("surface is not all triangles")
    return np.asarray(poly.points)[faces[:, 1:]], np.asarray(poly.cell_data["ModelFaceID"])
