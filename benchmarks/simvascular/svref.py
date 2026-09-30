"""SimVascular's answers for the cut coronary trees, from the VMR result files.

    python benchmarks/simvascular/svref.py surface   # cap flows and pressures -> svsurface.npz
    python benchmarks/simvascular/svref.py probes    # probe points shared by both codes
    python benchmarks/simvascular/svref.py volume    # SimVascular at the probes -> svprobes.npz
    python benchmarks/simvascular/svref.py wall      # its own TAWSS, from P1 gradients

The result files hold the last cardiac cycle (steps 10020-11020 of
11020, t = 10.02-11.02 s) every 5 ms, in CGS units (cm, cm/s, dyn/cm^2).
Everything written here is SI, with time folded into one period
(t = 0 at step 10020).

``surface``: from the surface results, per outlet cap, the volume flow out
(velocity integrated over the cap's triangles) and the area-weighted mean
pressure; the time-averaged wall shear stress (``vTAWSS``) and OSI.

``volume``: from the volume results, the velocity and pressure on each
inlet's cut plane (for zvCFD's inlet profile and the pressure reference),
and the velocity and pressure fields sampled at given points.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmrcase as vc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results" / "simvascular"
DYN = 0.1                    # Pa per dyn/cm^2
CM = 1e-2                    # m per cm
STEP0, DSTEP, DT_SV = 10020, 5, 1e-3


def _steps(names) -> list[int]:
    return sorted({int(m[1]) for k in names for m in [re.match(r"pressure_(\d+)$", k)] if m})


def surface(out: Path = RES / "svsurface.npz") -> dict:
    import pyvista as pv

    t0 = time.time()
    res = pv.read(vc.RESULTS_VTP)
    geo = pv.read(vc.SURFACE)
    names = vc.face_names()
    # the result surface is the mesh surface, renumbered: map through global ids
    order = np.argsort(geo.cell_data["GlobalElementID"])
    gpos = np.searchsorted(geo.cell_data["GlobalElementID"], res.cell_data["GlobalElementID"],
                           sorter=order)
    fid = np.asarray(geo.cell_data["ModelFaceID"])[order[gpos]]
    faces = res.faces.reshape(-1, 4)[:, 1:]
    pts = np.asarray(res.points)
    a, b, c = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    area_vec = 0.5 * np.cross(b - a, c - a)                  # cm^2, orientation per triangle
    area = np.linalg.norm(area_vec, axis=1)
    steps = _steps(res.point_data.keys())
    caps = sorted(k for k in np.unique(fid) if k != vc.WALL_ID)
    cap_names = [names[k] for k in caps]
    # outward normal of each cap: away from the model's interior near the cap
    q = np.zeros((len(steps), len(caps)))
    p = np.zeros((len(steps), len(caps)))
    for j, k in enumerate(caps):
        sel = fid == k
        nvec = area_vec[sel].sum(0)
        nvec /= np.linalg.norm(nvec)
        cent = (((a + b + c)[sel] / 3) * area[sel, None]).sum(0) / area[sel].sum()
        # the wall points nearest the cap sit inside it: outward points away from them
        wall_near = pts[faces[fid == vc.WALL_ID].reshape(-1)]
        d = np.linalg.norm(wall_near - cent, axis=1)
        inner = wall_near[np.argsort(d)[:200]].mean(0)
        if (cent - inner) @ nvec < 0:
            nvec = -nvec
        tri = faces[sel]
        w = area[sel] / 3.0
        for i, s in enumerate(steps):
            v = np.asarray(res.point_data[f"velocity_{s}"])
            pr = np.asarray(res.point_data[f"pressure_{s}"])
            vn = v[tri] @ nvec                               # (T, 3) normal velocity at vertices
            q[i, j] = (vn.mean(1) * area[sel]).sum() * CM ** 3            # m^3/s, out
            p[i, j] = (pr[tri] * w[:, None]).sum() / area[sel].sum() * DYN
    t = (np.array(steps) - STEP0) * DT_SV
    tawss = np.asarray(res.point_data["vTAWSS"]) * DYN
    osi = np.asarray(res.point_data["vOSI"])
    wall_pts = np.unique(faces[fid == vc.WALL_ID])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, t=t, caps=np.array(cap_names), q_out=q, p=p,
                        points=pts * CM, tawss=tawss, osi=osi, wall_points=wall_pts,
                        faces=faces, face_id=fid)
    print(f"{len(steps)} steps, {len(caps)} caps -> {out} ({time.time() - t0:.0f} s)")
    return {"t": t, "caps": cap_names, "q_out": q, "p": p}


CACHE = vc.VMR_DIR / "sv_cache"


class SVVolume:
    """SimVascular's volume solution: velocity and pressure at any point, any saved step.

    The first use reads the 8.5 GB result file once and caches its nodes,
    tetrahedra, and velocity + pressure for every step as a float32
    ``(steps, nodes, 4)`` memmap (SI units). Values at a point are the
    linear (P1) interpolation in the tetrahedron containing it, which is
    how svSolver represents its solution.
    """

    def __init__(self):
        if not (CACHE / "up.npy").exists():
            self._build_cache()
        self.points = np.load(CACHE / "points.npy")                  # cm
        self.tets = np.load(CACHE / "tets.npy")
        self.t = np.load(CACHE / "t.npy")
        self.up = np.load(CACHE / "up.npy", mmap_mode="r")           # (S, N, 4): u (m/s), p (Pa)
        self._grid = None

    @staticmethod
    def _build_cache():
        import pyvista as pv

        t0 = time.time()
        m = pv.read(vc.RESULTS_VTU)
        if not (m.celltypes == 10).all():
            raise ValueError("expected a tetrahedral mesh")
        steps = _steps(m.point_data.keys())
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(CACHE / "points.npy", np.asarray(m.points, np.float64))
        np.save(CACHE / "tets.npy", m.cells.reshape(-1, 5)[:, 1:].astype(np.int64))
        np.save(CACHE / "t.npy", (np.array(steps) - STEP0) * DT_SV)
        up = np.lib.format.open_memmap(CACHE / "up.npy", mode="w+", dtype=np.float32,
                                       shape=(len(steps), m.n_points, 4))
        for i, s in enumerate(steps):
            up[i, :, :3] = np.asarray(m.point_data[f"velocity_{s}"]) * CM
            up[i, :, 3] = np.asarray(m.point_data[f"pressure_{s}"]) * DYN
        up.flush()
        print(f"cached {len(steps)} steps of {m.n_points} nodes ({time.time() - t0:.0f} s)")

    def locate(self, xyz_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Containing tetrahedron and barycentric weights ``(N, 4)``; cell -1 outside."""
        import pyvista as pv

        if self._grid is None:
            self._grid = pv.UnstructuredGrid(
                np.hstack([np.full((len(self.tets), 1), 4), self.tets]).ravel(),
                np.full(len(self.tets), 10, np.uint8), self.points)
        xyz = np.asarray(xyz_cm, np.float64)
        cells = np.asarray(self._grid.find_containing_cell(xyz), np.int64)
        w = np.zeros((len(xyz), 4))
        ok = cells >= 0
        v = self.points[self.tets[cells[ok]]]                         # (n, 4, 3)
        tmat = np.transpose(v[:, 1:] - v[:, :1], (0, 2, 1))           # (n, 3, 3)
        lam = np.linalg.solve(tmat, (xyz[ok] - v[:, 0])[..., None])[..., 0]
        w[ok] = np.c_[1 - lam.sum(1), lam]
        return cells, w

    def sample(self, located, steps=None) -> np.ndarray:
        """``(len(steps), N, 4)`` velocity and pressure at located points; NaN outside."""
        cells, w = located
        steps = range(len(self.t)) if steps is None else steps
        ok = cells >= 0
        nodes = self.tets[np.maximum(cells, 0)]
        out = np.full((len(steps), len(cells), 4), np.nan, np.float32)
        for i, s in enumerate(steps):
            f = np.asarray(self.up[s])
            out[i, ok] = (f[nodes[ok]] * w[ok, :, None]).sum(1)
        return out


def inlet_profiles(xyz_cm: np.ndarray, n_in: np.ndarray) -> np.ndarray:
    """SimVascular's velocity along ``n_in`` at the points, every saved step: ``(S, N)`` m/s.

    Points outside SimVascular's mesh (voxel centres just beyond its wall)
    take 0, the wall velocity.
    """
    sv = SVVolume()
    u = sv.sample(sv.locate(xyz_cm))[..., :3] @ np.asarray(n_in, float)
    return np.nan_to_num(u, nan=0.0)


def volume(out: Path = RES / "svprobes.npz") -> None:
    """SimVascular at the probe points (every step) and on each inlet cut (mean pressure)."""
    sv = SVVolume()
    pr = np.load(RES / "probes.npz", allow_pickle=True)
    t0 = time.time()
    loc = sv.locate(pr["points"])
    vals = sv.sample(loc)
    # the cut planes: SimVascular's flow and mean pressure over each inlet's section,
    # sliced from the uncut surface (the cut surface's cap lies in the plane itself)
    import pyvista as pv

    whole = pv.read(vc.SURFACE)
    cut_q, cut_p = [], []
    for cut in vc.CUTS:
        c, n = vc.path_frame(cut.path, cut.arc)
        pts, uv, _, _ = _section_points(whole, cut.path, cut.arc)
        v = sv.sample(sv.locate(pts))
        area = len(uv) * SECTION_STEP ** 2 * CM ** 2
        cut_q.append(np.nanmean(v[..., :3] @ n, 1) * area)
        cut_p.append(np.nanmean(v[..., 3], 1))
    np.savez_compressed(out, t=sv.t, u=vals[..., :3], p=vals[..., 3], outside=loc[0] < 0,
                        cut_names=np.array([c.name for c in vc.CUTS]), cut_q=np.array(cut_q),
                        cut_p=np.array(cut_p))
    print(f"{len(loc[0])} probes ({(loc[0] < 0).sum()} outside), {len(sv.t)} steps -> {out} "
          f"({time.time() - t0:.0f} s)")


def wall_shear(out: Path = RES / "svwall.npz") -> None:
    """SimVascular's own wall shear stress, from its P1 velocity gradients.

    For every wall triangle of the cut trees, the tetrahedron behind it
    gives a constant velocity gradient G (linear elements); the traction is
    ``mu (G + G^T) n`` and its tangential part the wall shear stress. Face
    values are time-averaged in magnitude over the saved steps (TAWSS) and
    area-averaged to the nodes. The result file's own ``vTAWSS`` array is
    all zeros, so this is computed here.
    """
    from scipy.spatial import cKDTree

    sv = SVVolume()
    s = np.load(RES / "svsurface.npz")
    pr = np.load(RES / "probes.npz", allow_pickle=True)
    spts = s["points"] / CM
    faces, fid = s["faces"], s["face_id"]
    wall_set = np.zeros(len(spts), bool)
    wall_set[pr["wall_node"]] = True
    tri = faces[(fid == vc.WALL_ID) & wall_set[faces].all(1)]
    d, vol_id = cKDTree(sv.points).query(spts)
    if d[np.unique(tri)].max() > 5e-4:
        raise ValueError("surface nodes do not match volume nodes")
    vtri = np.sort(vol_id[tri], 1)
    n_nodes = len(sv.points)

    def key(a):
        return (a[:, 0] * n_nodes + a[:, 1]) * n_nodes + a[:, 2]

    tets = sv.tets
    fk, ft = [], []
    for drop in range(4):
        f = np.sort(np.delete(tets, drop, axis=1), 1)
        fk.append(key(f))
        ft.append(np.arange(len(tets)))
    fk, ft = np.concatenate(fk), np.concatenate(ft)
    o = np.argsort(fk)
    fk, ft = fk[o], ft[o]
    pos = np.searchsorted(fk, key(vtri))
    if not (fk[np.minimum(pos, len(fk) - 1)] == key(vtri)).all():
        raise ValueError("a wall face has no tetrahedron")
    tet = tets[ft[pos]]
    x = sv.points[tet] * CM                                           # (F, 4, 3) m
    jac = np.transpose(x[:, 1:] - x[:, :1], (0, 2, 1))                # columns: edges
    inv = np.linalg.inv(jac)                                          # rows: grad of phi_1..3
    grad = np.concatenate([-inv.sum(1, keepdims=True), inv], 1)       # (F, 4, 3)
    p0, p1, p2 = (sv.points[vtri[:, k]] * CM for k in range(3))
    nrm = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= (2 * area)[:, None]
    fourth = x.mean(1)                                                # inside the fluid
    nrm *= np.sign(((p0 - fourth) * nrm).sum(1))[:, None]             # outward
    ta = np.zeros(len(tet))
    for i in range(len(sv.t)):
        u = np.asarray(sv.up[i])[tet, :3]                             # (F, 4, 3)
        g = np.einsum("fki,fkj->fij", u, grad)                        # du_i/dx_j
        trac = vc.MU * np.einsum("fij,fj->fi", g + np.transpose(g, (0, 2, 1)), nrm)
        wss = trac - (trac * nrm).sum(1, keepdims=True) * nrm
        ta += np.linalg.norm(wss, axis=1)
    ta /= len(sv.t)
    node_sum = np.zeros(len(spts))
    node_w = np.zeros(len(spts))
    for k in range(3):
        np.add.at(node_sum, tri[:, k], ta * area)
        np.add.at(node_w, tri[:, k], area)
    tawss = np.where(node_w > 0, node_sum / np.maximum(node_w, 1e-30), np.nan)
    np.savez_compressed(out, tawss_fe=tawss, faces=tri, face_tawss=ta)
    print(f"{len(tri)} wall faces -> {out}")


# Cross-sections for velocity profiles: (label, path, arc length in cm)
SECTIONS = (("LM", "LAD", 2.05), ("LAD proximal", "LAD", 3.0), ("LCX proximal", "LCX", 0.8),
            ("LAD mid", "LAD", 6.0), ("RCA proximal", "RCA", 1.3), ("RCA mid", "RCA", 5.0))
H_WALL = 0.015               # cm: near-wall probes at H and 2H inward along the wall normal
SECTION_STEP = 0.004         # cm: probe spacing on a cross-section


def _section_points(surf, path: str, arc: float):
    """Grid points (x, y, z, cm) inside the vessel's cross-section at ``arc`` along ``path``."""
    from matplotlib.path import Path as MPath

    c, t = vc.path_frame(path, arc)
    e1 = np.cross(t, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(t, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(t, e1)
    conn = surf.slice(normal=t, origin=c).connectivity("all")
    rid = np.asarray(conn.point_data["RegionId"])
    pts = np.asarray(conn.points)
    for k in np.unique(rid):
        uv = np.c_[(pts[rid == k] - c) @ e1, (pts[rid == k] - c) @ e2]
        cc = uv.mean(0)
        loop = uv[np.argsort(np.arctan2(uv[:, 1] - cc[1], uv[:, 0] - cc[0]))]
        poly = MPath(loop)
        if not poly.contains_point((0.0, 0.0)):
            continue
        lo, hi = loop.min(0), loop.max(0)
        g = np.stack(np.meshgrid(np.arange(lo[0], hi[0], SECTION_STEP),
                                 np.arange(lo[1], hi[1], SECTION_STEP)), -1).reshape(-1, 2)
        g = g[poly.contains_points(g)]
        return c + g[:, :1] * e1 + g[:, 1:] * e2, g, t, loop
    raise ValueError(f"no cross-section at {path} {arc}")


def probes(out: Path = RES / "probes.npz") -> None:
    """Near-wall and cross-section probe points, shared by both codes."""
    import pyvista as pv

    surf, _, _ = vc.coronary_surface()
    sv = np.load(RES / "svsurface.npz")
    pts_cm = sv["points"] / CM
    fid = sv["face_id"]
    faces = sv["faces"]
    # wall nodes of the result surface that belong to the cut trees
    from scipy.spatial import cKDTree

    wall = np.unique(faces[fid == vc.WALL_ID])
    d, _ = cKDTree(np.asarray(surf.points)).query(pts_cm[wall])
    wall = wall[d < 5e-4]                   # the result file rounds coordinates (<= 1 um)
    # outward normals on the result surface, and away from caps and cuts
    res = pv.PolyData(pts_cm, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    res = res.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True,
                              consistent_normals=True, split_vertices=False)
    nrm = np.asarray(res.point_data["Normals"])[wall]
    cap_pts = np.asarray(surf.points)[np.unique(surf.faces.reshape(-1, 4)[:, 1:][
        np.asarray(surf.cell_data["ModelFaceID"]) != vc.WALL_ID])]
    dc, _ = cKDTree(cap_pts).query(pts_cm[wall])
    keep = dc > 3 * H_WALL + 0.02
    wall, nrm = wall[keep], nrm[keep]
    p1 = pts_cm[wall] - H_WALL * nrm
    p2 = pts_cm[wall] - 2 * H_WALL * nrm
    sec_pts, sec_id, sec_uv, sec_t, sec_loop = [], [], [], [], []
    for k, (label, path, arc) in enumerate(SECTIONS):
        p, uv, t, loop = _section_points(surf, path, arc)
        sec_pts.append(p)
        sec_id.append(np.full(len(p), k))
        sec_uv.append(uv)
        sec_t.append(t)
        sec_loop.append(loop)
    points = np.vstack([p1, p2] + sec_pts)
    kind = np.r_[np.full(len(p1), 1), np.full(len(p2), 2), np.full(sum(map(len, sec_pts)), 3)]
    loops = np.empty(len(sec_loop), object)
    loops[:] = sec_loop
    np.savez_compressed(out, points=points, kind=kind, wall_node=wall, normal=nrm, h=H_WALL,
                        section=np.concatenate(sec_id), section_uv=np.vstack(sec_uv),
                        section_t=np.array(sec_t), section_loops=loops,
                        section_labels=np.array([s[0] for s in SECTIONS]))
    print(f"{len(wall)} wall nodes x 2 depths, {sum(map(len, sec_pts))} section points -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["surface", "probes", "volume", "wall"])
    args = ap.parse_args()
    if args.what == "surface":
        surface()
    elif args.what == "probes":
        probes()
    elif args.what == "volume":
        volume()
    elif args.what == "wall":
        wall_shear()


if __name__ == "__main__":
    main()
