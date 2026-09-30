"""3-D renderings for docs/validation/simvascular.md (pyvista, off screen).

    python benchmarks/simvascular/renders.py [--best vmr0066-60um-sv] [--ms 530]

- ``simvascular_model.png``: the VMR model, and the two coronary trees cut
  from it (inlets marked);
- ``zvcfd_streamlines.png``: zvCFD streamlines at one phase, traced
  through zvCFD's own voxel field (trilinear, RK2), coloured by speed;
- ``simvascular_tawss_maps.png``: TAWSS from both codes on the same wall,
  and their difference;
- ``zvcfd_voxels.png``: zvCFD's voxel geometry at the left-main
  bifurcation, faces coloured by zvCFD's pressure, with SimVascular's
  surface as a wireframe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compare as cmp  # noqa: E402
import figures as fg  # noqa: E402
import vmrcase as vc  # noqa: E402

FIG = fg.FIG
BG = fg.SURFACE
VIEW = (-1.0, 0.3, 0.5)


def _plotter(shape=(1, 1), size=(1500, 1100)):
    import pyvista as pv

    pv.global_theme.font.color = fg.INK2
    p = pv.Plotter(off_screen=True, shape=shape, window_size=size, border=False)
    p.set_background(BG)
    return p


def _bar(title, fmt="%.2f"):
    return {"title": title + "\n", "color": fg.INK2, "fmt": fmt, "vertical": True,
            "position_x": 0.88, "position_y": 0.2, "width": 0.04, "height": 0.6,
            "title_font_size": 18, "label_font_size": 15, "n_labels": 5}


def model():
    import pyvista as pv

    full = pv.read(vc.SURFACE)
    cut, names, info = vc.coronary_surface()
    fid = np.asarray(cut.cell_data["ModelFaceID"])
    p = _plotter()
    p.add_mesh(full, color="#c3c2b7", opacity=0.22, smooth_shading=True)
    p.add_mesh(cut.extract_cells(np.flatnonzero(fid == vc.WALL_ID)), color="#6f6e69",
               smooth_shading=True)
    p.add_mesh(cut.extract_cells(np.flatnonzero(fid > 100)), color=fg.SERIES[3])
    labels = [np.array(info[n]["centre_cm"]) for n in ("LCA_inlet", "RCA_inlet")]
    p.add_point_labels(np.array(labels), ["LCA inlet (left main cut)", "RCA inlet"],
                       font_size=20, text_color=fg.INK, shape_color=BG, shape_opacity=0.8,
                       point_color=fg.SERIES[3], point_size=12, always_visible=True)
    p.view_vector(VIEW)
    p.camera.zoom(1.35)
    p.screenshot(FIG / "simvascular_model.png")
    p.close()


class VoxelField:
    """Trilinear lookup in a zvCFD field snapshot (voxel centres, velocity, pressure)."""

    def __init__(self, path: Path):
        f = np.load(path)
        self.xyz = f["xyz"].astype(np.float64)
        self.u, self.p, self.flag = f["u"], f["p"], f["flag"]
        self.h = float(f["voxel"])
        self.x0 = self.xyz.min(0)
        ijk = np.rint((self.xyz - self.x0) / self.h).astype(np.int64)
        self.n = ijk.max(0) + 2
        key = self._key(ijk)
        self.order = np.argsort(key)
        self.sorted = key[self.order]

    def _key(self, ijk):
        return (ijk[:, 0] * self.n[1] + ijk[:, 1]) * self.n[2] + ijk[:, 2]

    def index(self, ijk) -> np.ndarray:
        """Row of each voxel ``(N, 3)``, -1 where there is none."""
        ok = ((ijk >= 0) & (ijk < self.n)).all(1)
        key = np.where(ok, self._key(np.clip(ijk, 0, self.n - 1)), -1)
        pos = np.clip(np.searchsorted(self.sorted, key), 0, len(self.sorted) - 1)
        hit = ok & (self.sorted[pos] == key)
        return np.where(hit, self.order[pos], -1)

    def velocity(self, pts) -> tuple[np.ndarray, np.ndarray]:
        v = (pts - self.x0) / self.h
        base = np.floor(v).astype(np.int64)
        fr = v - base
        acc = np.zeros((len(pts), 3))
        tot = np.zeros(len(pts))
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    idx = self.index(base + np.array([dz, dy, dx]))
                    w = ((fr[:, 0] if dz else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1])
                         * (fr[:, 2] if dx else 1 - fr[:, 2]))
                    w = np.where(idx >= 0, w, 0.0)
                    acc += w[:, None] * self.u[np.maximum(idx, 0)]
                    tot += w
        return acc / np.maximum(tot, 1e-12)[:, None], tot


def streamlines(fields: Path, n_seeds=500, seed=0):
    import pyvista as pv

    vf = VoxelField(fields)
    rng = np.random.default_rng(seed)
    inlet = np.flatnonzero(vf.flag == 4)
    seeds = vf.xyz[rng.choice(inlet, min(n_seeds, len(inlet)), replace=False)]
    step = 0.5 * vf.h
    lines, speeds = [], []
    pts = seeds.copy()
    alive = np.ones(len(pts), bool)
    tracks = [[p.copy()] for p in pts]
    track_speed = [[] for _ in pts]
    for _ in range(12000):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        u1, w1 = vf.velocity(pts[idx])
        s1 = np.linalg.norm(u1, axis=1)
        mid = pts[idx] + 0.5 * step * u1 / np.maximum(s1, 1e-12)[:, None]
        u2, w2 = vf.velocity(mid)
        s2 = np.linalg.norm(u2, axis=1)
        new = pts[idx] + step * u2 / np.maximum(s2, 1e-12)[:, None]
        stop = (w1 < 0.3) | (w2 < 0.3) | (s1 < 1e-4)
        for j, k in enumerate(idx):
            if stop[j]:
                alive[k] = False
                continue
            tracks[k].append(new[j].copy())
            track_speed[k].append(s1[j])
        pts[idx[~stop]] = new[~stop]
    for tr, sp in zip(tracks, track_speed):
        if len(tr) > 10:
            lines.append(np.array(tr[:-1]))
            speeds.append(np.array(sp))
    poly = pv.PolyData()
    allp = np.vstack(lines)
    off = np.cumsum([0] + [len(x) for x in lines])
    cells = np.hstack([np.r_[len(x), np.arange(off[i], off[i + 1])] for i, x in enumerate(lines)])
    poly = pv.PolyData(allp, lines=cells)
    poly.point_data["speed"] = np.concatenate(speeds)
    cut, _, _ = vc.coronary_surface()
    clim = (0, float(np.percentile(poly["speed"], 99)))
    focus = vc.read_path("LCX")[0]
    p = _plotter(shape=(1, 2), size=(2400, 1100))
    for c, (radius, zoom) in enumerate(((0.0035, 1.2), (0.004, None))):
        p.subplot(0, c)
        p.add_mesh(cut, color="#c3c2b7", opacity=0.15 if c == 0 else 0.25, smooth_shading=True)
        p.add_mesh(poly.tube(radius=radius), scalars="speed", cmap=fg.SEQ, clim=clim,
                   scalar_bar_args=_bar("zvCFD speed (m/s)"),
                   smooth_shading=True, show_scalar_bar=c == 0)
        p.view_vector(VIEW)
        if zoom:
            p.camera.zoom(zoom)
        else:
            p.set_focus(focus)
            p.camera.position = focus + 4.5 * np.array(VIEW) / np.linalg.norm(VIEW)
    p.screenshot(FIG / "zvcfd_streamlines.png")
    p.close()
    return len(lines)


def tawss_maps(zv, ref):
    import pyvista as pv

    wm = cmp.wall_metrics(zv, ref)
    pr = np.load(cmp.RES / "probes.npz", allow_pickle=True)
    s = np.load(cmp.RES / "svsurface.npz")
    pts, faces = s["points"] / 1e-2, s["faces"]
    surf = pv.PolyData(pts, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    keep = np.zeros(len(pts), bool)
    keep[pr["wall_node"]] = True
    tri_keep = keep[faces].all(1)
    vals = {}
    for key in ("tawss_sv", "tawss_zv"):
        a = np.full(len(pts), np.nan)
        a[pr["wall_node"]] = wm[key]
        vals[key] = a
    diff = (vals["tawss_zv"] - vals["tawss_sv"]) / np.nanmedian(wm["tawss_sv"]) * 100
    sub = surf.extract_cells(np.flatnonzero(tri_keep))
    ids = sub.point_data["vtkOriginalPointIds"]
    top = float(np.nanpercentile(wm["tawss_sv"], 98))
    p = _plotter(shape=(1, 3), size=(2400, 950))
    for c, (arr, title, bar, cmap, clim) in enumerate((
            (vals["tawss_sv"], "SimVascular", "TAWSS (Pa)", fg.SEQ, (0, top)),
            (vals["tawss_zv"], "zvCFD", "TAWSS (Pa)", fg.SEQ, (0, top)),
            (diff, "zvCFD − SimVascular", "% of median", fg.DIV, (-40, 40)))):
        p.subplot(0, c)
        m = sub.copy()
        m.point_data["v"] = arr[ids]
        p.add_mesh(m, scalars="v", cmap=cmap, clim=clim, nan_color="#c3c2b7",
                   scalar_bar_args=_bar(bar, "%.0f" if c == 2 else "%.1f") | {
                       "position_x": 0.82}, smooth_shading=True)
        p.add_text(title, position="upper_left", font_size=14, color=fg.INK)
        p.view_vector(VIEW)
        p.camera.zoom(1.3)
    p.screenshot(FIG / "simvascular_tawss_maps.png")
    p.close()


def voxels(fields: Path, radius=0.35):
    import pyvista as pv

    vf = VoxelField(fields)
    c = vc.read_path("LCX")[0]
    near = np.flatnonzero(np.linalg.norm(vf.xyz - c, axis=1) < radius)
    ijk = np.rint((vf.xyz[near] - vf.x0) / vf.h).astype(np.int64)
    quads = []
    h = vf.h
    corners = {0: [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)], 1: [(0, 0, 0), (0, 0, 1),
                                                                  (1, 0, 1), (1, 0, 0)],
               2: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]}
    for ax in range(3):
        for sgn in (-1, 1):
            d = np.zeros(3, np.int64)
            d[ax] = sgn
            nb = vf.index(ijk + d)
            open_ = nb < 0
            base = vf.xyz[near[open_]] - 0.5 * h + (h if sgn > 0 else 0.0) * (np.eye(3)[ax])
            quad = np.stack([base + h * np.array(q, float) for q in corners[ax]], 1)
            quads.append((quad, vf.p[near[open_]]))
    pts = np.vstack([q.reshape(-1, 3) for q, _ in quads])
    n = sum(len(q) for q, _ in quads)
    cells = np.hstack([np.full((n, 1), 4), np.arange(4 * n).reshape(n, 4)]).ravel()
    poly = pv.PolyData(pts, cells)
    poly.cell_data["p"] = np.concatenate([v for _, v in quads])
    surf = pv.read(vc.SURFACE)
    clip = surf.clip_box([c[0] - radius, c[0] + radius, c[1] - radius, c[1] + radius,
                          c[2] - radius, c[2] + radius], invert=False)
    p = _plotter()
    lo, hi = np.percentile(poly["p"], [2, 98])
    p.add_mesh(poly, scalars="p", cmap=fg.SEQ, clim=(lo, hi), show_edges=True,
               edge_color="#ffffff", line_width=0.3,
               scalar_bar_args=_bar("zvCFD pressure relative to p_ref (Pa)", "%.0f"))
    p.add_mesh(clip, style="wireframe", color=fg.INK2, opacity=0.35, line_width=0.6)
    p.set_focus(c)
    p.camera.position = c + 2.4 * np.array(VIEW) / np.linalg.norm(VIEW)
    p.screenshot(FIG / "zvcfd_voxels.png")
    p.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="vmr0066-60um-sv")
    ap.add_argument("--coarse", default="vmr0066-100um-sv")
    ap.add_argument("--ms", type=int, default=530)
    ap.add_argument("--runs", default=str(cmp.RUNS))
    ap.add_argument("--only", default="model,streamlines,tawss,voxels")
    args = ap.parse_args()
    only = set(args.only.split(","))
    runs = Path(args.runs)
    if "model" in only:
        model()
    if "streamlines" in only:
        print("streamlines:", streamlines(runs / args.best / f"fields_{args.ms:04d}.npz"))
    if "tawss" in only:
        tawss_maps(cmp.load_run(runs / args.best), cmp.sv_reference())
    if "voxels" in only:
        print("voxel faces:", voxels(runs / args.coarse / f"fields_{args.ms:04d}.npz"))
    print("renders ->", FIG)


if __name__ == "__main__":
    main()
