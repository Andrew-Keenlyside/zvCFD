"""Pressure patches on caps that cut obliquely through the voxels: a straight-pipe test.

    python benchmarks/simvascular/oblique_outlet.py

A straight circular pipe (radius R voxels, length L), voxelised from a
triangulated surface with its two caps as pressure patches (dp between
them), along the lattice z axis and along two oblique directions. In
steady Poiseuille flow the pressure falls linearly along the axis; the
linear fit over the middle half of the pipe, extrapolated to each cap,
gives the pressure the flow actually feels there. The difference from the
imposed patch pressure is the patch's pressure jump. Flux is compared with
Hagen-Poiseuille through the voxelised cross-section area.

Found by the SimVascular comparison (docs/validation/simvascular.md),
whose 24 coronary outlets all cut the voxels obliquely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "benchmarks" / "results" / "simvascular" / "oblique_outlet.json"


def pipe_surface(axis, radius, length, n_theta=96, n_len=60):
    """Triangles (x, y, z) and zone ids (1 wall, 2 inlet cap, 3 outlet cap) of a closed pipe."""
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    e1 = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(-length / 2, length / 2, n_len + 1)
    ring = lambda z: z * a + radius * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)  # noqa: E731
    tris, zone = [], []
    for i in range(n_len):
        r0, r1 = ring(zs[i]), ring(zs[i + 1])
        for j in range(n_theta):
            k = (j + 1) % n_theta
            tris += [[r0[j], r1[j], r1[k]], [r0[j], r1[k], r0[k]]]
            zone += [1, 1]
    for z, zid, flip in ((zs[0], 2, True), (zs[-1], 3, False)):
        r, c = ring(z), z * a
        for j in range(n_theta):
            k = (j + 1) % n_theta
            tris.append([c, r[k], r[j]] if flip else [c, r[j], r[k]])
            zone.append(zid)
    return np.array(tris), np.array(zone)


def axis_capped(axis, radius, length):
    """The pipe along ``axis``, but with both caps cut perpendicular to the lattice z axis."""
    import pyvista as pv

    tri, zone = pipe_surface(axis, radius, length * 1.6, n_len=100)
    wall = tri[zone == 1]
    poly = pv.PolyData(wall.reshape(-1, 3),
                       np.hstack([np.full((len(wall), 1), 3),
                                  np.arange(3 * len(wall)).reshape(-1, 3)]).ravel()).clean()
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    zc = length / 2 * a[2]
    out = poly.clip(normal=(0, 0, 1), origin=(0, 0, zc)).clip(normal=(0, 0, -1),
                                                               origin=(0, 0, -zc))
    out = out.extract_surface(algorithm="dataset_surface").triangulate()
    pts = np.asarray(out.points)
    faces = out.faces.reshape(-1, 4)[:, 1:]
    tris = [pts[faces]]
    zones = [np.ones(len(faces), int)]
    for z, zid, flip in ((-zc, 2, True), (zc, 3, False)):
        e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        edge = ue[cnt == 1]
        edge = edge[np.abs(pts[edge][:, :, 2] - z).max(1) < 1e-6]
        c = pts[np.unique(edge)].mean(0)
        t = np.stack([np.broadcast_to(c, (len(edge), 3)), pts[edge[:, 0]], pts[edge[:, 1]]], 1)
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2]
        want = -1 if flip else 1
        t[n * want < 0] = t[n * want < 0][:, [0, 2, 1]]
        tris.append(t)
        zones.append(np.full(len(t), zid))
    return np.concatenate(tris), np.concatenate(zones)


def run_case(axis, radius=8.0, length=160.0, tau=0.8, dp_lat=1e-4, steps=40000, log=print,
             z_caps=False, scheme="abb"):
    """One pipe. ``scheme``: ``"abb"`` (patch links, the current condition) or ``"neem"``
    (non-equilibrium extrapolation on patch cells, the condition before the fix)."""
    from zvcfd.boundary import Patch, cell_coords, from_cells
    from zvcfd.geometry import (
        VoxelGrid,
        cap_links,
        domain_from_voxels,
        patch_cells,
        voxelize_surface,
    )
    from zvcfd.lbm import SparseLBM

    tri, zone = axis_capped(axis, radius, length) if z_caps else pipe_surface(axis, radius, length)
    tri_zyx = tri[..., ::-1] + 1000.0                                  # keep coordinates positive
    grid = VoxelGrid.around(tri_zyx.reshape(-1, 3), 1.0, pad=3)
    zyx, rep = voxelize_surface(tri_zyx, grid)
    domain = domain_from_voxels(zyx, grid.shape)
    groups, links = [], []
    for zid, drho in ((2, dp_lat * 3.0), (3, 0.0)):                    # rho = 1 + 3 p (cs^2 = 1/3)
        cells, geo = patch_cells(domain, grid, tri_zyx[zone == zid])
        links.append(cap_links(domain, grid, tri_zyx[zone == zid]) if scheme == "abb" else None)
        groups.append((Patch(f"cap{zid}", "pressure", pressure=drho, normal=geo["normal"],
                             area=geo["area"]), cells))
    boundary = from_cells(domain, groups, links)
    sim = SparseLBM(domain, tau=tau, collision="trt", boundary=boundary)
    for k, (p, _) in enumerate(groups):
        sim.set_patch(k, rho=1.0 + p.pressure)
    prev = None
    for it in range(steps // 1000):
        sim.step(1000)
        q = sim.patch_flux()
        if prev is not None and abs(q[0] - prev) < 1e-7 * abs(q[0]):
            break
        prev = q[0]
    f = sim.fields()
    b = domain.brick ** 3
    br, loc = np.nonzero(domain.flags == 0)
    cells = br.astype(np.int64) * b + loc
    xyz = (grid.origin + (cell_coords(domain, cells) + 0.5) * grid.voxel)[:, ::-1] - 1000.0
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    s = xyz @ a
    rho = f["rho"].get()[br, loc]
    p = (rho - 1.0) / 3.0
    mid = np.abs(s) < length / 4
    # the pressure is uniform across the pipe in Poiseuille flow: fit p(s) over the middle,
    # and extrapolate to where the axis meets each cap
    coef = np.polyfit(s[mid], p[mid], 1)
    p_in, p_out = np.polyval(coef, -length / 2), np.polyval(coef, length / 2)
    nu = (tau - 0.5) / 3.0
    q_hp = np.pi * radius ** 4 / (8 * nu) * (-coef[0])                 # at the measured gradient
    q = -sim.patch_flux()[1]
    res = {"scheme": scheme, "axis": list(map(float, a)), "caps": "z" if z_caps else "normal",
           "tau": tau,
           "radius": radius, "length": length,
           "fluid_voxels": int(domain.fluid_cells), "steps": int(sim.steps),
           "dp_imposed": dp_lat, "dp_felt": float(p_in - p_out),
           "jump_in_pct": float((dp_lat - p_in) / dp_lat * 100),
           "jump_out_pct": float((p_out - 0.0) / dp_lat * 100),
           "flux_vs_hp_at_felt_gradient_pct": float((q / q_hp - 1) * 100),
           "flux_vs_hp_at_imposed_dp_pct": float(
               (q / (np.pi * radius ** 4 / (8 * nu) * dp_lat / length) - 1) * 100)}
    log(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
    return res


def main():
    rows = []
    for scheme in ("neem", "abb"):
        for axis in ((0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 3)):
            for tau in (0.55, 0.8):
                rows.append(run_case(axis, tau=tau, scheme=scheme))
        for tau in (0.55, 0.8):                 # oblique pipe, caps perpendicular to z
            rows.append(run_case((1, 1, 1), tau=tau, z_caps=True, scheme=scheme))
    OUT.write_text(json.dumps(rows, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
