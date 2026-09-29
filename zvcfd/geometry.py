"""Closed surface meshes to sparse brick domains, with inlet/outlet patches.

``voxelize_mesh`` turns a watertight triangulated surface (the boundary of
an Ansys Fluent mesh from :mod:`zvcfd.io.fluent_msh`, or any triangle soup
with zone ids) into a :class:`~zvcfd.domain.BrickDomain` and a
:class:`~zvcfd.boundary.BoundarySet` whose patches are the mesh's inlet
and outlet zones.

Method: parity ray casting along z. Every voxel column (y, x) casts one ray
through the voxel centres; triangles are binned by the columns their
footprint covers, each (triangle, column) pair gives at most one crossing,
and voxels between the 1st-2nd, 3rd-4th, ... crossings are fluid. No
dense volume is ever allocated: fluid voxels go straight to bricks. Ray
origins are offset by a tiny irrational fraction of a voxel so they never
hit an edge or vertex exactly. Columns with an odd number of crossings
(a leaky surface) are dropped and counted.

Patches: each non-wall zone's triangles are sampled densely; the fluid
voxel half a voxel inside each sample becomes a patch voxel, with the
zone's area, outward normal and centroid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zvcfd.boundary import Patch, cell_ids, from_cells
from zvcfd.domain import SOLID, BrickDomain, morton3

_EPS = (1.1e-5, 2.3e-5)          # column offsets, in voxels (y, x)


@dataclass
class VoxelGrid:
    """Where the voxels are: origin (z, y, x) of voxel (0, 0, 0)'s corner, edge, shape."""

    origin: np.ndarray
    voxel: float
    shape: tuple[int, int, int]

    @classmethod
    def around(cls, points_zyx: np.ndarray, voxel: float, *, pad: int = 2, brick: int = 8):
        lo = points_zyx.min(0) - pad * voxel
        hi = points_zyx.max(0) + pad * voxel
        n = np.ceil((hi - lo) / voxel).astype(int)
        n = ((n + brick - 1) // brick) * brick
        return cls(lo, float(voxel), tuple(int(v) for v in n))

    def to_voxel(self, p_zyx: np.ndarray) -> np.ndarray:
        """Continuous voxel coordinates (voxel centres at integers)."""
        return (np.asarray(p_zyx) - self.origin) / self.voxel - 0.5


def _column_crossings(tri: np.ndarray, grid: VoxelGrid, chunk: int = 1_000_000):
    """(column id, z crossing in voxel units) for every triangle-column intersection."""
    ny, nx = grid.shape[1], grid.shape[2]
    v = grid.to_voxel(tri.reshape(-1, 3)).reshape(-1, 3, 3)      # (T, 3 vertices, zyx)
    cols, zs = [], []
    for s in range(0, len(v), chunk):
        t = v[s:s + chunk]
        ymin = np.ceil(t[:, :, 1].min(1) - _EPS[0]).astype(np.int64)
        ymax = np.floor(t[:, :, 1].max(1) - _EPS[0]).astype(np.int64)
        xmin = np.ceil(t[:, :, 2].min(1) - _EPS[1]).astype(np.int64)
        xmax = np.floor(t[:, :, 2].max(1) - _EPS[1]).astype(np.int64)
        ny_t = np.maximum(ymax - ymin + 1, 0)
        nx_t = np.maximum(xmax - xmin + 1, 0)
        cnt = ny_t * nx_t
        if not cnt.sum():
            continue
        tid = np.repeat(np.arange(len(t)), cnt)
        k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        cy = ymin[tid] + k // np.maximum(nx_t[tid], 1)
        cx = xmin[tid] + k % np.maximum(nx_t[tid], 1)
        py, px = cy + _EPS[0], cx + _EPS[1]
        a, b, c = t[tid, 0], t[tid, 1], t[tid, 2]
        # 2-D barycentric test in (y, x), then interpolate z
        d = (b[:, 1] - a[:, 1]) * (c[:, 2] - a[:, 2]) - (c[:, 1] - a[:, 1]) * (b[:, 2] - a[:, 2])
        ok = np.abs(d) > 1e-14
        w1 = ((py - a[:, 1]) * (c[:, 2] - a[:, 2]) - (c[:, 1] - a[:, 1]) * (px - a[:, 2]))
        w2 = ((b[:, 1] - a[:, 1]) * (px - a[:, 2]) - (py - a[:, 1]) * (b[:, 2] - a[:, 2]))
        with np.errstate(divide="ignore", invalid="ignore"):
            w1, w2 = w1 / d, w2 / d
        w0 = 1 - w1 - w2
        hit = ok & (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (cy >= 0) & (cy < ny) & (cx >= 0) & (cx < nx)
        z = w0 * a[:, 0] + w1 * b[:, 0] + w2 * c[:, 0]
        cols.append((cy * nx + cx)[hit])
        zs.append(z[hit])
    if not cols:
        return np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(cols), np.concatenate(zs)


def voxelize_surface(tri: np.ndarray, grid: VoxelGrid) -> tuple[np.ndarray, dict]:
    """Fluid voxel coordinates ``(N, 3)`` (z, y, x) inside a closed surface ``(T, 3, 3)`` (zyx)."""
    col, z = _column_crossings(tri, grid)
    order = np.lexsort((z, col))
    col, z = col[order], z[order]
    uc, start, count = np.unique(col, return_index=True, return_counts=True)
    odd = count % 2 == 1
    keep = np.repeat(~odd, count)
    col, z = col[keep], z[keep]
    # pair consecutive crossings within each column: (z_in, z_out)
    zin, zout, cin = z[0::2], z[1::2], col[0::2]
    k0 = np.ceil(zin).astype(np.int64)
    k1 = np.floor(zout).astype(np.int64)
    k0 = np.clip(k0, 0, grid.shape[0] - 1)
    k1 = np.clip(k1, -1, grid.shape[0] - 1)
    n = np.maximum(k1 - k0 + 1, 0)
    rep = np.repeat(np.arange(len(n)), n)
    zi = k0[rep] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    ci = cin[rep]
    zyx = np.stack([zi, ci // grid.shape[2], ci % grid.shape[2]], 1)
    return zyx, {"columns": int(len(uc)), "odd_columns_dropped": int(odd.sum()),
                 "crossings": int(len(z))}


def domain_from_voxels(zyx: np.ndarray, shape, *, brick: int = 8) -> BrickDomain:
    """A :class:`BrickDomain` whose fluid voxels are exactly ``zyx``."""
    b = brick
    bc = zyx // b
    key = morton3(bc)
    uk, inv = np.unique(key, return_inverse=True)
    first = np.zeros(len(uk), np.int64)
    first[inv] = np.arange(len(zyx))
    coords = bc[first]
    flags = np.full((len(uk), b ** 3), SOLID, np.uint8)
    loc = (zyx[:, 2] % b) + b * ((zyx[:, 1] % b) + b * (zyx[:, 0] % b))
    flags[inv.reshape(-1), loc] = 0
    return BrickDomain.from_bricks(shape, coords, flags, brick=b)


def _sample_triangles(tri: np.ndarray, spacing: float) -> np.ndarray:
    """Points covering each triangle at roughly ``spacing`` (same units as ``tri``)."""
    e = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2).max(1)
    m = np.clip(np.ceil(e / spacing).astype(int), 1, 64)
    pts = []
    for mm in np.unique(m):
        t = tri[m == mm]
        i, j = np.meshgrid(np.arange(mm + 1), np.arange(mm + 1), indexing="ij")
        keep = i + j <= mm
        u, v = (i[keep] / mm), (j[keep] / mm)
        w = 1 - u - v
        p = (w[None, :, None] * t[:, None, 0] + u[None, :, None] * t[:, None, 1]
             + v[None, :, None] * t[:, None, 2])
        pts.append(p.reshape(-1, 3))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def patch_cells(domain: BrickDomain, grid: VoxelGrid, tri: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fluid voxels just inside a cap (inlet/outlet) and the cap's geometry (voxel units)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    cross = np.cross(b - a, c - a)
    area_vec = 0.5 * cross.sum(0)
    area = float(np.linalg.norm(0.5 * cross, axis=1).sum())
    n = area_vec / max(np.linalg.norm(area_vec), 1e-30)
    cent = (((a + b + c) / 3) * np.linalg.norm(cross, axis=1)[:, None]).sum(0) / \
        max(np.linalg.norm(cross, axis=1).sum(), 1e-30)
    pts = grid.to_voxel(_sample_triangles(tri, grid.voxel / 3))
    # which side is the fluid? the side where more half-voxel-offset samples land in fluid
    best = None
    for sgn in (1.0, -1.0):
        ids = cell_ids(domain, np.round(pts + sgn * 0.5 * n).astype(np.int64))
        ids = ids[ids >= 0]
        fl = domain.flags[ids // domain.brick ** 3, ids % domain.brick ** 3] if len(ids) else []
        cand = np.unique(ids[np.asarray(fl) == 0]) if len(ids) else ids
        if best is None or len(cand) > len(best[0]):
            best = (cand, sgn)
    cells, sgn = best
    normal_out = -sgn * n                 # inward is sgn * n
    return cells, {"area": area, "normal": tuple(normal_out),
                   "centroid": tuple(grid.to_voxel(cent))}


def voxelize_mesh(tri_xyz: np.ndarray, zone_of_tri: np.ndarray, zone_kinds: dict[int, str],
                  voxel: float, *, zone_names: dict[int, str] | None = None,
                  unit_scale: float = 1.0, pad: int = 2):
    """Voxelise a zoned closed surface: returns ``(domain, boundary, grid, report)``.

    Args:
        tri_xyz: ``(T, 3, 3)`` triangles, coordinates (x, y, z) in mesh units.
        zone_of_tri: zone id per triangle.
        zone_kinds: zone id -> Fluent kind (``wall``, ``velocity-inlet``,
            ``pressure-outlet``, ...). Non-wall zones become patches: inlets
            as ``velocity`` patches, outlets as ``pressure`` patches (values
            are set later, per run).
        voxel: voxel edge in mesh units.
        unit_scale: metres per mesh unit (1e-3 for millimetres); patch areas
            are returned in m².
    """
    tri = np.asarray(tri_xyz, float)[..., ::-1]                   # -> (z, y, x)
    grid = VoxelGrid.around(tri.reshape(-1, 3), voxel, pad=pad)
    zyx, rep = voxelize_surface(tri, grid)
    domain = domain_from_voxels(zyx, grid.shape)
    groups = []
    zone_of_tri = np.asarray(zone_of_tri)
    for z, kind in sorted(zone_kinds.items()):
        if kind == "wall" or kind == "interior":
            continue
        sel = zone_of_tri == z
        if not sel.any():
            continue
        cells, geo = patch_cells(domain, grid, tri[sel])
        pk = "velocity" if "inlet" in kind else "pressure"
        name = (zone_names or {}).get(z, f"zone-{z}")
        groups.append((Patch(name, pk, normal=geo["normal"],
                             area=geo["area"] * unit_scale ** 2, centroid=geo["centroid"]),
                       cells))
    boundary = from_cells(domain, groups)
    rep.update({"shape": grid.shape, "voxel": voxel, "fluid_voxels": int(len(zyx)),
                "bricks": domain.n_bricks, "fill": domain.fill,
                "patches": len(groups), "patch_cells": boundary.counts().tolist(),
                "patches_without_cells": [g[0].name for g, n in
                                          zip(groups, boundary.counts()) if n == 0]})
    return domain, boundary, grid, rep


def voxelize_fluent(path: str, voxel: float, *, unit_scale: float = 1e-3, pad: int = 2):
    """Voxelise an ASCII Fluent mesh's boundary; see :func:`voxelize_mesh`."""
    from zvcfd.io.fluent_msh import read_fluent_boundary

    mesh = read_fluent_boundary(path)
    tris, zone = mesh.triangles()
    kinds = {z: fz.kind for z, fz in mesh.zones.items()}
    names = {z: fz.name for z, fz in mesh.zones.items()}
    return voxelize_mesh(mesh.nodes[tris], zone, kinds, voxel, zone_names=names,
                         unit_scale=unit_scale, pad=pad)


__all__ = ["VoxelGrid", "domain_from_voxels", "patch_cells", "voxelize_fluent", "voxelize_mesh",
           "voxelize_surface"]
