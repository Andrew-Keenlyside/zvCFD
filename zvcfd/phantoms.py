"""Synthetic geometries for tests and benchmarks.

Flag volumes use 0 fluid / 1 solid, axis order (z, y, x).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def porous_spheres(n: int, porosity_target=0.45, radius=6, seed=0) -> np.ndarray:
    """Periodic overlapping-sphere solid; flag 1 = solid."""
    rng = np.random.default_rng(seed)
    solid = np.zeros((n, n, n), bool)
    zz, yy, xx = np.ogrid[:n, :n, :n]
    while 1 - solid.mean() > porosity_target:
        c = rng.integers(0, n, 3)
        dz = np.minimum(abs(zz - c[0]), n - abs(zz - c[0]))
        dy = np.minimum(abs(yy - c[1]), n - abs(yy - c[1]))
        dx = np.minimum(abs(xx - c[2]), n - abs(xx - c[2]))
        solid |= dz * dz + dy * dy + dx * dx <= radius * radius
    return solid.astype(np.uint8)


def vessel_tree(shape, n_vessels=40, r_min=3.0, r_max=8.0, seed=0) -> np.ndarray:
    """Random straight tubes crossing a box (a crude vasculature stand-in).

    Returns flag: 1 solid, 0 fluid.  Vessels are capsules between random
    points on opposite faces, so the network percolates in x.
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    fluid = np.zeros(shape, bool)
    for v in range(n_vessels):
        r = rng.uniform(r_min, r_max)
        axis = v % 3
        a = rng.uniform(0, 1, 3) * np.array([nz, ny, nx])
        b = rng.uniform(0, 1, 3) * np.array([nz, ny, nx])
        a[2 - axis] = 0
        b[2 - axis] = (nz, ny, nx)[2 - axis] - 1
        lo = np.maximum(np.floor(np.minimum(a, b) - r - 1), 0).astype(int)
        hi = np.minimum(np.ceil(np.maximum(a, b) + r + 2), shape).astype(int)
        z, y, x = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        p = np.stack([z, y, x], -1).astype(np.float32)
        d = b - a
        t = np.clip(((p - a) @ d) / (d @ d), 0, 1)
        dist = np.linalg.norm(p - (a + t[..., None] * d), axis=-1)
        fluid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= dist <= r
    return (~fluid).astype(np.uint8)


def vessel_network(shape, n=26, r=(5.0, 9.0), seed=3) -> np.ndarray:
    """Capsule vessels; half run along x so the network percolates in x."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    ext = np.array(shape, float)
    fluid = np.zeros(shape, bool)
    for v in range(n):
        rad = rng.uniform(*r)
        axis = 2 if v % 2 == 0 else (0 if v % 4 == 1 else 1)   # 2 = x in (z, y, x)
        a = rng.uniform(0.15, 0.85, 3) * ext
        b = rng.uniform(0.15, 0.85, 3) * ext
        a[axis], b[axis] = 0, ext[axis] - 1
        lo = np.maximum(np.floor(np.minimum(a, b) - rad - 1), 0).astype(int)
        hi = np.minimum(np.ceil(np.maximum(a, b) + rad + 2), shape).astype(int)
        z, y, x = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        p = np.stack([z, y, x], -1).astype(np.float32)
        d = b - a
        t = np.clip(((p - a) @ d) / (d @ d), 0, 1)
        fluid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= (
            np.linalg.norm(p - (a + t[..., None] * d), axis=-1) <= rad)
    # keep the component(s) touching both x faces, drop dead islands
    lab, _ = ndimage.label(fluid)
    keep = np.intersect1d(np.unique(lab[:, :, 0]), np.unique(lab[:, :, -1]))
    keep = keep[keep > 0]
    return np.isin(lab, keep)


def coarsen_mask(fluid: np.ndarray, rule: str = "majority") -> np.ndarray:
    """2x downsample of a fluid mask: ``majority`` (>= 4 of 8) or ``any`` (max-pool)."""
    nz, ny, nx = fluid.shape
    blocks = fluid.reshape(nz // 2, 2, ny // 2, 2, nx // 2, 2).sum(axis=(1, 3, 5))
    return blocks >= (1 if rule == "any" else 4)


def reservoir_flags(fluid: np.ndarray) -> np.ndarray:
    """Flags with fixed-density reservoirs on the fluid voxels of both x faces."""
    fl = np.where(fluid, 0, 1).astype(np.uint8)
    fl[:, :, 0][fluid[:, :, 0]] = 2
    fl[:, :, -1][fluid[:, :, -1]] = 2
    return fl


PHANTOMS = {
    "porous-256": lambda: porous_spheres(256),
    "vessels-512": lambda: vessel_tree((512, 512, 512), n_vessels=60, r_min=3, r_max=9),
    "network-128x128x512": lambda: np.where(vessel_network((128, 128, 512)), 0, 1).astype(np.uint8),
}
"""Named phantoms (flag volumes) used by the benchmarks; ``zvcfd phantom list``."""
