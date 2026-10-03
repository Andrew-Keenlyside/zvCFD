"""The NASA Turbulence Modeling Resource grids as one-cell-deep zvCFD meshes.

    TMR=/hdd/data/zvcfd_turb  (FlatPlate/Grids, NACA0012_grids from the Resource's zips)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from zvcfd.mesh.plot3d import extrude_2d, read_plot3d_2d

TMR = Path(os.environ.get("TMR", "/hdd/data/zvcfd_turb"))
FLAT = {35: "flatplate_clust2_4levelsdown_35x25.p2dfmt",
        69: "flatplate_clust2_3levelsdown_69x49.p2dfmt",
        137: "flatplate_clust2_2levelsdown_137x97.p2dfmt",
        273: "flatplate_clust2_1leveldown_273x193.p2dfmt.gz", 545: "flatplate_clust2.p2dfmt.gz"}
NACA = {113: "n0012_113-33.p2dfmt", 225: "n0012_225-65.p2dfmt", 449: "n0012_449-129.p2dfmt.gz",
        897: "n0012_897-257.p2dfmt.gz", 1793: "n0012_1793-513.p2dfmt.gz"}


def flat_plate(ni: int = 137, span: float = 0.05):
    """The zero-pressure-gradient plate: x in [-1/3, 2], y in [0, 1]; the plate from x = 0.

    Zones: inflow (x = -1/3), outflow (x = 2), top (y = 1), symmetry ahead of the plate
    (y = 0, x < 0), plate (y = 0, x >= 0), and the two z planes.
    """
    x, y = read_plot3d_2d(TMR / "FlatPlate" / "Grids" / FLAT[ni])
    tol = 1e-9

    def classify(c, n):
        w = np.full(len(c), -1)
        w[np.abs(c[:, 2]) < tol] = 0
        w[np.abs(c[:, 2] - span) < tol] = 1
        side = w < 0
        w[side & (c[:, 0] < x.min() + tol)] = 2
        w[side & (c[:, 0] > x.max() - tol)] = 3
        w[side & (c[:, 1] > y.max() - tol)] = 4
        bottom = side & (c[:, 1] < tol)
        w[bottom & (c[:, 0] < 0)] = 5
        w[bottom & (c[:, 0] >= 0)] = 6
        return {"which": w, "names": [("zmin", "symmetry"), ("zmax", "symmetry"),
                                      ("inflow", "velocity-inlet"),
                                      ("outflow", "pressure-outlet"), ("top", "pressure-outlet"),
                                      ("ahead", "symmetry"), ("plate", "wall")]}
    return extrude_2d(x, y, span, classify)


def naca0012(ni: int = 225, span: float = 0.05):
    """The NACA 0012 C-grid (chord 1, farfield ~500 chords): zones airfoil (wall),
    farfield (the C boundary), outflow (the two downstream ends), and the two z planes."""
    x, y = read_plot3d_2d(TMR / "NACA0012_grids" / NACA[ni])
    tol = 1e-9
    xmax = x.max()

    def classify(c, n):
        w = np.full(len(c), -1)
        w[np.abs(c[:, 2]) < tol] = 0
        w[np.abs(c[:, 2] - span) < tol] = 1
        side = w < 0
        r = np.hypot(c[:, 0] - 0.5, c[:, 1])
        w[side & (r < 2.0)] = 2                                     # the airfoil surface
        w[side & (r >= 2.0) & (c[:, 0] > xmax - 1e-6 * xmax)] = 3   # downstream ends
        w[(w < 0)] = 4                                              # the C boundary
        return {"which": w, "names": [("zmin", "symmetry"), ("zmax", "symmetry"),
                                      ("airfoil", "wall"), ("outflow", "pressure-outlet"),
                                      ("farfield", "velocity-inlet")]}
    return extrude_2d(x, y, span, classify)


# ---------------------------------------------------------------- ONERA M6 (low speed)

M6_ROOT, M6_SPAN, M6_TAPER, M6_SWEEP = 0.8059, 1.1963, 0.562, np.radians(30.0)
M6_MAC = 0.64607


def onera_d(x):
    """Half-thickness of the ONERA D section (sharp trailing edge, ``foilmod.txt``) at x/c."""
    t = np.loadtxt(TMR / "onera_m6" / "foilmod.txt")
    return np.interp(np.clip(x, 0, 1), t[:, 0], t[:, 1])


def _symmetric_section(ni):
    """The NACA 0012 C-grid made exactly symmetric about y = 0 (lower half mirrored)."""
    x, y = read_plot3d_2d(TMR / "NACA0012_grids" / NACA[ni])
    nj, n = x.shape
    mid = (n - 1) // 2
    lower = y[0, mid // 2] < 0                       # is the first half the lower surface?
    x, y = x.copy(), y.copy()
    src = np.arange(mid + 1)
    dst = n - 1 - src
    if lower:
        x[:, dst], y[:, dst] = x[:, src], -y[:, src]
    else:
        x[:, src], y[:, src] = x[:, dst], -y[:, dst]
    y[:, mid] = 0.0
    y[0, x[0] >= 1.0] = 0.0                           # the wake cut and trailing edge on y = 0
    return x, y


def _morph(x, y, half_thickness, delta=0.1):
    """Move the airfoil surface (j = 0, 0 <= x <= 1) to ``half_thickness(x)``, blending the
    displacement into the grid along each j-line with ``exp(-s/delta)``."""
    on = (x[0] >= 0) & (x[0] <= 1) & (np.abs(y[0]) > 0) | (x[0] == 0)
    target = np.where(on, np.sign(y[0]) * half_thickness(x[0]), y[0])
    dy = np.where(on, target - y[0], 0.0)
    s = np.hypot(x - x[0], y - y[0])                 # distance from the surface node
    return y + dy[None, :] * np.exp(-s / delta)


def onera_m6(ni: int = 113, n_wing: int = 32, n_out: int = 14, span_out: float = 4.0,
             r_far: float | None = None, tip_cluster: bool = True, growth: float = 1.35):
    """The ONERA M6 semispan in a C-H grid of hexahedra.

    Sections of the NACA 0012 C-grid, made symmetric and morphed to the ONERA D
    section, are stacked along the span (z) with the M6 planform: root chord
    0.8059 m, semispan 1.1963 m, taper 0.562, leading edge swept 30 deg, no twist.
    Beyond the tip the section thins to a sheet in one station (the tip) and the
    sheet's two sides merge, as a C-grid's wake cut does. Zones: wing (wall),
    farfield (the C boundary), outflow (downstream ends), root (z = 0, symmetry),
    side (z = span_out x semispan, symmetry).
    """
    from zvcfd.mesh.core import BoundaryZone, UnstructuredMesh
    from zvcfd.mesh.generate import _orient

    x2, y2 = _symmetric_section(ni)
    if r_far is not None:
        # crop the section's C-grid to about r_far chords: j-lines out to that radius at the
        # leading edge, the wake cut out to that distance downstream
        nj0, n0 = x2.shape
        mid = (n0 - 1) // 2
        rj = np.hypot(x2[:, mid] - 0.5, y2[:, mid])
        jmax = int(np.searchsorted(rj, r_far)) + 1
        i0 = int(np.argmax(x2[0, :mid] <= r_far))
        x2, y2 = x2[:jmax, i0:n0 - i0], y2[:jmax, i0:n0 - i0]
    nj, n = x2.shape
    b = M6_SPAN
    u = np.linspace(0, 1, n_wing + 1)
    s = np.sin(0.5 * np.pi * u) if tip_cluster else u            # clustered toward the tip
    zw = b * s
    dz = zw[-1] - zw[-2]
    zo = b + dz * np.cumsum(growth ** np.arange(n_out))
    zo = zo[zo < span_out * b]
    z = np.concatenate([zw, zo, [span_out * b]])
    planes = []
    for zk in z:
        if zk <= b + 1e-12:
            c = M6_ROOT * (1 - (1 - M6_TAPER) * zk / b)
            xle, t = zk * np.tan(M6_SWEEP), 1.0
        else:
            c, xle, t = M6_ROOT * M6_TAPER, b * np.tan(M6_SWEEP), 0.0
        yk = _morph(x2, y2, lambda xx, t=t: t * onera_d(xx))
        planes.append(np.stack([xle + c * x2, c * yk, np.full_like(x2, zk)], -1))
    P = np.stack(planes)                                            # (K, nj, n, 3)
    K = len(z)
    pts = P.reshape(-1, 3)
    kk, jj, ii = np.meshgrid(np.arange(K), np.arange(nj), np.arange(n), indexing="ij")
    from zvcfd.mesh.plot3d import _merge
    rep = _merge(pts, 1e-12 * np.ptp(pts, axis=0).max())
    keep, local = np.unique(rep, return_inverse=True)
    nodes = pts[keep]
    idx = local.reshape(K, nj, n)
    q = [idx[:-1, :-1, :-1], idx[:-1, :-1, 1:], idx[:-1, 1:, 1:], idx[:-1, 1:, :-1]]
    q2 = [a.copy() for a in
          [idx[1:, :-1, :-1], idx[1:, :-1, 1:], idx[1:, 1:, 1:], idx[1:, 1:, :-1]]]
    hexes = np.stack([a.ravel() for a in q + q2], 1)
    ok = np.array([len(set(h)) == 8 for h in hexes])
    hexes = _orient(nodes, "hex", hexes[ok])
    mesh = UnstructuredMesh(nodes, {"hex": hexes}, unit="m")
    # zones from the grid indices of each boundary face's nodes
    jn = np.zeros(len(nodes), int)
    kn = np.zeros(len(nodes), int)
    inn = np.zeros(len(nodes), int)
    jn[local], kn[local], inn[local] = jj.ravel(), kk.ravel(), ii.ravel()
    fx = mesh.faces()
    bd = fx["c1"] < 0
    f, own = fx["faces"][bd], fx["c0"][bd]
    fn = np.where(f < 0, f[:, :1], f)

    def all_(arr, v):
        return (arr[fn] == v).all(1)
    which = np.full(len(f), -1)
    which[all_(kn, 0)] = 0
    which[all_(kn, K - 1)] = 1
    which[(which < 0) & all_(jn, nj - 1)] = 2
    ends = np.isin(inn, [0, n - 1])                   # merged wake nodes carry either end's i
    which[(which < 0) & ends[fn].all(1)] = 3
    which[(which < 0) & all_(jn, 0)] = 4
    if (which < 0).any():
        raise ValueError(f"{int((which < 0).sum())} unclassified boundary faces")
    names = [("root", "symmetry"), ("side", "symmetry"), ("farfield", "velocity-inlet"),
             ("outflow", "pressure-outlet"), ("wing", "wall")]
    mesh.zones = {zid: BoundaryZone(zid, kind, name, f[which == zid - 3], own[which == zid - 3])
                  for zid, (name, kind) in enumerate(names, start=3)}
    return mesh
