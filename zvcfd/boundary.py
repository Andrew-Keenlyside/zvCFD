"""Inlet and outlet patches: which voxels, what condition, and the lumped outlet models.

A patch is a set of boundary voxels sharing one condition. Its voxels are
flagged 3 (pressure, including RCR outlets) or 4 (velocity) in the domain,
and each gets an *interior neighbour*: the fluid voxel one lattice link
inward that the boundary scheme extrapolates from (``boundary_neem`` in
:mod:`zvcfd.lbm._cuda`).

Patches are specified in physical SI units. The run converts them to
lattice units with :class:`zvcfd.units.Lattice`. Normals point **out** of
the fluid, axis order (z, y, x).

Sources of patches:

- :func:`face_patches` — fluid voxels on a face of the bounding box;
- :func:`zvcfd.geometry.voxelize_mesh` — the named inlet/outlet zones of a
  surface mesh (e.g. an Ansys Fluent ``.msh``);
- :func:`from_cells` — any explicit voxel set (e.g. a label image).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from zvcfd.domain import FLUID, SOLID, BrickDomain

PRESSURE_FLAG, VELOCITY_FLAG = 3, 4

# D3Q19 velocities as (z, y, x), excluding rest
_DIRS = np.array([(cz, cy, cx) for cx, cy, cz in zip(
    [1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0],
    [0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1],
    [0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1])], np.int64)


@dataclass
class Patch:
    """One inlet or outlet, in SI units.

    Attributes:
        name: Patch name (e.g. the Fluent zone name).
        kind: ``"pressure"``, ``"velocity"`` or ``"rcr"`` (a pressure patch
            whose pressure follows a three-element Windkessel).
        pressure: Pa. Fixed pressure, or the RCR distal (venous) pressure.
        velocity: m/s. Mean normal velocity *into* the domain (velocity kind).
        flow_rate: m³/s into the domain; overrides ``velocity`` when the area is known.
        normal: outward unit normal (z, y, x).
        area: m². Physical patch area (from the mesh); estimated from voxels if None.
        profile: ``"plug"`` or ``"parabolic"`` (circular patch, peak 2× mean).
        rcr: ``(Rp, C, Rd)`` in Pa·s/m³, m³/Pa, Pa·s/m³.
        waveform: ``[(t, factor), ...]`` periodic multiplier on velocity/flow rate
            or pressure; linear interpolation, period = last t.
    """

    name: str
    kind: str
    pressure: float = 0.0
    velocity: float = 0.0
    flow_rate: float | None = None
    normal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    area: float | None = None
    profile: str = "plug"
    rcr: tuple[float, float, float] | None = None
    waveform: list[tuple[float, float]] | None = None
    centroid: tuple[float, float, float] | None = None   # voxel units (z, y, x)
    # RCR state: distal pressure (Pa)
    _pd: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if self.kind not in ("pressure", "velocity", "rcr"):
            raise ValueError(f"patch kind {self.kind!r}")
        if self.kind == "rcr" and self.rcr is None:
            raise ValueError(f"patch {self.name}: rcr needs (Rp, C, Rd)")
        self._pd = self.pressure

    @property
    def flag(self) -> int:
        return VELOCITY_FLAG if self.kind == "velocity" else PRESSURE_FLAG

    def factor(self, t: float) -> float:
        if not self.waveform:
            return 1.0
        ts, vs = np.asarray(self.waveform, float).T
        return float(np.interp(t % ts[-1], ts, vs))

    def mean_velocity(self, t: float = 0.0) -> float:
        """Mean inflow velocity (m/s) at time ``t``."""
        if self.flow_rate is not None:
            if not self.area:
                raise ValueError(f"patch {self.name}: flow_rate needs an area")
            return self.flow_rate / self.area * self.factor(t)
        return self.velocity * self.factor(t)

    def rcr_pressure(self, q_out: float, dt: float) -> float:
        """Advance the Windkessel by ``dt`` with outflow ``q_out`` (m³/s); return P (Pa).

        C dPd/dt = Q - (Pd - P_venous)/Rd,  P = Pd + Rp Q  (explicit Euler).
        """
        rp, c, rd = self.rcr
        self._pd += dt * (q_out - (self._pd - self.pressure) / rd) / c
        return self._pd + rp * q_out


@dataclass
class BoundarySet:
    """Patch voxels of a domain, in the domain's global cell numbering.

    A global cell id is ``brick * B³ + local`` with ``local`` the voxel's
    index in its brick (x fastest).
    """

    patches: list[Patch]
    cells: np.ndarray          # (N,) int64
    nbr: np.ndarray            # (N,) int64, interior neighbour or -1
    patch: np.ndarray          # (N,) int32 patch index
    scale: np.ndarray          # (N,) float32 profile factor (velocity patches)

    def __len__(self) -> int:
        return len(self.cells)

    def counts(self) -> np.ndarray:
        return np.bincount(self.patch, minlength=len(self.patches))

    def apply_flags(self, domain: BrickDomain) -> None:
        """Set flags 3/4 on the patch voxels (in place)."""
        b = domain.brick ** 3
        fl = np.array([p.flag for p in self.patches], np.uint8)[self.patch]
        domain.flags[self.cells // b, self.cells % b] = fl

    def localize(self, local: BrickDomain) -> BoundarySet:
        """The patch voxels owned by a partition, renumbered to its local cells.

        Neighbours may fall in the partition's ghost bricks; they are kept.
        """
        b = local.brick ** 3
        order = np.argsort(local.global_ids)
        sg = local.global_ids[order]

        def to_local(cells):
            gb = cells // b
            pos = np.clip(np.searchsorted(sg, gb), 0, max(len(sg) - 1, 0))
            found = (cells >= 0) & (sg[pos] == gb)
            lid = np.where(found, order[pos], -1)
            return np.where(found, lid * b + cells % b, -1), lid

        lc, lid = to_local(self.cells)
        keep = (lid >= 0) & (lid < local.n_owned)
        nl, _ = to_local(self.nbr)
        return BoundarySet(self.patches, lc[keep], nl[keep], self.patch[keep], self.scale[keep])


def cell_ids(domain: BrickDomain, zyx: np.ndarray) -> np.ndarray:
    """Global cell ids of voxel coordinates ``(N, 3)``; -1 outside active bricks or the box."""
    zyx = np.asarray(zyx, np.int64).reshape(-1, 3)
    b = domain.brick
    inside = ((zyx >= 0) & (zyx < np.asarray(domain.shape))).all(1)
    out = -np.ones(len(zyx), np.int64)
    z = zyx[inside]
    br = domain.brick_index(z // b)
    loc = (z[:, 2] % b) + b * ((z[:, 1] % b) + b * (z[:, 0] % b))
    out[inside] = np.where(br >= 0, br * b ** 3 + loc, -1)
    return out


def cell_coords(domain: BrickDomain, cells: np.ndarray) -> np.ndarray:
    """Voxel coordinates (z, y, x) of global cell ids."""
    b = domain.brick
    cells = np.asarray(cells, np.int64)
    br, loc = cells // b ** 3, cells % b ** 3
    lz, ly, lx = loc // (b * b), (loc // b) % b, loc % b
    return domain.coords[br].astype(np.int64) * b + np.stack([lz, ly, lx], 1)


def cell_flags(domain: BrickDomain, cells: np.ndarray) -> np.ndarray:
    b = domain.brick ** 3
    cells = np.asarray(cells, np.int64)
    out = np.full(len(cells), SOLID, np.uint8)
    ok = cells >= 0
    out[ok] = domain.flags[cells[ok] // b, cells[ok] % b]
    return out


def inward_neighbours(domain: BrickDomain, cells: np.ndarray, normal_out) -> np.ndarray:
    """For each cell, the fluid neighbour along the lattice link closest to the inward normal."""
    zyx = cell_coords(domain, cells)
    n_in = -np.asarray(normal_out, float)
    n_in = n_in / max(np.linalg.norm(n_in), 1e-12)
    score = (_DIRS @ n_in) / np.linalg.norm(_DIRS, axis=1)
    order = np.argsort(-score)
    best = -np.ones(len(cells), np.int64)
    for k in order[:6]:
        if score[k] <= 0:
            break
        cand = cell_ids(domain, zyx + _DIRS[k])
        ok = (best < 0) & (cand >= 0) & (cell_flags(domain, cand) == FLUID)
        best[ok] = cand[ok]
    return best


def profile_scale(domain: BrickDomain, cells: np.ndarray, patch: Patch) -> np.ndarray:
    if patch.profile == "plug" or patch.kind != "velocity":
        return np.ones(len(cells), np.float32)
    zyx = cell_coords(domain, cells).astype(float)
    c = np.asarray(patch.centroid) if patch.centroid is not None else zyx.mean(0)
    n = np.asarray(patch.normal, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    d = zyx - c
    r2 = (d * d).sum(1) - (d @ n) ** 2
    s = 2.0 * np.clip(1.0 - r2 / (r2.max() + 1.0), 0.0, None)
    return (s / max(s.mean(), 1e-12)).astype(np.float32)     # mean 1: flow rate preserved


def from_cells(domain: BrickDomain, groups: list[tuple[Patch, np.ndarray]]) -> BoundarySet:
    """Build a boundary set from explicit ``(patch, global cell ids)`` groups."""
    cells, pid, scale = [], [], []
    for k, (patch, c) in enumerate(groups):
        c = np.unique(np.asarray(c, np.int64))
        c = c[(c >= 0) & (cell_flags(domain, c) == FLUID)]
        cells.append(c)
        pid.append(np.full(len(c), k, np.int32))
        scale.append(profile_scale(domain, c, patch))
    allc = np.concatenate(cells) if cells else np.zeros(0, np.int64)
    # neighbours must not themselves be patch cells: mark first, then search
    tmp = BoundarySet([g[0] for g in groups], allc, -np.ones(len(allc), np.int64),
                      np.concatenate(pid) if pid else np.zeros(0, np.int32),
                      np.concatenate(scale) if scale else np.zeros(0, np.float32))
    tmp.apply_flags(domain)
    for k, (patch, _) in enumerate(groups):
        sel = tmp.patch == k
        tmp.nbr[sel] = inward_neighbours(domain, tmp.cells[sel], patch.normal)
    return tmp


FACES = {"zmin": (0, 0, -1), "zmax": (0, -1, 1), "ymin": (1, 0, -1), "ymax": (1, -1, 1),
         "xmin": (2, 0, -1), "xmax": (2, -1, 1)}


def face_patches(domain: BrickDomain, faces: dict[str, Patch]) -> BoundarySet:
    """Patches on faces of the bounding box: ``{"xmin": Patch(...), "xmax": Patch(...)}``.

    The fluid voxels of the face layer become patch voxels; normals and areas
    are filled in from the face (area in voxel faces × ``voxel_area`` is left
    to the caller via ``Patch.area``; None means "count voxels").
    """
    b = domain.brick
    groups = []
    for face, patch in faces.items():
        ax, idx, sign = FACES[face]
        n = [0.0, 0.0, 0.0]
        n[ax] = float(sign)
        patch.normal = tuple(n)
        layer = domain.shape[ax] - 1 if idx == -1 else 0
        on = (domain.coords[:, ax].astype(np.int64) * b <= layer) & \
             (layer < (domain.coords[:, ax].astype(np.int64) + 1) * b)
        bricks = np.flatnonzero(on)
        loc = np.arange(b ** 3)
        lz, ly, lx = loc // (b * b), (loc // b) % b, loc % b
        lcoord = np.stack([lz, ly, lx], 1)[:, ax]
        sel = lcoord == layer % b
        cells = (bricks[:, None] * b ** 3 + loc[sel][None, :]).reshape(-1)
        groups.append((patch, cells))
    return from_cells(domain, groups)
