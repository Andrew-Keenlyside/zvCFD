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
from zvcfd.lumped import RCR, Coronary, LumpedOutlet

PRESSURE_FLAG, VELOCITY_FLAG = 3, 4
KINDS = ("pressure", "velocity", "rcr", "coronary")

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
        kind: ``"pressure"``, ``"velocity"``, ``"rcr"`` (a pressure patch
            whose pressure follows a three-element Windkessel) or
            ``"coronary"`` (an open-loop coronary outlet, Kim et al. 2010).
            Both lumped kinds come from :mod:`zvcfd.lumped`.
        pressure: Pa. Fixed pressure, or the venous pressure of a lumped outlet.
        velocity: m/s. Mean normal velocity *into* the domain (velocity kind).
        flow_rate: m³/s into the domain; overrides ``velocity`` when the area is known.
        normal: outward unit normal (z, y, x).
        area: m². Physical patch area (from the mesh); estimated from voxels if None.
        profile: ``"plug"`` or ``"parabolic"`` (circular patch, peak 2× mean).
        rcr: ``(Rp, C, Rd)`` in Pa·s/m³, m³/Pa, Pa·s/m³.
        coronary: ``(Ra, Ca, Ram, Cim, Rv)`` in Pa·s/m³ and m³/Pa.
        pim: intramyocardial pressure for ``coronary``: Pa, or ``[(t, Pa), ...]``
            (periodic, period = last t).
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
    coronary: tuple[float, float, float, float, float] | None = None
    pim: float | list[tuple[float, float]] | None = None
    # lumped outlet model (rcr, coronary), built on first use
    _model: LumpedOutlet | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"patch kind {self.kind!r}")
        if self.kind == "rcr" and self.rcr is None:
            raise ValueError(f"patch {self.name}: rcr needs (Rp, C, Rd)")
        if self.kind == "coronary" and self.coronary is None:
            raise ValueError(f"patch {self.name}: coronary needs (Ra, Ca, Ram, Cim, Rv)")

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

    @property
    def lumped(self) -> bool:
        return self.kind in ("rcr", "coronary")

    def outlet_model(self) -> LumpedOutlet:
        """The patch's 0-D outlet model (:mod:`zvcfd.lumped`), at rest at first use."""
        if not self.lumped:
            raise ValueError(f"patch {self.name}: kind {self.kind!r} has no outlet model")
        if self._model is None:
            if self.kind == "rcr":
                self._model = RCR.from_tuple(self.rcr, pv=self.pressure)
            else:
                ra, ca, ram, cim, rv = self.coronary
                self._model = Coronary(ra=ra, ca=ca, ram=ram, cim=cim, rv=rv,
                                       pim=self.pim or 0.0, pv=self.pressure)
        return self._model

    def outlet_pressure(self, q_out: float, t: float, dt: float) -> float:
        """Commit the outlet model's step ending at ``t`` with outflow ``q_out`` (m³/s);
        return the patch pressure (Pa) to hold until the next step."""
        return self.outlet_model().advance(q_out, t, dt)

    def rcr_pressure(self, q_out: float, dt: float) -> float:
        """Advance the outlet model by ``dt`` with outflow ``q_out`` (m³/s); return P (Pa)."""
        m = self.outlet_model()
        return m.advance(q_out, m.t + dt, dt)


_NO_LINKS = (np.zeros(0, np.int64), np.zeros(0, np.uint32), np.zeros(0, np.int32))


@dataclass
class BoundarySet:
    """Patch voxels and patch links of a domain, in the domain's global cell numbering.

    A global cell id is ``brick * B³ + local`` with ``local`` the voxel's
    index in its brick (x fastest).

    Two schemes carry a patch's condition:

    - **patch cells** (``cells``): a layer of voxels flagged 3 or 4, written
      each step by non-equilibrium extrapolation from an interior
      neighbour. Velocity patches, and pressure patches on box faces.
    - **patch links** (``link_cells``, ``link_mask``): lattice links from a
      fluid cell across a surface cap to a non-fluid node, each carrying an
      anti-bounce-back pressure condition. Pressure, RCR and coronary patches
      on surface caps (:func:`zvcfd.geometry.cap_links`). Bit ``q`` of a
      cell's mask is set when the population arriving along ``c_q`` crosses
      the cap. The cells stay fluid.

    A pressure-flagged patch that has links uses them, and its patch cells
    (kept, in case the patch becomes a velocity patch) are not flagged.
    """

    patches: list[Patch]
    cells: np.ndarray          # (N,) int64
    nbr: np.ndarray            # (N,) int64, interior neighbour or -1
    patch: np.ndarray          # (N,) int32 patch index
    scale: np.ndarray          # (N,) float32 profile factor (velocity patches)
    link_cells: np.ndarray = field(default_factory=lambda: _NO_LINKS[0].copy())   # (M,) int64
    link_mask: np.ndarray = field(default_factory=lambda: _NO_LINKS[1].copy())    # (M,) uint32
    link_patch: np.ndarray = field(default_factory=lambda: _NO_LINKS[2].copy())   # (M,) int32

    def __len__(self) -> int:
        return len(self.cells) + len(self.link_cells)

    def uses_links(self) -> np.ndarray:
        """Per patch: True where the condition is carried by links (a pressure-flagged
        patch that has any)."""
        has = np.bincount(self.link_patch, minlength=len(self.patches)) > 0
        flag = np.array([p.flag for p in self.patches], np.uint8)
        return has & (flag == PRESSURE_FLAG)

    def cell_entries(self) -> np.ndarray:
        """Mask of the patch-cell entries in use (those of patches not on links)."""
        return ~self.uses_links()[self.patch] if len(self.patches) else np.zeros(0, bool)

    def link_entries(self) -> np.ndarray:
        """Mask of the link entries in use."""
        return self.uses_links()[self.link_patch] if len(self.patches) else np.zeros(0, bool)

    def counts(self) -> np.ndarray:
        """Cells carrying each patch's condition (patch cells, or link cells)."""
        n = len(self.patches)
        return (np.bincount(self.patch[self.cell_entries()], minlength=n)
                + np.bincount(self.link_patch[self.link_entries()], minlength=n))

    def apply_flags(self, domain: BrickDomain) -> None:
        """Set flags 3/4 on the patch cells in use, and fluid on the others (in place)."""
        b = domain.brick ** 3
        fl = np.array([p.flag for p in self.patches], np.uint8)[self.patch]
        fl = np.where(self.cell_entries(), fl, FLUID).astype(np.uint8)
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
        kc, klid = to_local(self.link_cells)
        kkeep = (klid >= 0) & (klid < local.n_owned)
        return BoundarySet(self.patches, lc[keep], nl[keep], self.patch[keep], self.scale[keep],
                           kc[kkeep], self.link_mask[kkeep], self.link_patch[kkeep])


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


def from_cells(domain: BrickDomain, groups: list[tuple[Patch, np.ndarray]],
               links: list | None = None) -> BoundarySet:
    """Build a boundary set from explicit ``(patch, global cell ids)`` groups.

    ``links``, if given, holds per group ``None`` or ``(cells, mask)``: the
    group's patch links (:func:`zvcfd.geometry.cap_links`), used when the
    patch is a pressure-flagged one.
    """
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
    if links is not None:
        lc, lm, lp = [], [], []
        for k, lk in enumerate(links):
            if lk is None or not len(lk[0]):
                continue
            lc.append(np.asarray(lk[0], np.int64))
            lm.append(np.asarray(lk[1], np.uint32))
            lp.append(np.full(len(lk[0]), k, np.int32))
        if lc:
            tmp.link_cells, tmp.link_mask = np.concatenate(lc), np.concatenate(lm)
            tmp.link_patch = np.concatenate(lp)
    # interior neighbours never lie in any patch layer (as if every layer were
    # flagged), whichever scheme each patch ends up using
    b = domain.brick ** 3
    domain.flags[allc // b, allc % b] = np.array([p.flag for p in tmp.patches],
                                                  np.uint8)[tmp.patch]
    for k, (patch, _) in enumerate(groups):
        sel = tmp.patch == k
        tmp.nbr[sel] = inward_neighbours(domain, tmp.cells[sel], patch.normal)
    tmp.apply_flags(domain)
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
