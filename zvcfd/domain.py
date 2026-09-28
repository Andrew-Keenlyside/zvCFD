"""Sparse brick domains: the solver's view of a voxel geometry.

A domain is the set of ``B^3`` bricks (default 8^3) that hold at least
one non-solid voxel. Only those bricks are stored, on the GPU and on disk,
so memory scales with the fluid volume rather than the bounding box.

Bricks group into **chunks** of ``chunk_bricks^3`` bricks. A chunk is one
cell of the Zarr Vectors store, the unit a GPU owns, and the unit of halo
exchange. :meth:`BrickDomain.partition` assigns whole chunks to GPUs, so no
two workers ever write the same store cell (or, with
``shard_shape`` set on the store, the same shard). This is the
zarr-vectors concurrency contract: the shard is the unit of exclusion.

Voxel flags: 0 fluid, 1 solid, 2 fixed-density boundary (reservoir).
Axis order is (z, y, x) throughout, as in OME-Zarr.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FLUID, SOLID, RESERVOIR = 0, 1, 2

#: Neighbour slot k of a brick is offset (dz, dy, dx) with k = 9(dz+1) + 3(dy+1) + (dx+1).
NEIGHBOUR_OFFSETS = np.array(
    [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], np.int64
)


@dataclass
class BrickDomain:
    """Active bricks of a voxel geometry, with their neighbour table.

    Attributes:
        shape: Voxel shape of the bounding box, (z, y, x).
        brick: Brick edge in voxels.
        coords: ``(nb, 3)`` brick coordinates (z, y, x), in Morton order so
            bricks that are close in space are close in memory.
        flags: ``(nb, brick^3)`` uint8 voxel flags, x fastest within a brick.
        neighbours: ``(nb, 27)`` int32 brick index of each neighbour slot, -1
            where the neighbour brick is inactive (reads as solid).
        periodic: Per-axis periodicity used to build ``neighbours``.
    """

    shape: tuple[int, int, int]
    brick: int
    coords: np.ndarray
    flags: np.ndarray
    neighbours: np.ndarray
    periodic: tuple[bool, bool, bool] = (False, False, False)
    _lut: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------ building

    @classmethod
    def from_flags(cls, flags: np.ndarray, *, brick: int = 8,
                   periodic: bool | tuple[bool, bool, bool] = False) -> BrickDomain:
        """Build from a dense ``(z, y, x)`` flag volume (0 fluid, 1 solid, 2 reservoir)."""
        flags = np.asarray(flags, dtype=np.uint8)
        if flags.ndim != 3 or any(s % brick for s in flags.shape):
            raise ValueError(f"flag volume {flags.shape} must be 3-D and divisible by {brick}")
        per = (periodic,) * 3 if isinstance(periodic, bool) else tuple(periodic)
        nz, ny, nx = flags.shape
        bz, by, bx = nz // brick, ny // brick, nx // brick
        view = flags.reshape(bz, brick, by, brick, bx, brick).transpose(0, 2, 4, 1, 3, 5)
        active = (view != SOLID).any(axis=(3, 4, 5))
        coords = np.argwhere(active)
        coords = coords[np.argsort(morton3(coords), kind="stable")]
        local = view[coords[:, 0], coords[:, 1], coords[:, 2]].reshape(len(coords), -1)
        dom = cls(shape=(nz, ny, nx), brick=brick, coords=coords.astype(np.int32),
                  flags=np.ascontiguousarray(local), neighbours=np.empty((0, 27), np.int32),
                  periodic=per)
        dom.neighbours = dom._neighbour_table()
        return dom

    @classmethod
    def from_brick_mask(cls, active: np.ndarray, *, brick: int = 8,
                        periodic: bool | tuple[bool, bool, bool] = False) -> BrickDomain:
        """Build from a brick-level boolean mask; every voxel of an active brick is fluid.

        For synthetic I/O and scaling tests where voxel detail does not matter.
        """
        per = (periodic,) * 3 if isinstance(periodic, bool) else tuple(periodic)
        coords = np.argwhere(np.asarray(active, bool))
        coords = coords[np.argsort(morton3(coords), kind="stable")]
        shape = tuple(int(s) * brick for s in np.shape(active))
        dom = cls(shape=shape, brick=brick, coords=coords.astype(np.int32),
                  flags=np.zeros((len(coords), brick ** 3), np.uint8),
                  neighbours=np.empty((0, 27), np.int32), periodic=per)
        dom.neighbours = dom._neighbour_table()
        return dom

    @classmethod
    def from_bricks(cls, shape, coords: np.ndarray, flags: np.ndarray, *, brick: int = 8,
                    periodic: bool | tuple[bool, bool, bool] = False) -> BrickDomain:
        """Build from active brick coordinates ``(nb, 3)`` and their flags ``(nb, brick^3)``.

        The form a domain store holds; no dense volume is ever made. Bricks
        are re-sorted into Morton order.
        """
        per = (periodic,) * 3 if isinstance(periodic, bool) else tuple(periodic)
        coords = np.asarray(coords, np.int64).reshape(-1, 3)
        order = np.argsort(morton3(coords), kind="stable")
        dom = cls(shape=tuple(int(s) for s in shape), brick=brick,
                  coords=coords[order].astype(np.int32),
                  flags=np.ascontiguousarray(np.asarray(flags, np.uint8)[order]),
                  neighbours=np.empty((0, 27), np.int32), periodic=per)
        dom.neighbours = dom._neighbour_table()
        return dom

    @classmethod
    def from_mask(cls, fluid: np.ndarray, **kw) -> BrickDomain:
        """Build from a boolean fluid mask (True = fluid)."""
        return cls.from_flags(np.where(np.asarray(fluid, bool), FLUID, SOLID), **kw)

    def _neighbour_table(self) -> np.ndarray:
        grid = self.brick_grid
        lut = -np.ones(grid, np.int32)
        c = self.coords.astype(np.int64)
        lut[c[:, 0], c[:, 1], c[:, 2]] = np.arange(len(c), dtype=np.int32)
        self._lut = lut
        out = np.empty((len(c), 27), np.int32)
        for k, off in enumerate(NEIGHBOUR_OFFSETS):
            n = c + off
            ok = np.ones(len(c), bool)
            for ax in range(3):
                if self.periodic[ax]:
                    n[:, ax] %= grid[ax]
                else:
                    ok &= (n[:, ax] >= 0) & (n[:, ax] < grid[ax])
            v = -np.ones(len(c), np.int32)
            v[ok] = lut[n[ok, 0], n[ok, 1], n[ok, 2]]
            out[:, k] = v
        return out

    # ------------------------------------------------------------ properties

    @property
    def brick_grid(self) -> tuple[int, int, int]:
        return tuple(s // self.brick for s in self.shape)  # type: ignore[return-value]

    @property
    def n_bricks(self) -> int:
        return len(self.coords)

    @property
    def cells_per_brick(self) -> int:
        return self.brick ** 3

    @property
    def fluid_cells(self) -> int:
        return int((self.flags != SOLID).sum())

    @property
    def stored_cells(self) -> int:
        return self.n_bricks * self.cells_per_brick

    @property
    def fill(self) -> float:
        """Fraction of stored cells that are not solid (brick fill efficiency)."""
        return self.fluid_cells / max(self.stored_cells, 1)

    @property
    def active_fraction(self) -> float:
        return self.n_bricks / max(int(np.prod(self.brick_grid)), 1)

    def brick_index(self, coords: np.ndarray) -> np.ndarray:
        """Brick index for ``(n, 3)`` brick coordinates; -1 where inactive."""
        c = np.asarray(coords, np.int64).reshape(-1, 3)
        return self._lut[c[:, 0], c[:, 1], c[:, 2]]

    # ------------------------------------------------------------ chunks and partitions

    def chunk_of(self, chunk_bricks: int) -> np.ndarray:
        """``(nb, 3)`` chunk coordinate (z, y, x) of every brick."""
        return self.coords // chunk_bricks

    def chunks(self, chunk_bricks: int) -> tuple[np.ndarray, np.ndarray]:
        """Occupied chunks and, for each brick, the row of its chunk.

        Returns ``(chunk_coords (C, 3), brick_chunk (nb,))``.
        """
        cc = self.chunk_of(chunk_bricks)
        uniq, inv = np.unique(cc, axis=0, return_inverse=True)
        return uniq, inv.reshape(-1)

    def partition(self, n_parts: int, chunk_bricks: int) -> np.ndarray:
        """Assign whole chunks to ``n_parts`` workers, balancing fluid cells.

        Chunks are ordered along a Morton curve and cut into contiguous runs
        of roughly equal fluid work, which keeps each worker's region
        compact (a small halo surface). Returns ``(nb,)`` part ids.
        """
        uniq, inv = self.chunks(chunk_bricks)
        work = np.bincount(inv, weights=(self.flags != SOLID).sum(1), minlength=len(uniq))
        order = np.argsort(morton3(uniq), kind="stable")
        cum = np.cumsum(work[order])
        cuts = np.searchsorted(cum, cum[-1] * np.arange(1, n_parts) / n_parts)
        part_of_ordered = np.zeros(len(uniq), np.int32)
        for p, c in enumerate(cuts, start=1):
            part_of_ordered[c:] = p
        part_of_chunk = np.empty(len(uniq), np.int32)
        part_of_chunk[order] = part_of_ordered
        return part_of_chunk[inv]

    def halo(self, parts: np.ndarray, part: int) -> np.ndarray:
        """Bricks owned by other parts that neighbour ``part``'s bricks (its ghost set)."""
        mine = np.flatnonzero(parts == part)
        nb = self.neighbours[mine].reshape(-1)
        nb = np.unique(nb[nb >= 0])
        return nb[parts[nb] != part]

    def to_dense(self, values: np.ndarray, fill=0.0) -> np.ndarray:
        """Scatter ``(n_bricks, brick^3)`` rows (domain order) into a dense (z, y, x) volume.

        Solid voxels and inactive bricks take ``fill``.
        """
        b = self.brick
        bz, by, bx = self.brick_grid
        values = np.asarray(values)
        out = np.full((bz, by, bx, b, b, b), fill, dtype=values.dtype)
        c = self.coords
        out[c[:, 0], c[:, 1], c[:, 2]] = values.reshape(-1, b, b, b)
        flags = np.full((bz, by, bx, b, b, b), SOLID, np.uint8)
        flags[c[:, 0], c[:, 1], c[:, 2]] = self.flags.reshape(-1, b, b, b)
        vol = out.transpose(0, 3, 1, 4, 2, 5).reshape(self.shape)
        vol[flags.transpose(0, 3, 1, 4, 2, 5).reshape(self.shape) == SOLID] = fill
        return vol

    def summary(self) -> dict:
        return {"shape": self.shape, "brick": self.brick, "bricks": self.n_bricks,
                "active_fraction": self.active_fraction, "fluid_cells": self.fluid_cells,
                "stored_cells": self.stored_cells, "fill": self.fill}


def morton3(coords: np.ndarray) -> np.ndarray:
    """Morton (Z-order) key of non-negative ``(n, 3)`` integer coordinates (< 2^21)."""
    c = np.asarray(coords, np.uint64).reshape(-1, 3)
    key = np.zeros(len(c), np.uint64)
    for bit in range(21):
        for ax in range(3):
            key |= ((c[:, ax] >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3 * bit + 2 - ax)
    return key
