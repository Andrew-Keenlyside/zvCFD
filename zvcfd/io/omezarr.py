"""Reading OME-Zarr inputs with zarr-python v3 directly (no ome-zarr-py), as BRIDGE does.

Accepts the flat ``multiscales`` block of NGFF 0.4 and the nested
``ome`` block of 0.5/0.6, and always resolves ``datasets[level].path``
rather than assuming the array is called ``"0"``. Voxel size comes from
the dataset's ``scale`` transform, reordered to (z, y, x).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Multiscale:
    group: object
    datasets: list[dict]
    axes: list[dict]
    version: str

    def level_path(self, level: int) -> str:
        return self.datasets[level]["path"]

    def voxel_size(self, level: int) -> tuple[float, ...]:
        """Spatial scale of ``level`` as (z, y, x) in the axes' units."""
        ts = self.datasets[level].get("coordinateTransformations", [])
        scale = next((t["scale"] for t in ts if t.get("type") == "scale"), None)
        if scale is None:
            return (1.0, 1.0, 1.0)
        names = [a["name"] for a in self.axes]
        return tuple(float(scale[names.index(n)]) for n in ("z", "y", "x"))

    def unit(self) -> str | None:
        return next((a.get("unit") for a in self.axes if a.get("type") == "space"), None)

    def nearest_level(self, voxel_size: float) -> int:
        """Level whose largest spatial voxel edge is closest to ``voxel_size``."""
        sizes = [max(self.voxel_size(i)) for i in range(len(self.datasets))]
        return int(np.argmin([abs(s - voxel_size) for s in sizes]))


def open_multiscale(path: str) -> Multiscale:
    import zarr

    g = zarr.open_group(path, mode="r")
    attrs = dict(g.attrs)
    if "ome" in attrs:
        block = attrs["ome"]
        version = block.get("version", "0.5")
        ms = block["multiscales"][0]
    elif "multiscales" in attrs:
        ms = attrs["multiscales"][0]
        version = ms.get("version", "0.4")
    else:
        raise ValueError(f"{path}: no OME multiscales metadata")
    return Multiscale(g, ms["datasets"], ms.get("axes", []), version)


def read_level(path: str, level: int = 0, *, region=None) -> tuple[np.ndarray, Multiscale]:
    """Read ``level`` (optionally a ``[[z0, z1], [y0, y1], [x0, x1]]`` crop) as (z, y, x).

    Non-spatial axes (t, c) must have extent 1; they are squeezed out.
    """
    ms = open_multiscale(path)
    arr = ms.group[ms.level_path(level)]
    names = [a["name"] for a in ms.axes] or ["z", "y", "x"][-arr.ndim:]
    sel = []
    for i, n in enumerate(names):
        if n in ("z", "y", "x"):
            k = "zyx".index(n)
            sel.append(slice(*region[k]) if region is not None else slice(None))
        else:
            if arr.shape[i] != 1:
                raise ValueError(f"axis {n!r} has extent {arr.shape[i]}; select one first")
            sel.append(0)
    data = np.asarray(arr[tuple(sel)])
    order = [n for n in names if n in ("z", "y", "x")]
    return np.transpose(data, [order.index(a) for a in "zyx"]), ms


def fluid_mask(volume: np.ndarray, *, threshold: float | None = None,
               label: int | None = None) -> np.ndarray:
    """Fluid voxels of a segmentation or intensity volume."""
    if label is not None:
        return volume == label
    if threshold is not None:
        return volume > threshold
    return volume != 0
