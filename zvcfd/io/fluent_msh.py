"""Streaming reader for ASCII Fluent meshes (``.msh``), boundary-first.

Ansys users hand over meshes in this format (e.g. Simpleware ScanIP+FE
exports of HiP-CT vessel lumens). zvCFD does not solve on unstructured
cells; it needs the *boundary*: the wall surface to voxelise, and the
inlet and outlet patches to place boundary conditions. This reader
therefore parses node coordinates and every non-interior face zone, and
skips the interior faces (the bulk of the file) at C speed via ``mmap``.

Sections understood: ``(10 ...)`` nodes, ``(12 ...)`` cells (counts only),
``(13 ...)`` faces, ``(45 ...)``/``(39 ...)`` zone definitions. Binary
sections (``(2010 ...)``, ``(2013 ...)``) are not supported.
"""

from __future__ import annotations

import mmap
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

_HEADER = re.compile(rb"^\((10|12|13|39|45) \(([^()]*)\)", re.M)
_BC = {2: "interior", 3: "wall", 4: "pressure-inlet", 5: "pressure-outlet", 7: "symmetry",
       8: "periodic-shadow", 9: "pressure-far-field", 10: "velocity-inlet", 12: "periodic",
       14: "fan", 20: "mass-flow-inlet", 24: "interface", 31: "parent", 36: "outflow",
       37: "axis"}


@dataclass
class FaceZone:
    zone: int
    kind: str
    name: str = ""
    faces: list[np.ndarray] = field(default_factory=list)   # node-index polygons (0-based)

    @property
    def n_faces(self) -> int:
        return len(self.faces)


@dataclass
class FluentBoundary:
    """Nodes and boundary face zones of a Fluent mesh."""

    nodes: np.ndarray                       # (N, 3) float64
    zones: dict[int, FaceZone]
    n_cells: int
    n_faces: int
    n_interior_faces: int
    source: str = ""

    def zone_faces(self, zone: int) -> list[np.ndarray]:
        return self.zones[zone].faces

    def triangles(self, zones=None) -> tuple[np.ndarray, np.ndarray]:
        """Fan-triangulate boundary polygons: ``(tris (T, 3), zone_of_tri (T,))``."""
        tris, zid = [], []
        for z, fz in self.zones.items():
            if zones is not None and z not in zones:
                continue
            for poly in fz.faces:
                for k in range(1, len(poly) - 1):
                    tris.append((poly[0], poly[k], poly[k + 1]))
                    zid.append(z)
        return np.asarray(tris, np.int64).reshape(-1, 3), np.asarray(zid, np.int32)

    def summary(self) -> dict:
        """Counts, bounding box, enclosed volume, and per-zone patch areas."""
        tris, zid = self.triangles()
        p = self.nodes
        a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
        cross = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        # divergence theorem over the closed boundary; Fluent boundary normals point
        # into the domain, so take the magnitude
        volume = abs(float(np.sum(np.einsum("ij,ij->i", a, cross)) / 6.0))
        kinds: dict[str, dict] = {}
        patches = []
        for z, fz in sorted(self.zones.items()):
            zarea = float(area[zid == z].sum())
            k = kinds.setdefault(fz.kind, {"zones": 0, "faces": 0, "area": 0.0})
            k["zones"] += 1
            k["faces"] += fz.n_faces
            k["area"] += zarea
            if fz.kind != "wall":
                patches.append({"zone": z, "kind": fz.kind, "name": fz.name, "area": zarea,
                                "equivalent_diameter": float(np.sqrt(4 * zarea / np.pi))})
        used = np.unique(tris)
        lo, hi = p[used].min(0), p[used].max(0)
        return {"source": self.source, "nodes": len(p), "cells": self.n_cells,
                "faces": self.n_faces, "interior_faces": self.n_interior_faces,
                "boundary_faces": int(sum(fz.n_faces for fz in self.zones.values())),
                "bbox_min": lo.tolist(), "bbox_max": hi.tolist(), "volume": volume,
                "kinds": kinds, "patches": patches}


def voxel_estimate(summary: dict, voxel_size: float) -> dict:
    """Fluid and bounding-box voxel counts at ``voxel_size`` (mesh length units)."""
    ext = np.asarray(summary["bbox_max"]) - np.asarray(summary["bbox_min"])
    box = int(np.prod(np.ceil(ext / voxel_size)))
    fluid = summary["volume"] / voxel_size ** 3
    diam = [p["equivalent_diameter"] for p in summary["patches"]]
    return {"voxel_size": voxel_size, "box_voxels": box, "fluid_voxels": fluid,
            "fluid_fraction": fluid / max(box, 1),
            "min_patch_diameter_voxels": (min(diam) / voxel_size) if diam else None,
            "inlet_diameter_voxels": next((p["equivalent_diameter"] / voxel_size
                                           for p in summary["patches"]
                                           if "inlet" in p["kind"]), None)}


def read_fluent_boundary(path: str) -> FluentBoundary:
    """Read nodes and non-interior face zones of an ASCII Fluent mesh."""
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        nodes: list[np.ndarray] = []
        zones: dict[int, FaceZone] = {}
        names: dict[int, tuple[str, str]] = {}
        n_cells = n_faces = n_interior = 0
        for m in _HEADER.finditer(mm):
            kind = int(m.group(1))
            head = m.group(2).split()
            if kind in (39, 45):
                # zone definitions carry *decimal* ids; section headers use hex
                names[int(head[0])] = (head[1].decode(), head[2].decode() if len(head) > 2 else "")
                continue
            vals = [int(h, 16) for h in head]
            zone, first, last = vals[0], vals[1], vals[2]
            if kind == 12:
                if zone == 0:
                    n_cells = last - first + 1
                continue
            body_start = m.end()
            if zone == 0 or mm[body_start:body_start + 2].strip() == b")":
                if kind == 13 and zone == 0:
                    n_faces = last - first + 1
                continue
            open_ = mm.find(b"(", body_start)
            close = mm.find(b")", open_ + 1)
            if kind == 10:
                body = mm[open_ + 1:close]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    xyz = np.fromstring(body, sep=" ")
                nodes.append(xyz.reshape(-1, vals[4] if len(vals) > 4 else 3))
            elif kind == 13:
                bc = vals[3]
                if bc == 2:
                    n_interior += last - first + 1
                    continue
                ftype = vals[4] if len(vals) > 4 else 0
                zones[zone] = FaceZone(zone, _BC.get(bc, str(bc)),
                                       faces=_parse_faces(mm[open_ + 1:close], ftype))
        for z, fz in zones.items():
            if z in names:
                fz.kind, fz.name = names[z][0] or fz.kind, names[z][1]
        xyz = np.concatenate(nodes) if nodes else np.zeros((0, 3))
        return FluentBoundary(xyz, zones, n_cells, n_faces, n_interior, source=str(path))


def _parse_faces(body: bytes, ftype: int) -> list[np.ndarray]:
    """Face records: ``[n,] v1 .. vn c0 c1`` (hex, 1-based) -> 0-based node polygons."""
    faces = []
    for line in body.split(b"\n"):
        tok = line.split()
        if not tok:
            continue
        v = [int(t, 16) for t in tok]
        n = v[0] if ftype in (0, 5) else ftype
        start = 1 if ftype in (0, 5) else 0
        faces.append(np.asarray(v[start:start + n], np.int64) - 1)
    return faces
