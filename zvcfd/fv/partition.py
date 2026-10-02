"""Partitions of an unstructured mesh for the multi-GPU finite-volume solver.

**Ownership.** Nodes are grouped by the cubic chunks of the mesh store
(:mod:`zvcfd.io.mesh_store`), the chunks ordered along a Morton (Z-order)
curve, and the curve cut into ``n`` pieces of nearly equal node count at
chunk boundaries. So a partition owns whole store chunks, and its nodes
are spatially compact.

**Local meshes.** Partition ``p`` holds every element with at least one
node it owns, and the boundary faces touching those nodes. Its local
nodes are its owned nodes, then the *halo*: the other nodes of those
elements. Every quantity the discrete equations need at an owned node (the
control volume, its flux points, its boundary sub-faces) is then complete
locally; halo values come from their owners by :func:`halo_plan`.

The overlap (halo over owned nodes) is reported; CFX recommends keeping it
under about 10 %, which needs a few hundred thousand nodes per partition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zvcfd.mesh.core import BoundaryZone, UnstructuredMesh


def _morton3(ijk: np.ndarray, bits: int = 21) -> np.ndarray:
    """Interleave the bits of three non-negative integer coordinates (Z-order key)."""
    key = np.zeros(len(ijk), np.uint64)
    v = ijk.astype(np.uint64)
    for b in range(bits):
        for a in range(3):
            key |= ((v[:, a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return key


def morton_owner(nodes: np.ndarray, n_parts: int, chunk: float | None = None) -> np.ndarray:
    """Owner partition of each node: whole chunks along a Morton curve, balanced by nodes.

    ``chunk`` is the store chunk edge; the default gives about 64 chunks per
    partition, so the cuts can balance node counts to a few per cent.
    """
    if n_parts < 1:
        raise ValueError("n_parts must be at least 1")
    lo, hi = nodes.min(0), nodes.max(0)
    if chunk is None:
        vol = float(np.prod(np.maximum(hi - lo, 1e-300)))
        chunk = (vol / (64 * n_parts)) ** (1 / 3)
    ijk = np.floor((nodes - lo) / chunk).astype(np.int64)
    key = _morton3(ijk)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])      # first node of each chunk
    owner = np.empty(len(nodes), np.int64)
    target = len(nodes) * np.arange(1, n_parts) / n_parts
    cuts = starts[np.clip(np.searchsorted(starts, target), 0, len(starts) - 1)]
    bounds = np.r_[0, cuts, len(nodes)]
    for p in range(n_parts):
        owner[order[bounds[p]:bounds[p + 1]]] = p
    if len(np.unique(owner)) != n_parts:
        raise ValueError(f"cannot cut {len(nodes)} nodes into {n_parts} partitions of whole "
                         "chunks: use a smaller chunk")
    return owner


@dataclass
class LocalMesh:
    """One partition: its local mesh, node maps and face maps."""

    part: int
    mesh: UnstructuredMesh        # local numbering: owned nodes first, then halo
    nodes: np.ndarray             # (n_local,) global node of each local node
    n_own: int
    elements: dict                # kind -> global element indices held
    faces: dict                   # zone -> global face indices held

    @property
    def n_halo(self) -> int:
        return len(self.nodes) - self.n_own


def local_meshes(mesh: UnstructuredMesh, owner: np.ndarray) -> list[LocalMesh]:
    """The local mesh of every partition (see the module docstring)."""
    n_parts = int(owner.max()) + 1
    out = []
    for p in range(n_parts):
        own = np.flatnonzero(owner == p)
        mine = owner == p
        elems, used = {}, [own]
        for k, e in mesh.elements.items():
            sel = np.flatnonzero(mine[e].any(1))
            elems[k] = sel
            used.append(e[sel].reshape(-1))
        allnodes = np.unique(np.concatenate(used))
        halo = allnodes[~mine[allnodes]]
        halo = halo[np.lexsort((halo, owner[halo]))]              # grouped by owner
        g = np.concatenate([own, halo])
        loc = -np.ones(mesh.n_nodes, np.int64)
        loc[g] = np.arange(len(g))
        elements = {k: loc[mesh.elements[k][sel]] for k, sel in elems.items() if len(sel)}
        zones, fmap = {}, {}
        for zid, z in mesh.zones.items():
            f = z.faces
            ok = f >= 0
            touch = (mine[np.where(ok, f, 0)] & ok).any(1)
            sel = np.flatnonzero(touch)
            lf = np.where(ok[sel], loc[np.where(ok[sel], f[sel], 0)], -1)
            zones[zid] = BoundaryZone(zid, z.kind, z.name, lf, z.cells[sel])
            fmap[zid] = sel
        lm = UnstructuredMesh(mesh.nodes[g], elements, zones=zones, unit=mesh.unit,
                              source=mesh.source)
        out.append(LocalMesh(p, lm, g, len(own), elems, fmap))
    return out


def halo_plan(parts: list[LocalMesh], owner: np.ndarray) -> list[list[tuple]]:
    """For each partition ``p``: ``[(q, send, recv)]`` — copy ``q``'s local rows ``send``
    (owned) into ``p``'s local rows ``recv`` (halo)."""
    n = len(owner)
    own_index = np.empty(n, np.int64)
    for q in parts:
        own_index[q.nodes[:q.n_own]] = np.arange(q.n_own)
    plans = []
    for p in parts:
        halo = p.nodes[p.n_own:]
        rows = p.n_own + np.arange(len(halo))
        who = owner[halo]
        plan = []
        for q in np.unique(who):
            sel = who == q
            plan.append((int(q), own_index[halo[sel]], rows[sel]))
        plans.append(plan)
    return plans


def partition_report(parts: list[LocalMesh]) -> dict:
    """Owned and halo nodes per partition, and the overlap."""
    own = [p.n_own for p in parts]
    halo = [p.n_halo for p in parts]
    return {"parts": len(parts), "owned": own, "halo": halo,
            "overlap": [h / o for h, o in zip(halo, own)],
            "imbalance": max(own) / (sum(own) / len(own)) - 1.0}


__all__ = ["LocalMesh", "halo_plan", "local_meshes", "morton_owner", "partition_report"]
