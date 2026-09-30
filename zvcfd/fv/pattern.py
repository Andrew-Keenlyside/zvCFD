"""The block sparsity pattern of the coupled system: which nodes each node's equations touch.

In the element-based method a node's equations involve every node of every
element around it: shape functions carry pressure, velocity gradients and
diffusion from the whole element to its integration points. So the
pattern is the union over elements of the complete node graph of each
element, self included. It is built once per mesh, in batches, from
``row * N + col`` keys.
"""

from __future__ import annotations

import numpy as np

from zvcfd.mesh.core import UnstructuredMesh


def node_graph(mesh: UnstructuredMesh, *, batch: int = 1_000_000):
    """CSR ``(indptr, indices)`` of node couplings, rows and columns sorted, self included."""
    N = mesh.n_nodes
    keys = []
    for e in mesh.elements.values():
        n = e.shape[1]
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        for s in range(0, len(e), batch):
            b = e[s:s + batch]
            k = b[:, ii.ravel()] * N + b[:, jj.ravel()]
            keys.append(np.unique(k))
    key = np.unique(np.concatenate(keys))
    rows = key // N
    indptr = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=N), out=indptr[1:])
    return indptr, (key % N).astype(np.int64)


def node_order(mesh: UnstructuredMesh, *, chunk: float | None = None) -> np.ndarray:
    """A node permutation: whole store chunks in Morton order, reverse Cuthill–McKee inside.

    ``order[k]`` is the old index of new node ``k``. Grouping by chunk keeps
    each GPU partition (whole chunks, as ``BrickDomain.partition``) a
    contiguous block of rows. RCM keeps the matrix bandwidth low within it.
    ``chunk`` is the cubic chunk edge in mesh units (None: one chunk).
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    from zvcfd.domain import morton3

    indptr, indices = node_graph(mesh)
    g = sp.csr_matrix((np.ones(len(indices), np.int8), indices, indptr),
                      shape=(mesh.n_nodes, mesh.n_nodes))
    rank = np.empty(mesh.n_nodes, np.int64)
    rank[reverse_cuthill_mckee(g, symmetric_mode=True)] = np.arange(mesh.n_nodes)
    if chunk is None:
        return np.argsort(rank, kind="stable")
    c = np.floor((mesh.nodes - mesh.nodes.min(0)) / chunk).astype(np.int64)
    key = morton3(c[:, ::-1])
    return np.lexsort((rank, key))


def coupling_stats(indptr: np.ndarray) -> dict:
    """Blocks per row: mean, percentiles and total (self included)."""
    per = np.diff(indptr)
    return {"rows": int(len(per)), "blocks": int(indptr[-1]), "mean": float(per.mean()),
            "p50": float(np.median(per)), "p99": float(np.percentile(per, 99)),
            "max": int(per.max())}


def memory_estimate(blocks: int, n_nodes: int, n_ip: int, *, coarse: float = 0.3) -> dict:
    """GPU memory (GB) of the coupled solve, itemised as in ``docs/feasibility/fv_plan.md``.

    The matrix holds 16 values and a 4-byte column index per block. A
    fine-level ILU(0) smoother stores as much again (DILU almost nothing).
    Coarse levels add ``coarse`` × the matrix (operator complexity 1 +
    ``coarse``). Also counted: the lagged ip mass flows, six node-field
    levels of 4 unknowns, and FGMRES(10) work vectors, all double.
    """
    def total(value_bytes, smoother):
        matrix = blocks * (16 * value_bytes + 4) + (n_nodes + 1) * 8
        rest = n_ip * 8 + n_nodes * 4 * 8 * 6 + 2 * 10 * 4 * n_nodes * 8
        return (matrix * (1 + smoother + coarse) + rest) / 1e9

    return {"fp64_ilu0": total(8, 1.0), "fp64_dilu": total(8, 0.0),
            "mixed_ilu0": total(4, 1.0), "mixed_dilu": total(4, 0.0)}


def bandwidth(indptr: np.ndarray, indices: np.ndarray) -> int:
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    return int(np.abs(indices - rows).max(initial=0))


__all__ = ["bandwidth", "coupling_stats", "memory_estimate", "node_graph", "node_order"]
