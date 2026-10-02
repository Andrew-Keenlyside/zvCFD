"""GPU evaluation of the finite-volume discrete equations (CuPy, double precision).

:class:`GPUAssembler` computes, on the device, the three things the
coupled solver's assembly needs, from node coordinates and connectivity
alone:

- nodal gradients (SCV-weighted element gradients);
- the momentum diagonal ``a_P``, which sets the Rhie–Chow coefficient;
- the residual of every node's momentum and continuity equations for a
  given state (ip fluxes only; boundary and body-force terms are added
  by the caller, as in the reference solver).

The kernels (:mod:`zvcfd.fv._cuda`) are an independent implementation of
``docs/spec/fv_numerics.md``. ``tests/test_fv_gpu.py`` checks them against
the CPU reference on random states and meshes of every element type.
Elements are processed one colour at a time (no two elements of a colour
share a node), so there are no atomics and results are deterministic.
"""

from __future__ import annotations

import numpy as np

from zvcfd.fv import _cuda
from zvcfd.fv.geometry import topology
from zvcfd.mesh.core import UnstructuredMesh


def colour_elements(elem: np.ndarray, n_nodes: int, *, greedy_limit: int = 20000,
                    seed: int = 0) -> list[np.ndarray]:
    """Element colours: lists of element indices, no two in a list sharing a node.

    Small meshes use a sequential greedy colouring. Larger ones use the
    parallel form of the same thing (Jones & Plassmann 1993, first fit): in
    rounds, the uncoloured elements whose random key is lowest at every one
    of their nodes form an independent set, and each takes the smallest
    colour none of its nodes has used. That is vectorised and deterministic
    (fixed seed), and needs about as few colours as greedy (the lower bound
    is the most elements meeting at one node).
    """
    if len(elem) > greedy_limit:
        return _colour_first_fit(elem, n_nodes, seed)
    return _colour_greedy(elem, n_nodes)


def _first_zero_bit(masks: np.ndarray) -> np.ndarray:
    """Index of the lowest zero bit of each row of ``(k, words)`` uint64 bitmasks."""
    inv = ~masks
    word = np.argmax(inv != 0, axis=1)
    x = inv[np.arange(len(inv)), word]
    lsb = x & (~x + np.uint64(1))
    return word * 64 + np.log2(lsb.astype(np.float64)).astype(np.int64)


def _colour_first_fit(elem, n_nodes, seed, words: int = 8):
    rng = np.random.default_rng(seed)
    key = rng.permutation(len(elem)).astype(np.int64)
    used = np.zeros((n_nodes, words), np.uint64)          # colours used at each node
    colour = np.empty(len(elem), np.int64)
    left = np.arange(len(elem))
    nn = elem.shape[1]
    while len(left):
        e = elem[left]
        best = np.full(n_nodes, np.iinfo(np.int64).max)
        np.minimum.at(best, e.reshape(-1), np.repeat(key[left], nn))
        win = (best[e] == key[left][:, None]).all(1)
        chosen, ec = left[win], e[win]
        c = _first_zero_bit(np.bitwise_or.reduce(used[ec], axis=1))
        if c.max() >= 64 * words:
            raise RuntimeError(f"more than {64 * words} colours")
        colour[chosen] = c
        bit = np.left_shift(np.uint64(1), (c % 64).astype(np.uint64))
        for j in range(nn):                               # chosen elements share no node
            used[ec[:, j], c // 64] |= bit
        left = left[~win]
    return [np.flatnonzero(colour == k) for k in range(colour.max() + 1)]


def _colour_greedy(elem: np.ndarray, n_nodes: int) -> list[np.ndarray]:
    """Greedy colouring: lists of element indices, no two in a list sharing a node."""
    used = np.zeros(n_nodes, np.uint64)          # bit c set: colour c used at the node
    colour = np.empty(len(elem), np.int64)
    for e, row in enumerate(elem):
        taken = np.bitwise_or.reduce(used[row])
        c = 0
        while (int(taken) >> c) & 1:
            c += 1
        if c >= 64:
            raise RuntimeError("more than 64 colours")
        colour[e] = c
        used[row] |= np.uint64(1 << c)
    return [np.flatnonzero(colour == c) for c in range(colour.max() + 1)]


_STORED_BYTES = {"tet": 8 * (12 + 36 + 36 + 4) + 4 * 16}       # per element, with positions


def _stored_bytes(kind: str) -> int:
    from zvcfd.fv.geometry import topology

    if kind in _STORED_BYTES:
        return _STORED_BYTES[kind]
    t = topology(kind)
    return 8 * (t.n_ip * t.n * 3 + 2 * t.n_ip * 3 + t.n) + 4 * t.n * t.n


def _auto_store(mesh, fraction: float = 0.2):
    """``True``, ``"tets"`` or ``False``: what fits in ``fraction`` of the free device memory."""
    import cupy as cp

    free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
    need = {k: len(e) * _stored_bytes(k) for k, e in mesh.elements.items()}
    if sum(need.values()) <= fraction * free:
        return True
    if need.get("tet", 0) <= fraction * free:
        return "tets"
    return False


class GPUAssembler:
    """Device kernels for one mesh. Needs cupy and a CUDA device.

    Args:
        mesh: the mesh.
        rho, mu: density and (constant) viscosity.
        rheology: None (Newtonian ``mu``) or a :class:`zvcfd.rheology.CarreauYasuda`,
            evaluated at each flux point from the local strain rate (no clips).
        transpose: include the ``μ (∇u)ᵀ`` stress term.
        stokes: no advection.
        block: threads per block.
        store_geometry: keep each element's flux-point areas and centroids, sub-volumes
            and shape-function gradients on the device (computed once), and the
            assembly's block positions, instead of recomputing them in every kernel:
            ``True`` for every element type, ``"tets"`` for tetrahedra only (whose
            gradients are constant: 0.7 KB per tetrahedron against 3.5 KB per
            wedge and 5.8 KB per hexahedron), ``False`` to recompute everything, or
            ``"auto"`` (default): everything if it takes under 20 % of the free device
            memory, else tetrahedra only if they do, else nothing. (The HiP-CT
            coronary mesh: 6.5 M tetrahedra, 5.0 GB; its 8.2 M wedges would add 30 GB.)
    """

    def __init__(self, mesh: UnstructuredMesh, *, rho: float = 1.0, mu: float = 1.0,
                 rheology=None, transpose: bool = False, stokes: bool = False,
                 block: int = 128, store_geometry="auto"):
        import cupy as cp

        self.cp = cp
        self.mesh, self.rho, self.mu = mesh, float(rho), float(mu)
        self.transpose, self.stokes, self.block = int(transpose), int(stokes), block
        if rheology is None:
            self.rheo = (np.int32(0), np.float64(mu), np.float64(mu), np.float64(0.0),
                         np.float64(2.0), np.float64(1.0))
        else:
            r = rheology
            self.rheo = (np.int32(1), np.float64(r.mu_0), np.float64(r.mu_inf),
                         np.float64(r.lam), np.float64(r.a), np.float64(r.n))
        self.N = mesh.n_nodes
        self.X = cp.asarray(mesh.nodes, dtype=cp.float64)
        if store_geometry == "auto":
            store_geometry = _auto_store(mesh)
        self.store_mode = store_geometry
        self.kinds = []
        for k, e in mesh.elements.items():
            t = topology(k)
            self.kinds.append({
                "kind": k, "topo": t, "elem": cp.asarray(e, dtype=cp.int64),
                "colours": [cp.asarray(c, dtype=cp.int64) for c in colour_elements(e, self.N)],
                "W": cp.asarray(t.point_weights, dtype=cp.float64),
                "tris": cp.asarray(t.tris, dtype=cp.int32),
                "edges": cp.asarray(t.edges, dtype=cp.int32),
                "N": cp.asarray(t.N_ip, dtype=cp.float64),
                "dN": cp.asarray(t.dN_ip, dtype=cp.float64),
                "stored": bool(store_geometry is True or (store_geometry == "tets" and k == "tet")),
                "constg": int(k == "tet"),
                "kernels": {}, "pos": {}})
        dummy = cp.zeros(1)
        for K in self.kinds:
            if K["stored"]:
                E, t = int(K["elem"].shape[0]), K["topo"]
                ng = E * t.n * 3 if K["constg"] else E * t.n_ip * t.n * 3
                K["geo"] = (cp.zeros(ng), cp.zeros(E * t.n_ip * 3), cp.zeros(E * t.n_ip * 3),
                            cp.zeros(E * t.n))
                self._launch(K, self._kernel(K, "geostore"), K["geo"], geo=False)
            else:
                K["geo"] = (dummy,) * 4

    def geometry_bytes(self) -> int:
        """Device memory of the stored geometry (bytes)."""
        return int(sum(sum(a.nbytes for a in K["geo"]) for K in self.kinds if K["stored"]))

    # ------------------------------------------------------------ compilation

    def _kernel(self, K, name, nc=1):
        key = (name, nc)
        if key not in K["kernels"]:
            t = K["topo"]
            subs = {"n": t.n, "nip": t.n_ip, "npts": t.point_weights.shape[0], "nc": nc,
                    "N": self.N, "stored": int(K["stored"] and name != "geostore"),
                    "constg": K["constg"]}
            src = (_cuda.COMMON + getattr(_cuda, name.upper())) % subs
            K["kernels"][key] = self.cp.RawKernel(src, name, options=("-std=c++14",))
        return K["kernels"][key]

    def _launch(self, K, kern, args, geo=True):
        """Run a kernel over every colour of one element type (in stream order, no sync)."""
        extra = tuple(K["geo"]) if geo else ()
        for lst in K["colours"]:
            n = int(lst.size)
            grid = (n + self.block - 1) // self.block
            kern((grid,), (self.block,), (K["elem"], lst, np.int64(n), self.X, K["W"],
                                          K["tris"], K["dN"], K["edges"]) + extra + tuple(args))

    def _positions(self, K, pattern):
        """Block positions ``(E, n, n)`` int32 of one element type in ``pattern`` (cached)."""
        cp = self.cp
        key = (id(pattern[1]), int(pattern[1].size))
        if key not in K["pos"]:
            K["pos"].clear()
            t = K["topo"]
            pos = cp.zeros(int(K["elem"].shape[0]) * t.n * t.n, dtype=cp.int32)
            self._launch(K, self._kernel(K, "blockpos"), (pattern[0], pattern[1], pos))
            K["pos"][key] = pos
        return K["pos"][key]

    # ------------------------------------------------------------ kernels

    def gradient(self, phi):
        """Nodal gradients of ``phi (N,)`` or ``(N, C)``: ``(N, 3)`` or ``(N, C, 3)``."""
        cp = self.cp
        phi = cp.asarray(phi, dtype=cp.float64)
        vec = phi.ndim == 2
        nc = phi.shape[1] if vec else 1
        out = cp.zeros(self.N * nc * 3 + self.N)
        flat = cp.ascontiguousarray(phi.reshape(self.N, nc))
        for K in self.kinds:
            self._launch(K, self._kernel(K, "gradient", nc), (flat, out))
        vol = out[self.N * nc * 3:]
        g = out[:self.N * nc * 3].reshape(self.N, nc, 3) / vol[:, None, None]
        self.node_volume = vol
        return g if vec else g[:, 0]

    def diagonal(self, mdot: dict, U=None):
        """Momentum diagonal ``a_P`` (N,), with ``mdot = {kind: (E, n_ip)}`` lagged mass flows.

        ``U`` is needed for a generalised-Newtonian viscosity (strain rate at each ip).
        """
        cp = self.cp
        diag = cp.zeros(self.N)
        U_ = self._vec(U if U is not None else cp.zeros((self.N, 3)))
        for K in self.kinds:
            m = cp.asarray(mdot[K["kind"]], dtype=cp.float64)
            self._launch(K, self._kernel(K, "diagonal"),
                         (cp.ascontiguousarray(m), U_) + self.rheo +
                         (np.int32(self.transpose), np.int32(self.stokes), diag))
        return diag

    def _vec(self, a):
        return self.cp.ascontiguousarray(self.cp.asarray(a, dtype=self.cp.float64).reshape(-1))

    def residual(self, U, P, mdot: dict, gradU, beta, gradP, dnode):
        """Momentum and continuity residuals ``(N, 4)`` of the ip fluxes for the given state."""
        cp = self.cp
        R = cp.zeros(self.N * 4)
        args = [cp.ascontiguousarray(cp.asarray(a, dtype=cp.float64).reshape(-1))
                for a in (U, P, gradU, beta, gradP, dnode)]
        U_, P_, gU, be, gP, dn = args
        for K in self.kinds:
            m = cp.ascontiguousarray(cp.asarray(mdot[K["kind"]], dtype=cp.float64))
            self._launch(K, self._kernel(K, "residual"),
                         (K["N"], U_, P_, m, gU, be, gP, dn, np.float64(self.rho)) + self.rheo +
                         (np.int32(self.transpose), np.int32(self.stokes), R))
        return R.reshape(self.N, 4)

    def _levels(self, levels, kind):
        """Kernel arguments for up to two old time levels ``[(c, U_old, mdot_old), ...]``."""
        cp = self.cp
        dummy = cp.zeros(1)
        out = [np.int32(len(levels))]
        for i in range(2):
            if i < len(levels):
                c, Uo, mo = levels[i]
                out += [np.float64(c), self._vec(Uo),
                        cp.ascontiguousarray(cp.asarray(mo[kind], dtype=cp.float64))]
            else:
                out += [np.float64(0.0), dummy, dummy]
        return tuple(out)

    def assemble(self, pattern, U, mdot: dict, gradU, beta, gradP, dnode, snode, *,
                 levels=(), own=None):
        """Block matrix values ``(nnzb, 4, 4)`` and right-hand side ``(N, 4)`` of the ip fluxes.

        ``pattern`` is ``(indptr, indices)`` on the device (sorted columns). The
        Rhie–Chow ``∇̄p`` term is lagged (on the right-hand side), as CFX does.
        ``levels`` are the old time levels ``[(c_l / c_0, U_l, mdot_l)]`` of the
        transient Rhie–Chow part (a false time step: ``[(1, U, mdot)]``).
        ``own`` (N,) flags nodes whose row leaves out the deferred corrections of
        the faces they are upwind of (inflow through a pressure boundary).
        Boundary rows, body forces, known boundary mass flows and the time term
        are added by the caller.
        """
        cp = self.cp
        indptr, indices = pattern
        data = cp.zeros(int(indices.size) * 16)
        b = cp.zeros(self.N * 4)
        args = [self._vec(a) for a in (U, gradU, beta, gradP, dnode, snode)]
        U_, gU, be, gP, dn, sn = args
        if own is None:
            own = cp.zeros(self.N, dtype=cp.int32)
        own = cp.ascontiguousarray(cp.asarray(own, dtype=cp.int32))
        for K in self.kinds:
            m = cp.ascontiguousarray(cp.asarray(mdot[K["kind"]], dtype=cp.float64))
            pos = self._positions(K, pattern) if K["stored"] else cp.zeros(1, dtype=cp.int32)
            self._launch(K, self._kernel(K, "assemble"),
                         (K["N"], U_, m, gU, be, gP, dn, sn) + self._levels(levels, K["kind"]) +
                         (np.float64(self.rho),) + self.rheo +
                         (np.int32(self.transpose), np.int32(self.stokes), own, indptr,
                          indices, pos, data, b))
        return data.reshape(-1, 4, 4), b.reshape(self.N, 4)

    def massflow(self, U, P, gradP, dnode, snode, *, levels=()):
        """New Rhie–Chow ip mass flows ``{kind: (E, n_ip)}`` and each node's net outflow (N,)."""
        cp = self.cp
        imb = cp.zeros(self.N)
        out = {}
        U_, P_, gP, dn, sn = (self._vec(a) for a in (U, P, gradP, dnode, snode))
        for K in self.kinds:
            E = int(K["elem"].shape[0])
            m = cp.zeros(E * K["topo"].n_ip)
            self._launch(K, self._kernel(K, "massflow"),
                         (K["N"], U_, P_, gP, dn, sn) + self._levels(levels, K["kind"]) +
                         (np.float64(self.rho), m, imb))
            out[K["kind"]] = m.reshape(E, -1)
        return out, imb

    def limiter(self, U, gradU):
        """Barth–Jespersen blend factors ``(N, 3)``."""
        cp = self.cp
        U_ = self._vec(U)
        lo = U_.copy()
        hi = U_.copy()
        for K in self.kinds:
            self._launch(K, self._kernel(K, "minmax"), (U_, lo, hi))
        beta = cp.ones(self.N * 3)
        gU = self._vec(gradU)
        for K in self.kinds:
            self._launch(K, self._kernel(K, "limiter"), (U_, gU, lo, hi, beta))
        return beta.reshape(self.N, 3)


__all__ = ["GPUAssembler", "colour_elements"]
