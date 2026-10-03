"""The fast GPU assembler: slot-ordered elements, per-block threads, node-based gradients.

:class:`FastAssembler` evaluates the same discrete equations as
:class:`zvcfd.fv.gpu.GPUAssembler` with the kernels of
:mod:`zvcfd.fv._cuda_fast`, and does its whole setup on the device:

- **Colouring** by a parallel Jones–Plassmann first fit (:func:`gpu_colour`):
  deterministic keys, rounds of independent sets, each element taking the
  smallest colour its nodes have not used.
- **Slots**: each type's elements in colour order, so a launch covers a
  contiguous range; connectivity (int32), mass flows, block positions and
  stored geometry follow that order and are read coalesced.
- **The pattern** (:func:`gpu_node_graph`): sorted unique node pairs of the
  elements, on the device.
- **Statics**, once per mesh: node volumes, the Newtonian viscous part of
  the momentum diagonal, and the nodal-gradient operator on the pattern
  (three coefficients per block), so a gradient is one sparse product
  (8 lanes per row) rather than a pass over every colour.

Mass flows are held in slot order (``{kind: (E, n_ip)}`` with rows in slot
order); :meth:`FastAssembler.to_slots` and :meth:`FastAssembler.from_slots`
convert from and to element order. :meth:`FastAssembler.massflow` also
returns the advective part of the next momentum diagonal, so a Newtonian
solve needs no diagonal pass (``diagonal = dvisc + adv``).
"""

from __future__ import annotations

import numpy as np

from zvcfd.fv import _cuda_fast as K_
from zvcfd.fv.geometry import topology
from zvcfd.mesh.core import UnstructuredMesh

_COLOUR = r"""
extern "C" __global__ void colour_claim(const int *conn, int n, long long E,
                                        const unsigned long long *key, const int *colour,
                                        unsigned long long *best) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E || colour[e] >= 0) return;
    unsigned long long k = key[e];
    for (int a = 0; a < n; ++a) atomicMin(&best[conn[e * n + a]], k);
}

// ctr[0]: elements still uncoloured after this round; ctr[1]: elements that found no free
// colour in W words (the caller widens the masks and starts again)
extern "C" __global__ void colour_pick(const int *conn, int n, long long E,
                                       const unsigned long long *key, int *colour,
                                       const unsigned long long *best,
                                       unsigned long long *used, int W,
                                       unsigned long long *ctr) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E || colour[e] >= 0) return;
    unsigned long long k = key[e];
    for (int a = 0; a < n; ++a)
        if (best[conn[e * n + a]] != k) { atomicAdd(&ctr[0], 1ull); return; }
    // winners of a round share no node: their reads and writes of `used` never overlap
    int c = -1;
    for (int w = 0; w < W && c < 0; ++w) {
        unsigned long long m = 0ull;
        for (int a = 0; a < n; ++a) m |= used[(long long)conn[e * n + a] * W + w];
        unsigned long long free_ = ~m;
        if (free_) c = w * 64 + (__ffsll((long long)free_) - 1);
    }
    if (c < 0) { atomicAdd(&ctr[1], 1ull); return; }       // out of colours: flagged
    colour[e] = c;
    unsigned long long bit = 1ull << (c % 64);
    for (int a = 0; a < n; ++a) used[(long long)conn[e * n + a] * W + c / 64] |= bit;
}
"""

_kernels: dict = {}


def _raw(name: str, src: str, opts=("-std=c++14",)):
    import cupy as cp

    key = (cp.cuda.Device().id, name, hash(src))
    if key not in _kernels:
        _kernels[key] = cp.RawKernel(src, name, options=opts)
    return _kernels[key]


def _keys(E: int):
    """Deterministic distinct pseudo-random keys: a hash in the high bits, the index low."""
    import cupy as cp

    e = cp.arange(E, dtype=cp.uint64)
    z = e + cp.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> cp.uint64(30))) * cp.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> cp.uint64(27))) * cp.uint64(0x94D049BB133111EB)
    z = z ^ (z >> cp.uint64(31))
    return (z & cp.uint64(0xFFFFFFFF00000000)) | e


def gpu_colour(conn, n_nodes: int, words: int = 4):
    """Colour of each element ``(E,)`` int32: no two elements of a colour share a node.

    Each node keeps a mask of the colours its elements took, ``words`` × 64
    bits. If an element finds none free, the masks are doubled and the
    colouring starts again. A first fit never depends on the mask width
    when the colours fit, so the result is the same as with wide masks from
    the start.
    """
    import cupy as cp

    E, n = int(conn.shape[0]), int(conn.shape[1])
    flat = cp.ascontiguousarray(conn.astype(cp.int32).reshape(-1))
    key = _keys(E)
    best = cp.empty(n_nodes, dtype=cp.uint64)
    ctr = cp.zeros(2, dtype=cp.uint64)
    claim, pick = _raw("colour_claim", _COLOUR), _raw("colour_pick", _COLOUR)
    grid = ((E + 255) // 256,)
    while True:
        colour = cp.full(E, -1, dtype=cp.int32)
        used = cp.zeros(n_nodes * words, dtype=cp.uint64)
        for _ in range(1000000):
            best.fill(np.uint64(np.iinfo(np.uint64).max))
            ctr.fill(0)
            claim(grid, (256,), (flat, np.int32(n), np.int64(E), key, colour, best))
            pick(grid, (256,), (flat, np.int32(n), np.int64(E), key, colour, best, used,
                                np.int32(words), ctr))
            remaining, full = (int(v) for v in ctr.get())
            if full or remaining == 0:
                break
        else:
            raise RuntimeError("colouring did not finish")
        if not full:
            return colour
        words *= 2


def gpu_node_graph(elements: dict, n_nodes: int):
    """The coupling pattern on the device: ``(indptr int64 (N+1,), indices int32)``, sorted."""
    import cupy as cp

    keys = []
    for e in elements.values():
        e = cp.asarray(e, dtype=cp.int64)
        n = e.shape[1]
        keys.append((e[:, :, None] * n_nodes + e[:, None, :]).reshape(-1))
        del n
    keys = cp.unique(cp.concatenate(keys))
    rows = keys // n_nodes
    indptr = cp.zeros(n_nodes + 1, dtype=cp.int64)
    indptr[1:] = cp.cumsum(cp.bincount(rows, minlength=n_nodes))
    return indptr, (keys % n_nodes).astype(cp.int32)


def _stored_bytes(topo, constg: bool) -> int:
    ng = 1 if constg else topo.n_ip
    return 8 * (2 * topo.n_ip * 3 + ng * topo.n * 3 + topo.n)


class FastAssembler:
    """Device evaluation of the discrete equations (see the module docstring).

    Args:
        mesh, rho, mu, rheology, transpose, stokes: as :class:`zvcfd.fv.gpu.GPUAssembler`.
        store_geometry: ``"auto"`` (every type if it fits in ``fraction`` of the free
            device memory, else tetrahedra only if they fit, else none), ``True``,
            ``"tets"`` or ``False``. Unstored geometry is rebuilt in shared memory
            by each kernel (one thread per flux point).
        pattern: the coupling pattern ``(indptr, indices)`` on the device, if already
            built.
    """

    def __init__(self, mesh: UnstructuredMesh, *, rho: float = 1.0, mu: float = 1.0,
                 rheology=None, transpose: bool = False, stokes: bool = False,
                 store_geometry="auto", pattern=None, fraction: float = 0.2):
        import cupy as cp

        self.cp = cp
        self.mesh, self.rho, self.mu = mesh, float(rho), float(mu)
        self.transpose, self.stokes = int(transpose), int(stokes)
        if rheology is None:
            self.rheo = (np.int32(0), np.float64(mu), np.float64(mu), np.float64(0.0),
                         np.float64(2.0), np.float64(1.0))
        else:
            r = rheology
            self.rheo = (np.int32(1), np.float64(r.mu_0), np.float64(r.mu_inf),
                         np.float64(r.lam), np.float64(r.a), np.float64(r.n))
        self.N = mesh.n_nodes
        self.X = cp.asarray(mesh.nodes, dtype=cp.float64)
        self.pattern = pattern if pattern is not None else gpu_node_graph(mesh.elements, self.N)
        indptr, indices = self.pattern
        if store_geometry == "auto":
            free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
            need = {k: len(e) * _stored_bytes(topology(k), k == "tet")
                    for k, e in mesh.elements.items()}
            store_geometry = True if sum(need.values()) <= fraction * free else \
                "tets" if need.get("tet", 0) <= fraction * free else False
        self.store_mode = store_geometry
        self.kinds = []
        for k, e in mesh.elements.items():
            t = topology(k)
            stored = bool(store_geometry is True or (store_geometry == "tets" and k == "tet"))
            ed = cp.asarray(e, dtype=cp.int32)
            colour = gpu_colour(ed, self.N)
            order = cp.argsort(colour.astype(cp.int64) * len(e) + cp.arange(len(e)))
            counts = cp.asnumpy(cp.bincount(colour, minlength=int(colour.max()) + 1))
            offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            K = {"kind": k, "topo": t, "n": t.n, "nip": t.n_ip, "E": len(e),
                 "order": order,                                  # slot -> element
                 "conn": cp.ascontiguousarray(ed[order].reshape(-1)),
                 "offsets": offsets, "stored": stored, "constg": int(k == "tet"),
                 "epb": K_.epb(t.n), "src": {}}
            self.kinds.append(K)
        self.node_volume = cp.zeros(self.N)
        self.dvisc = cp.zeros(self.N)
        self.gop = cp.zeros(int(indices.size) * 3)
        for K in self.kinds:
            if K["stored"]:
                K["geo"] = cp.zeros(K["E"] * self._rec(K))
                g = self._kernel(K, "geobuild", stored=0)
                nb = (K["E"] + K["epb"] - 1) // K["epb"]
                g((nb,), (K["epb"] * K["n"] ** 2,), (K["conn"], self.X, np.int64(K["E"]),
                                                     K["geo"]))
            else:
                K["geo"] = cp.zeros(1)
            nt = K["E"] * K["n"] ** 2
            K["pos"] = cp.zeros(nt, dtype=cp.int32)
            bp = self._kernel(K, "blockpos")
            bp(((nt + 255) // 256,), (256,), (K["conn"], np.int64(K["E"]), indptr, indices,
                                             K["pos"]))
            self._launch(K, "statics", (np.float64(self.rheo[1] if rheology is None else 0.0),
                                        np.int32(self.transpose), self.node_volume,
                                        self.dvisc, self.gop), pre=(K["pos"],))
        rows = cp.searchsorted(indptr, cp.arange(int(indices.size), dtype=cp.int64),
                               side="right") - 1
        g3 = self.gop.reshape(-1, 3)
        g3 /= self.node_volume[rows][:, None]
        self._node = cp.RawModule(code=K_.NODE, options=("-std=c++14",))
        self._grad_k = self._node.get_function("csr_gradient")
        self._minmax_k = self._node.get_function("csr_minmax")

    # ------------------------------------------------------------ plumbing

    @staticmethod
    def _rec(K) -> int:
        ng = 1 if K["constg"] else K["nip"]
        return 2 * K["nip"] * 3 + ng * K["n"] * 3 + K["n"]

    def _kernel(self, K, name, stored=None):
        stored = int(K["stored"]) if stored is None else stored
        key = (name, stored)
        if key not in K["src"]:
            t = K["topo"]
            subs = {"n": t.n, "nip": t.n_ip, "npts": t.point_weights.shape[0],
                    "epb": K["epb"], "stored": stored, "constg": K["constg"],
                    "tables": K_.tables(t)}
            src = (K_.HEADER % subs) + (getattr(K_, name.upper()) % subs)
            K["src"][key] = _raw(name, src)
        return K["src"][key]

    def _launch(self, K, name, args, pre=()):
        """One launch per colour: slots ``[base, base + count)``."""
        kern = self._kernel(K, name)
        nt = K["epb"] * K["n"] ** 2
        off = K["offsets"]
        for c in range(len(off) - 1):
            base, count = int(off[c]), int(off[c + 1] - off[c])
            nb = (count + K["epb"] - 1) // K["epb"]
            kern((nb,), (nt,), (K["conn"], np.int64(base), np.int64(count), self.X, K["geo"])
                 + tuple(pre) + tuple(args))

    def geometry_bytes(self) -> int:
        return int(sum(K["geo"].nbytes for K in self.kinds if K["stored"]))

    def static_bytes(self) -> int:
        """Device memory held for the whole run: geometry, block positions, gradient operator."""
        return self.geometry_bytes() + int(sum(K["pos"].nbytes + K["conn"].nbytes
                                               for K in self.kinds)) + int(self.gop.nbytes)

    def to_slots(self, mdot: dict) -> dict:
        """Mass flows from element order to slot order."""
        cp = self.cp
        return {K["kind"]: cp.ascontiguousarray(cp.asarray(mdot[K["kind"]])[K["order"]])
                for K in self.kinds}

    def from_slots(self, mdot: dict) -> dict:
        cp = self.cp
        out = {}
        for K in self.kinds:
            a = cp.empty_like(mdot[K["kind"]])
            a[K["order"]] = mdot[K["kind"]]
            out[K["kind"]] = a
        return out

    def zero_mdot(self) -> dict:
        return {K["kind"]: self.cp.zeros((K["E"], K["nip"])) for K in self.kinds}

    def _vec(self, a):
        return self.cp.ascontiguousarray(self.cp.asarray(a, dtype=self.cp.float64).reshape(-1))

    def _levels(self, levels, kind):
        cp = self.cp
        dummy = cp.zeros(1)
        out = [np.int32(len(levels))]
        for i in range(2):
            if i < len(levels):
                c, Uo, mo = levels[i]
                out += [np.float64(c), self._vec(Uo), cp.ascontiguousarray(mo[kind])]
            else:
                out += [np.float64(0.0), dummy, dummy]
        return tuple(out)

    # ------------------------------------------------------------ kernels

    def gradient(self, phi):
        """Nodal gradients of ``phi (N,)`` or ``(N, C)``, ``C ≤ 4``: ``(N, 3)`` or ``(N, C, 3)``."""
        cp = self.cp
        phi = cp.asarray(phi, dtype=cp.float64)
        vec = phi.ndim == 2
        nc = phi.shape[1] if vec else 1
        if nc > 4:
            return cp.concatenate([self.gradient(phi[:, i:i + 4]) for i in range(0, nc, 4)], 1)
        out = cp.empty(self.N * nc * 3)
        flat = cp.ascontiguousarray(phi.reshape(self.N, nc))
        indptr, indices = self.pattern
        nthreads = self.N * 8
        self._grad_k(((nthreads + 255) // 256,), (256,),
                     (np.int64(self.N), np.int32(nc), indptr, indices, self.gop, flat, out))
        g = out.reshape(self.N, nc, 3)
        return g if vec else g[:, 0]

    def diagonal(self, mdot: dict, U=None, *, mu_t=None):
        """Momentum diagonal ``a_P`` from slot-ordered mass flows (any rheology; ``mu_t``
        (N,) an eddy viscosity at the nodes, added at every flux point)."""
        cp = self.cp
        diag = cp.zeros(self.N)
        U_ = self._vec(U if U is not None else cp.zeros((self.N, 3)))
        mt = self._mut(mu_t)
        for K in self.kinds:
            self._launch(K, "diag", (cp.ascontiguousarray(mdot[K["kind"]]), U_) + self.rheo +
                         (np.int32(self.transpose), np.int32(self.stokes)) + mt + (diag,))
        return diag

    def assemble(self, U, mdot: dict, gradU, beta, gradP, dnode, snode, *, levels=(),
                 own=None, mu_t=None):
        """Block values ``(nnzb, 4, 4)`` and right-hand side ``(N, 4)`` of the ip fluxes.

        ``own`` (N,) flags nodes whose row leaves out the deferred corrections of
        the faces they are upwind of (inflow through a pressure boundary). ``mu_t``
        (N,) is an eddy viscosity at the nodes, added at every flux point.
        """
        cp = self.cp
        indptr, indices = self.pattern
        data = cp.zeros(int(indices.size) * 16)
        b = cp.zeros(self.N * 4)
        U_, gU, be, gP, dn, sn = (self._vec(a) for a in (U, gradU, beta, gradP, dnode, snode))
        own = self._own(own)
        mt = self._mut(mu_t)
        for K in self.kinds:
            self._launch(K, "assemble",
                         (U_, cp.ascontiguousarray(mdot[K["kind"]]), gU, be, gP, dn, sn)
                         + self._levels(levels, K["kind"]) + (np.float64(self.rho),) + self.rheo
                         + (np.int32(self.transpose), np.int32(self.stokes)) + mt
                         + (own, K["pos"], data, b))
        return data.reshape(-1, 4, 4), b.reshape(self.N, 4)

    def _mut(self, mu_t):
        """``(has_mut, mu_t)`` kernel arguments; a one-element dummy when there is none."""
        cp = self.cp
        if mu_t is None:
            if getattr(self, "_no_mut", None) is None:
                self._no_mut = cp.zeros(1)
            return np.int32(0), self._no_mut
        return np.int32(1), cp.ascontiguousarray(cp.asarray(mu_t, dtype=cp.float64))

    def _own(self, own):
        """``own`` as int32 on the device; no node flagged when it is ``None``."""
        cp = self.cp
        if own is None:
            if getattr(self, "_no_own", None) is None:
                self._no_own = cp.zeros(self.N, dtype=cp.int32)
            return self._no_own
        return cp.ascontiguousarray(cp.asarray(own, dtype=cp.int32))

    def massflow(self, U, P, gradP, dnode, snode, *, levels=()):
        """``(mdot {kind: (E, n_ip)} slot order, net outflow (N,), advective diagonal (N,))``."""
        cp = self.cp
        imb = cp.zeros(self.N)
        adv = cp.zeros(self.N)
        out = {}
        U_, P_, gP, dn, sn = (self._vec(a) for a in (U, P, gradP, dnode, snode))
        for K in self.kinds:
            m = cp.empty(K["E"] * K["nip"])
            self._launch(K, "massflow", (U_, P_, gP, dn, sn) + self._levels(levels, K["kind"])
                         + (np.float64(self.rho), m, imb, adv))
            out[K["kind"]] = m.reshape(K["E"], K["nip"])
        return out, imb, adv

    def limiter(self, U, gradU):
        """Barth–Jespersen blend factors ``(N, 3)``."""
        cp = self.cp
        U_ = self._vec(U)
        lo, hi = cp.empty(self.N * 3), cp.empty(self.N * 3)
        indptr, indices = self.pattern
        self._minmax_k(((self.N + 255) // 256,), (256,),
                       (np.int64(self.N), indptr, indices, U_, lo, hi))
        beta = cp.ones(self.N * 3)
        gU = self._vec(gradU)
        for K in self.kinds:
            self._launch(K, "limiter", (U_, gU, lo, hi, beta))
        return beta.reshape(self.N, 3)


__all__ = ["FastAssembler", "gpu_colour", "gpu_node_graph"]
