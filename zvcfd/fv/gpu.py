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


def colour_elements(elem: np.ndarray, n_nodes: int) -> list[np.ndarray]:
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


class GPUAssembler:
    """Device kernels for one mesh. Needs cupy and a CUDA device.

    Args:
        mesh: the mesh.
        rho, mu: density and (constant) viscosity.
        transpose: include the ``μ (∇u)ᵀ`` stress term.
        stokes: no advection.
        block: threads per block.
    """

    def __init__(self, mesh: UnstructuredMesh, *, rho: float = 1.0, mu: float = 1.0,
                 transpose: bool = False, stokes: bool = False, block: int = 128):
        import cupy as cp

        self.cp = cp
        self.mesh, self.rho, self.mu = mesh, float(rho), float(mu)
        self.transpose, self.stokes, self.block = int(transpose), int(stokes), block
        self.N = mesh.n_nodes
        self.X = cp.asarray(mesh.nodes, dtype=cp.float64)
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
                "kernels": {}})

    # ------------------------------------------------------------ compilation

    def _kernel(self, K, name, nc=1):
        key = (name, nc)
        if key not in K["kernels"]:
            t = K["topo"]
            subs = {"n": t.n, "nip": t.n_ip, "npts": t.point_weights.shape[0], "nc": nc,
                    "N": self.N}
            src = (_cuda.COMMON + getattr(_cuda, name.upper())) % subs
            K["kernels"][key] = self.cp.RawKernel(src, name, options=("-std=c++14",))
        return K["kernels"][key]

    def _launch(self, K, kern, args):
        cp = self.cp
        for lst in K["colours"]:
            n = int(lst.size)
            grid = (n + self.block - 1) // self.block
            kern((grid,), (self.block,), (K["elem"], lst, np.int64(n), self.X, K["W"],
                                          K["tris"], K["dN"], K["edges"]) + tuple(args))
        cp.cuda.Device().synchronize()

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

    def diagonal(self, mdot: dict):
        """Momentum diagonal ``a_P`` (N,), with ``mdot = {kind: (E, n_ip)}`` lagged mass flows."""
        cp = self.cp
        diag = cp.zeros(self.N)
        for K in self.kinds:
            m = cp.asarray(mdot[K["kind"]], dtype=cp.float64)
            self._launch(K, self._kernel(K, "diagonal"),
                         (cp.ascontiguousarray(m), np.float64(self.mu), np.int32(self.transpose),
                          np.int32(self.stokes), diag))
        return diag

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
                         (K["N"], U_, P_, m, gU, be, gP, dn, np.float64(self.rho),
                          np.float64(self.mu), np.int32(self.transpose), np.int32(self.stokes),
                          R))
        return R.reshape(self.N, 4)


__all__ = ["GPUAssembler", "colour_elements"]
