"""CPU reference solver: element-based finite volumes, coupled u–v–w–p, direct linear solves.

This is the oracle for the GPU kernels and the vehicle for the validation
cases. It is double precision, vectorised with numpy, assembles into
scipy sparse matrices, and solves each linearisation exactly (SuperLU),
so every difference between it and the GPU solver is a GPU-side error.
It is meant for meshes of up to a few tens of thousands of nodes.

Equations
---------
Incompressible, isothermal, constant density ``ρ``, viscosity ``μ``
(or a generalised-Newtonian ``μ(γ̇)``), body force ``f`` per unit volume.
For the control volume ``V_i`` of every node ``i``, with the
integration-point faces of the median dual (:mod:`zvcfd.fv.geometry`),
their outward area vectors ``A`` and boundary sub-faces:

    ρ V_i ∂u/∂t|_i + Σ_ip ṁ_ip u_ip + Σ_ip p_ip A_ip
        − Σ_ip μ (∇u + ∇uᵀ)_ip · A_ip + boundary terms = f_i V_i
    Σ_ip ṁ_ip + boundary mass flows = 0

- ``p_ip = Σ_c N_c(ξ_ip) p_c`` and ``(∇u)_ip = Σ_c u_c ∇N_c(ξ_ip)``: shape
  functions of the element that holds the integration point (element-based).
- **Mass flow** with Rhie–Chow-type pressure redistribution (Rhie & Chow
  1983): ``ṁ_ip = ρ [ū_ip · A + d_ip (∇̄p_ip − ∇p_ip) · A] + transient part``.
  Here ``ū`` and ``∇p`` come from shape functions, ``∇̄p`` interpolates
  nodal pressure gradients, and ``d_ip`` the mean of ``V/(a_P + t)`` at the
  edge's two nodes (``a_P`` the momentum diagonal, ``t`` the time term's). The term vanishes
  for linear pressure fields. Here ``∇̄p`` is **implicit** (a sparse
  operator product, which widens the pressure stencil to neighbours of
  neighbours), so a Stokes problem is one linear solve. The GPU solver lags
  it, as CFX does, and converges to the same discrete solution.
- **Transient Rhie–Chow part.** With a time term, ``ṁ_ip`` also carries
  ``f_ip Σ_l (c_l/c_0) (ṁ_ip^l − ρ ū_ip^l · A)`` over the previous time levels
  ``l``, with ``f_ip = 1 − d_ip / mean(V/a_P)`` the time term's share. Then
  ``d_ip/(1 − f_ip) = mean(V/a_P)`` holds exactly, so a steady state reached
  by time-marching, or with a false time step, does not depend on ``Δt``
  (Choi 1999).
- **Advection**: implicit upwind in the lagged ``ṁ`` (Picard), plus a
  deferred correction to ``φ_up + β ∇φ_up · (x_ip − x_up)``: ``β`` fixed
  (Specified Blend) or from a Barth–Jespersen limiter at the upwind node
  (High Resolution; Barth & Jespersen 1989), frozen after a few iterations.
- **Nodal gradients** average element gradients weighted by
  sub-control-volume, which is exact for linear fields at every node.
- **Time**: a false time step ``dt`` for steady runs (pseudo-transient
  relaxation), or :meth:`ReferenceSolver.solve_transient` with first- or
  second-order backward differences (BDF1, BDF2) and coefficient loops.

Boundary conditions
-------------------
Per zone, from a ``{zone id: spec}`` table. Values may be constants,
``f(x)`` or ``f(x, t)`` (``x`` is ``(M, 3)``).

``{"kind": "wall"}``
    ``u = 0`` at the zone's nodes; no mass flow.
``{"kind": "velocity", "value": ...}``
    ``u`` prescribed at the zone's nodes; the inflow through each sub-face,
    with velocity interpolated on the face to the sub-face's area centroid
    (exact for linear profiles), enters the continuity rows. A moving wall
    is a velocity zone with tangential velocity.
``{"kind": "symmetry"}``
    a plane of symmetry aligned with a coordinate axis: the normal velocity
    component is zero at the zone's nodes, the tangential components are
    solved with no boundary flux (zero shear), and no mass crosses.
``{"kind": "pressure", "value": ..., "grad": optional, "backflow": optional}``
    ``p`` prescribed at the zone's nodes (static pressure) and on its
    sub-faces for the pressure force; momentum ``ṁ_b u_node`` crosses with
    the flow in either direction (``backflow="consistent"``, the default),
    or only where it leaves (``"outflow"``). The
    viscous flux is either zero-normal-gradient (only the transpose part
    ``μ (∇u)ᵀ · A``, from nodal gradients, lagged) or, with ``grad``
    (a callable giving ``du_k/dx_j`` as ``(M, 3, 3)``), the full prescribed
    viscous traction: that is how manufactured solutions cross an outlet.
    The zone's mass flow is the consistent flux, whatever the interior
    leaves at those nodes, so the mass balance is exact.

A node on several zones takes velocity from a wall first, then from an
inlet, and pressure from an outlet. With no pressure zone, one node's
pressure is pinned (``reference_pressure``).

Forces and balances
-------------------
:meth:`ReferenceSolver.zone_forces` gives the force of the fluid on each
zone from **consistent reactions**: the residual of each node's momentum
equation before its boundary condition replaced it. That is the boundary
flux the discrete equations need there, so drag and lift come without
differentiating the solution at the wall. :meth:`ReferenceSolver.balances`
checks the global mass and momentum balances with them.

References: C. M. Rhie, W. L. Chow, AIAA J. 21, 1525 (1983); S. K. Choi,
Numer. Heat Transfer B 36, 545 (1999); T. J. Barth, D. C. Jespersen, AIAA
paper 89-0366 (1989); G. E. Schneider, M. J. Raw, Numer. Heat Transfer 11,
363 (1987).
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from zvcfd.fv.geometry import (
    boundary_subface_weights,
    dual_geometry,
    element_points,
    gradients,
    ip_areas,
    ip_points,
    scv_volumes,
    topology,
)
from zvcfd.mesh.core import UnstructuredMesh

# BDF coefficients: (c0, c1, c2) with du/dt ~ (c0 u^{n+1} - c1 u^n - c2 u^{n-1}) / dt
BDF = {"bdf1": (1.0, 1.0, 0.0), "bdf2": (1.5, 2.0, -0.5)}


@dataclass
class Fluid:
    """Density (kg/m³) and viscosity: a constant ``mu`` (Pa·s), or ``viscosity(gamma_dot)``."""

    rho: float = 1.0
    mu: float = 1.0
    viscosity: object | None = None      # callable: shear rate (1/s) -> mu (Pa s)


@dataclass
class _Kind:
    """Per-element-type precomputed geometry."""

    kind: str
    elem: np.ndarray        # (E, n)
    A: np.ndarray           # (E, ne, 3) ip areas, edge node 0 -> 1
    G: np.ndarray           # (E, ne, n, 3) shape-function gradients at ips
    N: np.ndarray           # (ne, n) shape functions at ips
    xip: np.ndarray         # (E, ne, 3) ip positions
    scv: np.ndarray         # (E, n) sub-control volumes
    gbar: np.ndarray        # (E, n, 3) element-mean gradient operator
    ea: np.ndarray          # (ne,) local node of each ip's side 0
    eb: np.ndarray          # (ne,) local node of each ip's side 1
    mdot: np.ndarray = field(default=None)   # (E, ne) mass flow a -> b
    old: list = field(default_factory=list)  # [(mdot, rho ubar.A)] at previous time levels


@dataclass
class SolveReport:
    iterations: int
    converged: bool
    history: list
    seconds: float


def _call(fn, x, t):
    """Evaluate a boundary or source value: a constant, ``f(x)`` or ``f(x, t)``."""
    if not callable(fn):
        return fn
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n = 1
    return fn(x, t) if n >= 2 else fn(x)


class ReferenceSolver:
    """Steady, pseudo-transient or transient coupled solve on an :class:`UnstructuredMesh`.

    Args:
        mesh: the mesh (coordinates in metres, or any consistent unit).
        fluid: :class:`Fluid`.
        bcs: ``{zone id: {"kind": ..., "value": ...}}``; every zone needs one.
        source: body force per unit volume, ``f(x)`` or ``f(x, t)`` -> ``(N, 3)``.
        advection: ``"upwind"``, ``"high-resolution"``, or a blend factor in [0, 1].
        dt: false time step (s) for steady solves; None for none.
        reference_pressure: ``(node, value)`` to pin when no zone fixes pressure.
        stokes: drop advection altogether (creeping flow).
        freeze_limiter: with High Resolution, keep the limiter's blend factors fixed
            after this many Picard iterations. The Barth–Jespersen limiter is not
            differentiable, and left free it holds Picard in a limit cycle (a
            relative change of ~5 × 10⁻³ on a tetrahedral box); frozen, the
            iteration converges to round-off. None never freezes.
        rhie_chow: scale on the pressure-redistribution coefficient ``d`` (1 = V / a_P).
        transpose: keep the ``μ (∇u)ᵀ`` part of the viscous stress. It integrates to
            zero for constant viscosity; with ``"auto"`` it is kept only for a
            generalised-Newtonian fluid.
        t: initial time (s), for time-dependent values.
        lag_rhie_chow: take the interpolated nodal pressure gradient ``∇̄p`` from
            the previous iterate (explicit), as CFX does and the GPU solver will,
            instead of implicitly. The converged solution is the same; getting
            there takes more iterations.
    """

    def __init__(self, mesh: UnstructuredMesh, fluid: Fluid, bcs: dict, *, source=None,
                 advection="high-resolution", dt: float | None = None,
                 reference_pressure: tuple[int, float] | None = None, stokes: bool = False,
                 rhie_chow: float = 1.0, transpose="auto", freeze_limiter: int | None = 10,
                 t: float = 0.0, lag_rhie_chow: bool = False):
        self.mesh, self.fluid, self.bcs = mesh, fluid, bcs
        self.advection, self.dt, self.stokes = advection, dt, stokes
        self.rhie_chow = rhie_chow
        self.freeze_limiter = freeze_limiter
        self.lag_rhie_chow = lag_rhie_chow
        self.iteration = 0
        self._frozen_beta = None
        self.transpose = (fluid.viscosity is not None) if transpose == "auto" else bool(transpose)
        self.source = source
        self.t = float(t)
        N = mesh.n_nodes
        self.N = N
        self.geom = dual_geometry(mesh, keep_areas=False)
        self.V = self.geom.node_volume
        self.kinds: list[_Kind] = []
        for k, e in mesh.elements.items():
            topo = topology(k)
            x = mesh.nodes[e]
            P = element_points(x, topo)
            G, _ = gradients(x, topo)
            self.kinds.append(_Kind(
                k, e, ip_areas(P, topo), G, topo.N_ip, ip_points(P, topo),
                scv_volumes(P, topo), G.mean(1), topo.edges[:, 0], topo.edges[:, 1],
                np.zeros((len(e), topo.n_ip))))
        self.U = np.zeros((N, 3))
        self.P = np.zeros(N)
        # time stepping: None (steady or false time step), or (scheme, dt, old U levels)
        self._time = None
        self._classify(reference_pressure)
        self._update_boundary(self.t)
        self.U[self.vel_fixed_comp] = self.vel_value[self.vel_fixed_comp]
        self.P[self.p_fixed] = self.p_value[self.p_fixed]

    # ------------------------------------------------------------ boundary sets

    def _classify(self, reference_pressure):
        """Which nodes each kind of condition fixes (once)."""
        N = self.N
        wall = np.zeros(N, bool)
        inlet = np.zeros(N, bool)
        sym = np.zeros((N, 3), bool)
        outlet = np.zeros(N, bool)
        self.sub = {}                                # zone -> (nodes (F,4), sub areas (F,4,3))
        self._zone_nodes = {}
        for zid, z in self.mesh.zones.items():
            if zid not in self.bcs:
                raise ValueError(f"zone {zid} ({z.name!r}) has no boundary condition")
            spec = self.bcs[zid]
            S = self.geom.boundary[zid]
            f = z.faces
            self.sub[zid] = (f, S)
            nodes = np.unique(f[f >= 0])
            self._zone_nodes[zid] = nodes
            kind = spec["kind"]
            if kind == "wall":
                wall[nodes] = True
            elif kind == "velocity":
                inlet[nodes] = True
            elif kind == "pressure":
                outlet[nodes] = True
            elif kind == "symmetry":
                n = S.reshape(-1, 3).sum(0)
                ax = int(np.argmax(np.abs(n)))
                if np.abs(S.reshape(-1, 3)[:, [a for a in range(3) if a != ax]]).max() > \
                        1e-9 * np.abs(S.reshape(-1, 3)[:, ax]).max():
                    raise ValueError(f"zone {zid}: symmetry planes must be axis-aligned")
                sym[nodes, ax] = True
            else:
                raise ValueError(f"zone {zid}: kind {kind!r}")
        self.wall = wall
        self.vel_fixed = wall | inlet                  # nodes with the whole velocity fixed
        self.vel_fixed_comp = self.vel_fixed[:, None] | sym
        self.sym = sym                                 # (N, 3): node on a plane normal to axis
        self.outlet = outlet
        self.p_fixed = outlet.copy()
        self._pin = None
        if not outlet.any():
            node, value = reference_pressure or (int(np.flatnonzero(~self.vel_fixed)[0]
                                                    if (~self.vel_fixed).any() else 0), 0.0)
            self.p_fixed[node] = True
            self._pin = (int(node), value)

    def _update_boundary(self, t: float):
        """Boundary values, boundary mass flows and the body force at time ``t``."""
        N, rho = self.N, self.fluid.rho
        x = self.mesh.nodes
        vel = np.zeros((N, 3))
        pval = np.zeros(N)
        for zid, spec in self.bcs.items():
            if zid not in self._zone_nodes:
                continue
            nodes = self._zone_nodes[zid]
            if spec["kind"] == "velocity":
                v = _call(spec["value"], x[nodes], t)
                vel[nodes] = np.broadcast_to(np.asarray(v, float), (len(nodes), 3))
        for zid, spec in self.bcs.items():
            if zid in self._zone_nodes and spec["kind"] == "pressure":
                nodes = self._zone_nodes[zid]
                pval[nodes] = np.broadcast_to(np.asarray(_call(spec["value"], x[nodes], t),
                                                         float), (len(nodes),))
        vel[self.wall] = 0.0
        if self._pin is not None:
            node, value = self._pin
            pval[node] = (float(np.asarray(_call(value, x[node:node + 1], t)).reshape(-1)[0])
                          if callable(value) else value)
        self.vel_value = vel
        self.p_value = pval
        # known mass outflow through wall and inlet sub-faces: velocity interpolated on the
        # face to each sub-face's area centroid, exact for linear profiles
        self.inflow = np.zeros(N)
        self.sub_flow = {}
        self.sub_vel = {}
        self._outlet = {}
        for zid, (f, S) in self.sub.items():
            ok = f >= 0
            W = boundary_subface_weights(f)
            fz = np.where(ok, f, 0)
            spec = self.bcs[zid]
            if spec["kind"] == "pressure":
                xs = np.einsum("fij,fjk->fik", W, x[fz])              # sub-face centroids
                pts = xs[ok]
                p_sub = np.zeros(ok.shape)
                p_sub[ok] = np.broadcast_to(np.asarray(_call(spec["value"], pts, t), float),
                                            (len(pts),))
                grad = None
                if spec.get("grad") is not None:
                    grad = np.zeros(ok.shape + (3, 3))
                    grad[ok] = _call(spec["grad"], pts, t)
                self._outlet[zid] = (p_sub, grad)
                continue
            usub = np.einsum("fij,fjk->fik", W, vel[fz])
            fl = rho * np.einsum("fik,fik->fi", S, usub) * ok
            self.sub_flow[zid] = fl
            self.sub_vel[zid] = usub
            self.inflow += np.bincount(f[ok], fl[ok], N)
        src = _call(self.source, x, t) if self.source is not None else None
        self.f = np.zeros((N, 3)) if src is None else np.asarray(src, float).reshape(N, 3)

    # ------------------------------------------------------------ gradients

    def nodal_gradient(self, phi: np.ndarray) -> np.ndarray:
        """SCV-weighted element gradients: ``(N, 3)`` for ``(N,)``, ``(N, C, 3)`` for ``(N, C)``."""
        phi = np.asarray(phi)
        vec = phi.ndim == 2
        out = np.zeros((self.N,) + (phi.shape[1:] if vec else ()) + (3,))
        for K in self.kinds:
            g = np.einsum("enk,en...->e...k", K.gbar, phi[K.elem])        # element gradient
            w = K.scv[..., None] if not vec else K.scv[..., None, None]
            contrib = w * g[:, None]
            flat = contrib.reshape(-1, *contrib.shape[2:])
            idx = K.elem.reshape(-1)
            for j in np.ndindex(flat.shape[1:]):
                out[(slice(None),) + j] += np.bincount(idx, flat[(slice(None),) + j], self.N)
        out = out / (self.V[:, None, None] if vec else self.V[:, None])
        return self._mirror(out)

    def _mirror(self, g: np.ndarray) -> np.ndarray:
        """Impose mirror symmetry on nodal gradients at symmetry-plane nodes.

        A scalar, and each velocity component along the plane, has zero normal
        derivative there; the normal velocity component (zero on the plane)
        has zero tangential derivatives.
        """
        sym = getattr(self, "sym", None)
        if sym is None or not sym.any():
            return g
        for ax in range(3):
            on = sym[:, ax]
            if not on.any():
                continue
            if g.ndim == 2:
                g[on, ax] = 0.0
            else:
                idx = np.flatnonzero(on)
                for j in range(3):
                    for k in range(3):
                        if (j == ax) != (k == ax):
                            g[idx, j, k] = 0.0
        return g

    def _node_viscosity(self) -> np.ndarray:
        if self.fluid.viscosity is None:
            return np.full(self.N, self.fluid.mu)
        g = self.nodal_gradient(self.U)
        s = 0.5 * (g + np.swapaxes(g, 1, 2))
        return self.fluid.viscosity(np.sqrt(2.0 * np.einsum("ijk,ijk->i", s, s)))

    def _viscosity(self, K: _Kind) -> np.ndarray:
        """Viscosity at each ip ``(E, ne)``."""
        if self.fluid.viscosity is None:
            return np.full(K.A.shape[:2], self.fluid.mu)
        gu = np.einsum("eink,enj->eijk", K.G, self.U[K.elem])             # du_j/dx_k
        s = 0.5 * (gu + np.swapaxes(gu, 2, 3))
        gamma = np.sqrt(2.0 * np.einsum("eijk,eijk->ei", s, s))
        return self.fluid.viscosity(gamma)

    # ------------------------------------------------------------ advection correction

    def _beta(self, grad: np.ndarray) -> np.ndarray | float:
        """Blend factor per node and component: fixed, or Barth–Jespersen."""
        if self.advection == "upwind":
            return 0.0
        if self.advection != "high-resolution":
            return float(self.advection)
        U = self.U
        lo = U.copy()
        hi = U.copy()
        for K in self.kinds:
            emin, emax = U[K.elem].min(1), U[K.elem].max(1)
            for c in range(K.elem.shape[1]):
                np.minimum.at(lo, K.elem[:, c], emin)
                np.maximum.at(hi, K.elem[:, c], emax)
        beta = np.ones_like(U)
        for K in self.kinds:
            for s in range(len(K.ea)):
                for side in (K.ea[s], K.eb[s]):
                    node = K.elem[:, side]
                    dx = K.xip[:, s] - self.mesh.nodes[node]
                    delta = np.einsum("ejk,ek->ej", grad[node], dx)
                    r = np.where(delta > 1e-300, (hi[node] - U[node]) / np.where(
                        delta > 1e-300, delta, 1.0), np.where(delta < -1e-300, (lo[node] - U[
                            node]) / np.where(delta < -1e-300, delta, 1.0), 1.0))
                    np.minimum.at(beta, node, np.clip(r, 0.0, 1.0))
        return beta

    # ------------------------------------------------------------ time terms

    def _time_terms(self):
        """``(diag (N,), rhs (N, 3), share f (N,))`` of the time derivative, or Nones.

        ``diag`` is the coefficient of ``u^{n+1}`` in each momentum row, ``rhs``
        the known part; the false time step uses the current iterate as ``u^n``.
        """
        rho = self.fluid.rho
        if self._time is not None:
            scheme, dt, old = self._time
            c0, c1, c2 = BDF[scheme] if len(old) > 1 else BDF["bdf1"]
            w = rho * self.V / dt
            rhs = w[:, None] * (c1 * old[0] + (c2 * old[1] if len(old) > 1 and c2 else 0.0))
            return c0 * w, rhs
        if self.dt:
            w = rho * self.V / self.dt
            return w, w[:, None] * self.U
        return None, None

    def _time_coefs(self):
        """``[(level, c_l / c_0)]`` for the transient Rhie–Chow part.

        With a false time step the previous iterate is the old level, so the
        converged steady state does not depend on ``dt``.
        """
        if self._time is None and self.dt:
            return [(0, 1.0)]
        if self._time is not None:
            scheme, dt, old = self._time
            c0, c1, c2 = BDF[scheme] if len(old) > 1 else BDF["bdf1"]
            out = [(0, c1 / c0)]
            if len(old) > 1 and c2:
                out.append((1, c2 / c0))
            return out
        return []

    # ------------------------------------------------------------ assembly

    def assemble(self):
        """The linearised block system ``(A (4N × 4N) csr, b (4N,))`` about the current state.

        The system before boundary rows are replaced is kept as ``self._raw``
        for consistent reactions.
        """
        N, rho = self.N, self.fluid.rho
        gradU = self.nodal_gradient(self.U)                  # (N, 3, 3): du_j/dx_k
        if self.stokes:
            beta = 0.0
        elif self._frozen_beta is not None:
            beta = self._frozen_beta
        else:
            beta = self._beta(gradU)
            if (self.advection == "high-resolution" and self.freeze_limiter is not None
                    and self.iteration >= self.freeze_limiter):
                self._frozen_beta = beta
        blocks, rows_cols = [], []
        rc_rows, rc_cols, rc_w = [], [], []
        b = np.zeros((N, 4))
        diag = np.zeros(N)
        mom = []
        for K in self.kinds:
            E, n = K.elem.shape
            mu = self._viscosity(K)
            Km = np.zeros((E, n, n, 4, 4))
            for s in range(len(K.ea)):
                a, bb = K.ea[s], K.eb[s]
                A = K.A[:, s]                                    # (E, 3)
                G = K.G[:, s]                                    # (E, n, 3)
                R = np.zeros((E, n, 4, 4))
                GA = np.einsum("enk,ek->en", G, A)
                for k in range(3):
                    R[:, :, k, k] -= mu[:, s, None] * GA         # mu grad u_k . A
                    if self.transpose:
                        for j in range(3):                       # mu (du_j/dx_k) A_j
                            R[:, :, k, j] -= mu[:, s, None] * G[:, :, k] * A[:, None, j]
                    R[:, :, k, 3] += K.N[s][None, :] * A[:, None, k]   # p_ip A_k
                if not self.stokes:
                    md = K.mdot[:, s]
                    for k in range(3):
                        R[:, a, k, k] += np.maximum(md, 0.0)
                        R[:, bb, k, k] += np.minimum(md, 0.0)
                Km[:, a] += R
                Km[:, bb] -= R
            mom.append(Km)
            for c in range(n):
                diag += np.bincount(K.elem[:, c], Km[:, c, c, 0, 0], N)
        tdiag, trhs = self._time_terms()
        self._a_space = diag
        self._a_time = tdiag if tdiag is not None else np.zeros(N)
        self.aP = diag + self._a_time
        self._dnode = True
        tcoef = self._time_coefs()
        for K, Km in zip(self.kinds, mom):
            E, n = K.elem.shape
            for s in range(len(K.ea)):
                a, bb = K.ea[s], K.eb[s]
                A = K.A[:, s]
                G = K.G[:, s]
                ia, ib = K.elem[:, a], K.elem[:, bb]
                d, fip = self._rc_coefficients(ia, ib)
                R = np.zeros((E, n, 4, 4))
                for j in range(3):
                    R[:, :, 3, j] = rho * K.N[s][None, :] * A[:, None, j]
                R[:, :, 3, 3] = -rho * d[:, None] * np.einsum("enk,ek->en", G, A)
                # rho d grad-bar(p)_ip . A = sum_c w_c . gradbar(p)_c, w_c = rho d N_c A
                w = rho * d[:, None, None] * K.N[s][None, :, None] * A[:, None, :]
                rc_rows.append(np.concatenate([ia, ib]))
                rc_w.append(np.concatenate([w, -w]))
                rc_cols.append(np.concatenate([K.elem, K.elem]))
                Km[:, a] += R
                Km[:, bb] -= R
                if tcoef and K.old:
                    tr = fip * sum(c * (K.old[lvl][0][:, s] - K.old[lvl][1][:, s])
                                   for lvl, c in tcoef if lvl < len(K.old))
                    np.add.at(b[:, 3], ia, -tr)
                    np.add.at(b[:, 3], ib, tr)
                # deferred correction: explicit high-order part of the advected value
                if not self.stokes and np.any(np.asarray(beta) > 0):
                    md = K.mdot[:, s]
                    up = np.where(md >= 0, ia, ib)
                    dx = K.xip[:, s] - self.mesh.nodes[up]
                    bt = beta[up] if np.ndim(beta) else beta
                    corr = md[:, None] * bt * np.einsum("ejk,ek->ej", gradU[up], dx)
                    for k in range(3):
                        np.add.at(b[:, k], ia, -corr[:, k])
                        np.add.at(b[:, k], ib, corr[:, k])
            ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            r = (4 * K.elem[:, ii][..., None, None] + np.arange(4)[:, None])
            c = (4 * K.elem[:, jj][..., None, None] + np.arange(4)[None, :])
            r, c = np.broadcast_arrays(r, c)
            blocks.append(Km.reshape(-1))
            rows_cols.append((r.reshape(-1), c.reshape(-1)))
        rows = np.concatenate([rc[0] for rc in rows_cols])
        cols = np.concatenate([rc[1] for rc in rows_cols])
        vals = np.concatenate(blocks)
        # body force, known boundary mass flows, time term
        b[:, :3] += self.f * self.V[:, None]
        b[:, 3] -= self.inflow
        extra_r, extra_c, extra_v = [], [], []
        if tdiag is not None:
            for k in range(3):
                extra_r.append(4 * np.arange(N) + k)
                extra_c.append(4 * np.arange(N) + k)
                extra_v.append(tdiag)
            b[:, :3] += trhs
        # pressure outlets: boundary pressure force, viscous traction, outflowing momentum
        self._outlet_terms(b, extra_r, extra_c, extra_v, gradU)
        if extra_r:
            rows = np.concatenate([rows] + extra_r)
            cols = np.concatenate([cols] + extra_c)
            vals = np.concatenate([vals] + extra_v)
        M = sp.csr_matrix((vals, (rows, cols)), shape=(4 * N, 4 * N))
        if self.lag_rhie_chow:
            # explicit: rho d gradbar(p)_ip . A from the current pressure, on the right
            gp = self.nodal_gradient(self.P)
            for rows_, cols_, w_ in zip(rc_rows, rc_cols, rc_w):
                val = np.einsum("enk,enk->e", w_, gp[cols_])
                np.add.at(b[:, 3], rows_, -val)
        else:
            M = M + self._rhie_chow_operator(rc_rows, rc_cols, rc_w)
        self._raw = (M, b.reshape(-1).copy())
        self._tterms = (tdiag, trhs) if self._time is not None else None
        return self._dirichlet(M, b.reshape(-1))

    def _rc_coefficients(self, ia, ib):
        """Rhie–Chow ``d_ip`` and the time share ``f_ip`` at integration points.

        ``d_ip`` is the mean over the edge's nodes of ``V/(a + t)`` (momentum
        and time-term diagonals), and ``f_ip = 1 − d_ip / mean(V/a)``. Then
        ``d_ip/(1 − f_ip) = mean(V/a)`` exactly: the converged steady state does
        not depend on ``Δt``. Being built from per-node ratios, it is also
        unchanged when a symmetry plane halves a control volume.
        """
        V, a, t = self.V, self._a_space, self._a_time
        d_i = V / np.maximum(a + t, 1e-300)
        s_i = V / np.maximum(a, 1e-300)
        d = 0.5 * (d_i[ia] + d_i[ib])
        sp_ = 0.5 * (s_i[ia] + s_i[ib])
        return self.rhie_chow * d, 1.0 - d / np.maximum(sp_, 1e-300)

    def _gradient_operator(self) -> sp.csr_matrix:
        """``(3N × N)``: nodal gradients as a linear map, row ``3 i + k``."""
        if getattr(self, "_Gop", None) is None:
            r, c, v = [], [], []
            for K in self.kinds:
                E, n = K.elem.shape
                w = K.scv / self.V[K.elem]                                   # (E, n)
                for i in range(n):
                    for cc in range(n):
                        for k in range(3):
                            r.append(3 * K.elem[:, i] + k)
                            c.append(K.elem[:, cc])
                            v.append(w[:, i] * K.gbar[:, cc, k])
            G = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
                              shape=(3 * self.N, self.N))
            keep = np.ones(3 * self.N)
            keep[(3 * np.arange(self.N)[:, None] + np.arange(3))[self.sym]] = 0.0
            self._Gop = (sp.diags(keep) @ G).tocsr()            # mirror: no normal gradient
        return self._Gop

    def _rhie_chow_operator(self, rows, cols, ws) -> sp.csr_matrix:
        """The continuity rows' ``+ρ d ∇̄p · A`` terms as a ``(4N × 4N)`` matrix (p columns)."""
        N = self.N
        r = np.concatenate([np.repeat(x[:, None], w.shape[1] * 3, 1).reshape(-1)
                            for x, w in zip(rows, ws)])
        c = np.concatenate([(3 * cc[..., None] + np.arange(3)).reshape(-1)
                            for cc in cols])
        v = np.concatenate([w.reshape(-1) for w in ws])
        R = sp.csr_matrix((v, (r, c)), shape=(N, 3 * N))
        C = (R @ self._gradient_operator()).tocoo()
        return sp.csr_matrix((C.data, (4 * C.row + 3, 4 * C.col + 3)), shape=(4 * N, 4 * N))

    def _outlet_boundary_flux(self, zid, gradU):
        """Force of the fluid on an outlet's sub-faces ``(F, 4, 3)``: pressure minus viscous."""
        f, S = self.sub[zid]
        ok = f >= 0
        p_sub, grad = self._outlet[zid]
        force = p_sub[..., None] * S
        if grad is not None:
            if self.fluid.viscosity is None:
                mu = np.full(ok.shape, self.fluid.mu)
            else:                                  # viscosity of the prescribed gradient
                S2 = 0.5 * (grad + np.swapaxes(grad, -1, -2))
                mu = self.fluid.viscosity(np.sqrt(2 * np.einsum("fikj,fikj->fi", S2, S2)))
            tau = grad + np.swapaxes(grad, -1, -2) if self.transpose else grad
            visc = mu[..., None] * np.einsum("fikj,fij->fik", tau, S)
        elif self.transpose:
            mu = self._node_viscosity()[np.where(ok, f, 0)]
            visc = mu[..., None] * np.einsum("fijk,fij->fik", gradU[np.where(ok, f, 0)], S)
        else:
            visc = np.zeros_like(force)
        return (force - visc) * ok[..., None]

    def _outlet_terms(self, b, er, ec, ev, gradU):
        for zid, (f, S) in self.sub.items():
            if self.bcs[zid]["kind"] != "pressure":
                continue
            ok = f >= 0
            nodes = f[ok]
            flux = self._outlet_boundary_flux(zid, gradU)[ok]
            for k in range(3):
                np.add.at(b[:, k], nodes, -flux[:, k])
            if not self.stokes:
                # outflow momentum at the node: m_b u_node, m_b from the consistent flux
                share = np.linalg.norm(S[ok], axis=1)
                tot = np.bincount(nodes, share, self.N)
                mb = getattr(self, "_mb", np.zeros(self.N))
                flow = mb[nodes] if self.bcs[zid].get("backflow", "consistent") == "consistent" \
                    else np.maximum(mb[nodes], 0.0)
                w = flow * np.divide(share, tot[nodes], out=np.zeros_like(share),
                                     where=tot[nodes] > 0)
                acc = np.bincount(nodes, w, self.N)
                for k in range(3):
                    er.append(4 * np.arange(self.N) + k)
                    ec.append(4 * np.arange(self.N) + k)
                    ev.append(acc)

    def _dirichlet(self, M: sp.csr_matrix, b: np.ndarray):
        fixed = np.zeros(4 * self.N, bool)
        val = np.zeros(4 * self.N)
        for k in range(3):
            fixed[4 * np.flatnonzero(self.vel_fixed_comp[:, k]) + k] = True
            val[4 * np.arange(self.N) + k] = self.vel_value[:, k]
        fixed[4 * np.flatnonzero(self.p_fixed) + 3] = True
        val[4 * np.arange(self.N) + 3] = self.p_value
        keep = sp.diags((~fixed).astype(float))
        M = keep @ M + sp.diags(fixed.astype(float))
        b = np.where(fixed, val, b)
        return M.tocsr(), b

    # ------------------------------------------------------------ mass flows

    def _ubarA(self, K: _Kind, s: int) -> np.ndarray:
        return self.fluid.rho * np.einsum("ek,ek->e", np.einsum("n,enk->ek", K.N[s],
                                                                self.U[K.elem]), K.A[:, s])

    def update_mass_flows(self) -> None:
        """Rhie–Chow ip mass flows from the current state (the next Picard linearisation)."""
        rho = self.fluid.rho
        gradP = self.nodal_gradient(self.P)
        dnode = getattr(self, "_dnode", None)
        if dnode is None:
            return
        tcoef = self._time_coefs()
        imbalance = np.zeros(self.N)
        for K in self.kinds:
            for s in range(len(K.ea)):
                a, bb = K.ea[s], K.eb[s]
                ia, ib = K.elem[:, a], K.elem[:, bb]
                A = K.A[:, s]
                gp_bar = np.einsum("n,enk->ek", K.N[s], gradP[K.elem])
                gp = np.einsum("enk,en->ek", K.G[:, s], self.P[K.elem])
                d, fip = self._rc_coefficients(ia, ib)
                K.mdot[:, s] = self._ubarA(K, s) + rho * d * np.einsum("ek,ek->e",
                                                                       gp_bar - gp, A)
                if tcoef and K.old:
                    K.mdot[:, s] += fip * sum(c * (K.old[lvl][0][:, s] - K.old[lvl][1][:, s])
                                              for lvl, c in tcoef if lvl < len(K.old))
                imbalance += np.bincount(ia, K.mdot[:, s], self.N)
                imbalance -= np.bincount(ib, K.mdot[:, s], self.N)
        # consistent outflow at pressure-fixed nodes: what the interior sends there
        self._interior_out = imbalance
        self._mb = np.where(self.outlet, -(imbalance + self.inflow), 0.0)

    # ------------------------------------------------------------ solve

    def _iterate(self, max_iterations, tol, log, hist, *, reuse_lu):
        lu = None
        converged = False
        it = 0
        for it in range(1, max_iterations + 1):
            self.iteration = it - 1
            if self._time is None and self.dt:
                self._store_old(first=True)        # false time step: old level = last iterate
            M, b = self.assemble()
            if reuse_lu:
                lu = lu or spl.splu(M.tocsc())
                x = lu.solve(b)
            else:
                x = spl.splu(M.tocsc()).solve(b)
            X = x.reshape(-1, 4)
            du = np.abs(X[:, :3] - self.U).max()
            dp = np.abs(X[:, 3] - self.P).max()
            scale_u = max(np.abs(X[:, :3]).max(), 1e-300)
            scale_p = max(np.ptp(X[:, 3]), 1e-300)
            self.U, self.P = X[:, :3].copy(), X[:, 3].copy()
            self.update_mass_flows()
            rec = {"iteration": it, "du": float(du / scale_u), "dp": float(dp / scale_p)}
            hist.append(rec)
            if log:
                log(f"  it {it:3d}  du {rec['du']:.2e}  dp {rec['dp']:.2e}")
            if it > 1 and rec["du"] < tol and rec["dp"] < tol:
                converged = True
                break
        return it, converged

    def solve(self, *, max_iterations: int = 50, tol: float = 1e-10, log=None) -> SolveReport:
        """Steady solve: Picard iterations, each an exact linear solve."""
        t0 = time.time()
        hist = []
        # the matrix is fixed for Newtonian Stokes flow without a false time step: factorise once
        reuse = self.stokes and self.fluid.viscosity is None and not self.dt \
            and not self.lag_rhie_chow
        it, converged = self._iterate(max_iterations, tol, log, hist, reuse_lu=reuse)
        return SolveReport(it, converged, hist, time.time() - t0)

    def initialise(self, U=None, P=None) -> None:
        """Start from given fields (``(N, 3)``, ``(N,)`` or callables of ``x``), with
        boundary values imposed and mass flows consistent with them."""
        x = self.mesh.nodes
        if U is not None:
            self.U = np.array(_call(U, x, self.t), float).reshape(self.N, 3)
        if P is not None:
            self.P = np.array(_call(P, x, self.t), float).reshape(self.N)
        self.U[self.vel_fixed_comp] = self.vel_value[self.vel_fixed_comp]
        self.P[self.p_fixed] = self.p_value[self.p_fixed]
        self.assemble()
        self.update_mass_flows()

    def solve_transient(self, dt: float, steps: int, *, scheme: str = "bdf2",
                        loops: int = 5, tol: float = 1e-10, callback=None,
                        log=None) -> SolveReport:
        """March ``steps`` time steps of ``dt`` with BDF1 or BDF2 and coefficient loops.

        Each step runs Picard coefficient loops (at most ``loops``) until the
        change falls below ``tol``. The first BDF2 step is BDF1.
        ``callback(solver)`` runs after each step.
        """
        if scheme not in BDF:
            raise ValueError(f"scheme {scheme!r}; expected one of {sorted(BDF)}")
        t0 = time.time()
        hist = []
        old = [self.U.copy()]
        self._store_old(first=True)
        total = 0
        for _ in range(steps):
            self.t += dt
            self._time = (scheme, dt, old)
            self._update_boundary(self.t)
            self.U[self.vel_fixed_comp] = self.vel_value[self.vel_fixed_comp]
            self.P[self.p_fixed] = self.p_value[self.p_fixed]
            n, _ = self._iterate(loops, tol, log, hist, reuse_lu=False)
            total += n
            old = [self.U.copy()] + old[:1]
            self._store_old()
            if callback is not None:
                callback(self)
        self._time = None
        return SolveReport(total, True, hist, time.time() - t0)

    def _store_old(self, first: bool = False) -> None:
        """Push the current ip mass flows and ``ρ ū · A`` onto the time levels."""
        if first and getattr(self, "_dnode", None) is None:
            for K in self.kinds:
                K.old = []
        for K in self.kinds:
            lvl = (K.mdot.copy(), np.stack([self._ubarA(K, s) for s in range(len(K.ea))], 1))
            K.old = ([lvl] + K.old)[:2] if not first else [lvl]

    # ------------------------------------------------------------ results

    def zone_flows(self) -> dict[int, float]:
        """Mass outflow (kg/s) through each zone; inflow is negative. Sums to zero."""
        out = {}
        mb = getattr(self, "_mb", np.zeros(self.N))
        for zid, (f, S) in self.sub.items():
            ok = f >= 0
            if self.bcs[zid]["kind"] == "pressure":
                share = np.linalg.norm(S, axis=2) * ok
                nodes = f[ok]
                tot = np.bincount(nodes, share[ok], self.N)
                w = share[ok] / tot[nodes]
                out[zid] = float(np.sum(mb[nodes] * w))
            else:
                out[zid] = float(self.sub_flow[zid].sum())
        return out

    def momentum_residual(self) -> np.ndarray:
        """``(N, 3)``: each node's momentum equation before boundary rows replaced it.

        Zero where momentum is solved (at convergence); at nodes with fixed
        velocity it is minus the boundary flux the discrete equations need.
        """
        M, b = self._raw
        x = np.concatenate([self.U, self.P[:, None]], 1).reshape(-1)
        return (M @ x - b).reshape(-1, 4)[:, :3]

    def _node_zone_shares(self):
        """``{zone: (N, 3)}``: each zone's share of each node's reaction, per component.

        A reaction component belongs to the zones that fix that component
        (walls and inlets fix all three, a symmetry plane only its normal),
        shared by sub-face area. Pressure zones are left out: their boundary
        flux is already in the assembled rows.
        """
        per = {}
        tot = np.zeros((self.N, 3))
        for zid, (f, S) in self.sub.items():
            kind = self.bcs[zid]["kind"]
            if kind == "pressure":
                continue
            ok = f >= 0
            a = np.bincount(f[ok], (np.linalg.norm(S, axis=2) * ok)[ok], self.N)
            comp = np.ones(3, bool)
            if kind == "symmetry":
                comp = np.abs(S.reshape(-1, 3).sum(0)) == np.abs(S.reshape(-1, 3).sum(0)).max()
            per[zid] = a[:, None] * comp[None, :]
            tot += per[zid]
        return {z: np.divide(v, tot, out=np.zeros_like(v), where=tot > 0) for z, v in per.items()}

    def zone_momentum_outflow(self) -> dict[int, np.ndarray]:
        """Momentum leaving through each zone (N), ``Σ ṁ u``."""
        out = {}
        mb = getattr(self, "_mb", np.zeros(self.N))
        for zid, (f, S) in self.sub.items():
            ok = f >= 0
            if self.bcs[zid]["kind"] == "pressure":
                share = np.linalg.norm(S, axis=2) * ok
                nodes = f[ok]
                tot = np.bincount(nodes, share[ok], self.N)
                w = share[ok] / tot[nodes]
                flow = mb[nodes] if self.bcs[zid].get("backflow", "consistent") == \
                    "consistent" else np.maximum(mb[nodes], 0.0)
                out[zid] = (flow * w)[:, None] * self.U[nodes] \
                    if not self.stokes else np.zeros((len(nodes), 3))
                out[zid] = out[zid].sum(0)
            elif self.stokes:
                out[zid] = np.zeros(3)
            else:
                out[zid] = np.einsum("fi,fik->k", self.sub_flow[zid], self.sub_vel[zid])
        return out

    def zone_forces(self) -> dict[int, np.ndarray]:
        """Force of the fluid on each zone (N): pressure and viscous, from consistent reactions.

        At nodes with fixed velocity the boundary flux is minus the momentum
        residual, less the momentum carried across (inlets), shared among the
        node's zones by sub-face area. At pressure outlets it is the imposed
        pressure force and traction.
        """
        r = self.momentum_residual()
        shares = self._node_zone_shares()
        gradU = self.nodal_gradient(self.U)
        conv = self.zone_momentum_outflow()
        fixed = self.vel_fixed_comp
        out = {}
        for zid, (f, S) in self.sub.items():
            kind = self.bcs[zid]["kind"]
            if kind == "pressure":
                out[zid] = self._outlet_boundary_flux(zid, gradU).reshape(-1, 3).sum(0)
                continue
            react = np.where(fixed, -r, 0.0)
            out[zid] = (shares[zid] * react).sum(0) - conv[zid]
        return out

    def balances(self) -> dict:
        """Global mass and momentum balances, from zone flows and consistent forces.

        ``mass``: net outflow over gross flow. ``momentum``: Σ_zones (force +
        momentum outflow) − ∫ f dV + (momentum change), over the scale of the
        forces: zero at convergence, to the iteration tolerance.
        ``identity``: the same less the residual left in the solved rows,
        zero to round-off whatever the convergence (a bookkeeping check).
        """
        q = self.zone_flows()
        scale_q = max(sum(abs(v) for v in q.values()), 1e-300)
        F = self.zone_forces()
        Mo = self.zone_momentum_outflow()
        body = (self.f * self.V[:, None]).sum(0)
        solved = ~self.vel_fixed_comp
        dmdt = np.zeros(3)
        if getattr(self, "_tterms", None) is not None:
            tdiag, trhs = self._tterms
            dmdt = np.where(solved, tdiag[:, None] * self.U - trhs, 0.0).sum(0)
        total = sum(F[z] + Mo[z] for z in F) - body + dmdt
        leftover = np.where(solved, self.momentum_residual(), 0.0).sum(0)
        scale_f = max(sum(np.abs(F[z]).sum() for z in F) + np.abs(body).sum(), 1e-300)
        return {"mass": float(sum(q.values()) / scale_q),
                "momentum": total / scale_f,
                "identity": (total - leftover) / scale_f}


__all__ = ["BDF", "Fluid", "ReferenceSolver", "SolveReport"]
