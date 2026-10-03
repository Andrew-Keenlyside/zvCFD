"""RANS turbulence for the finite-volume solver: the k-kL model (k-kL-MEAH2015m).

The k-kL model transports the turbulent kinetic energy ``k`` and ``Φ = kL``,
``L`` a turbulent length scale. It is Rotta's two-equation model as
Menter rewrote it (second instead of third velocity derivatives, through
the von Kármán length scale), closed by Abdol-Hamid; NASA's Turbulence
Modeling Resource defines it as k-kL-MEAH2015 (Abdol-Hamid, Carlson &
Rumsey, NASA/TM-2015-218968). For incompressible flow the Resource defines
the variant used here, **k-kL-MEAH2015m**: production ``P = μ_t S²`` and
no ``⅔ρk`` in the stress (it is absorbed in the pressure)::

    ∂(ρk)/∂t + ∇·(ρuk) = P_k − C_μ^¾ ρ k^{5/2}/Φ − 2μ k/d² + ∇·[(μ + σ_k μ_t)∇k]
    ∂(ρΦ)/∂t + ∇·(ρuΦ) = C_φ1 (Φ/k) P_k − C_φ2 ρ k^{3/2} − 6μ (Φ/d²) f_φ
                          + ∇·[(μ + σ_φ μ_t)∇Φ]

    μ_t = C_μ^¼ ρ Φ / √k,   P = μ_t S²,   S = √(2 S_ij S_ij),
    P_k = min(P, 20 C_μ^¾ ρ k^{5/2}/Φ),
    C_φ1 = ζ1 − ζ2 (Φ/(k L_vK))²,  C_φ2 = ζ3,
    f_φ = (1 + C_d1 ξ)/(1 + ξ⁴),  ξ = ρ d √(0.3k)/(20μ),
    L_vK = κ S/|∇²u|,  Φ/(k C11) ≤ L_vK ≤ C12 κ d f_p,
    f_p = min(max(P_k Φ/(C_μ^¾ ρ k^{5/2}), 0.5), 1),

with σ_k = σ_φ = 1, κ = 0.41, C_μ = 0.09, ζ1 = 1.2, ζ2 = 0.97, ζ3 = 0.13,
C11 = 10, C12 = 1.3, C_d1 = 4.7; ``d`` is the distance to the nearest wall.
At walls ``k = Φ = 0`` (and ``μ_t = 0``). The Resource's farfield values are
``k∞ = 9 × 10⁻⁹ a∞²`` and ``Φ∞ = 1.5589 × 10⁻⁶ μ∞ a∞/ρ∞`` with ``a∞`` the
speed of sound; for an incompressible run ``a∞ = U∞/M∞`` at the case's
nominal Mach number (:func:`tmr_freestream`).

**Discretisation.** The two equations are solved after each outer iteration
of the coupled velocity–pressure solve, ``k`` and then ``Φ`` with the new
``k`` (loosely coupled, as CFL3D and FUN3D do), on the same element-based
control volumes. Advection is first-order upwind with the solver's own mass
flows, in the bounded form ``∇·(ρuφ) − φ∇·(ρu)``. Diffusion is edge-based:
through each sub-face, a two-point flux along its element edge (implicit),
plus the shape-function remainder, which vanishes where the sub-face is
normal to the edge (lagged). The full shape-function diffusion is not
monotone on stretched elements. Sources are evaluated at the nodes. The
sinks are linearised by Newton's method and the positive parts kept
explicit, so that the matrices stay M-matrices and ``k`` and ``Φ`` positive.
The sources are under-relaxed: a pseudo-time step of a few turbulent time
scales. ``μ_t`` then goes to the momentum equations through
``GPUSolver.mu_t``. Steady problems converge when the relative change of
``k``, ``Φ`` and ``μ_t`` also falls below the tolerance; a false time step
marches the turbulence with the flow, and transient runs use backward Euler
for the turbulence equations. The scalar systems are solved by FGMRES with
an AmgX V-cycle (``docs/spec/turbulence.md``).

Boundary values: walls ``k = Φ = 0``; velocity inlets the inflow values;
pressure zones carry the inflow values in where flow enters and the node's
own values out where it leaves (zero gradient); symmetry planes have no flux.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zvcfd.fv.geometry import element_points, gradients, ip_areas, topology


@dataclass
class KkLConstants:
    sigma_k: float = 1.0
    sigma_phi: float = 1.0
    kappa: float = 0.41
    c_mu: float = 0.09
    zeta1: float = 1.2
    zeta2: float = 0.97
    zeta3: float = 0.13
    c11: float = 10.0
    c12: float = 1.3
    cd1: float = 4.7


def tmr_freestream(rho: float, mu: float, speed: float, mach: float) -> tuple[float, float]:
    """The Resource's farfield ``(k∞, Φ∞)`` for a flow of ``speed`` at nominal Mach ``mach``."""
    a = speed / mach
    return 9.0e-9 * a * a, 1.5589e-6 * mu * a / rho


def wall_distance(nodes: np.ndarray, tri: np.ndarray, *, candidates: int = 12,
                  batch: int = 200_000) -> np.ndarray:
    """Distance from every node to the nearest wall triangle ``tri (T, 3, 3)`` (exact)."""
    from scipy.spatial import cKDTree

    cen = tri.mean(1)
    tree = cKDTree(cen)
    k = min(candidates, len(tri))
    out = np.empty(len(nodes))
    for s in range(0, len(nodes), batch):
        p = nodes[s:s + batch]
        _, idx = tree.query(p, k=k)
        idx = idx.reshape(len(p), k)
        d = _point_triangle(p[:, None, :], tri[idx])
        out[s:s + batch] = d.min(1)
    return out


def _point_triangle(p, t):
    """Distance from points ``p (..., 3)`` to triangles ``t (..., 3, 3)`` (Ericson's method)."""
    a, b, c = t[..., 0, :], t[..., 1, :], t[..., 2, :]
    ab, ac, ap = b - a, c - a, p - a

    def dot(u, v):
        return (u * v).sum(-1)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp, cp_ = p - b, p - c
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp_), dot(ac, cp_)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = np.where(va + vb + vc == 0, 1.0, va + vb + vc)
    v, w = vb / den, vc / den
    q = a + v[..., None] * ab + w[..., None] * ac                    # interior projection
    # vertex and edge regions
    with np.errstate(divide="ignore", invalid="ignore"):
        tab = np.clip(np.where(d1 - d3 != 0, d1 / (d1 - d3), 0.0), 0, 1)
        tac = np.clip(np.where(d2 - d6 != 0, d2 / (d2 - d6), 0.0), 0, 1)
        tbc = np.clip(np.where((d4 - d3) + (d5 - d6) != 0, (d4 - d3) / ((d4 - d3) + (d5 - d6)), 0),
                      0, 1)
    e_ab = a + tab[..., None] * ab
    e_ac = a + tac[..., None] * ac
    e_bc = b + tbc[..., None] * (c - b)
    cand = [q, a, b, c, e_ab, e_ac, e_bc]
    inside = (va >= 0) & (vb >= 0) & (vc >= 0)
    dist = np.stack([np.linalg.norm(p - x, axis=-1) for x in cand[1:]], 0).min(0)
    return np.where(inside, np.linalg.norm(p - q, axis=-1), dist)


class KkLModel:
    """k-kL-MEAH2015m attached to a :class:`zvcfd.fv.solver.GPUSolver` (``solver.turbulence``).

    Args:
        solver: the GPU solver (fast kernels, one partition).
        k_inf, phi_inf: freestream ``k`` (m²/s²) and ``Φ = kL`` (m³/s²): the
            initial field and the value carried in at inlets and inflowing
            pressure-zone nodes (:func:`tmr_freestream`).
        relax: under-relaxation of the sources of ``k`` and ``Φ``: a pseudo-time step of
            about ``relax/(1 - relax)`` turbulent time scales.
        relax_mu_t: under-relaxation of the ``μ_t`` passed to the flow.
        rtol: relative tolerance of each scalar linear solve.
        inflow: optional ``f(x (n, 3)) -> (k (n,), Φ (n,))``: the values at velocity-inlet
            nodes (an inlet profile); otherwise ``k_inf`` and ``phi_inf``.
    """

    name = "k-kL-MEAH2015m"

    def __init__(self, solver, k_inf: float, phi_inf: float, *, relax: float = 0.7,
                 relax_mu_t: float = 0.7,
                 rtol: float = 1e-6, constants: KkLConstants | None = None, inflow=None,
                 log=None):
        import cupy as cp
        import cupyx.scipy.sparse as csp

        self.cp, self.csp = cp, csp
        self.s = solver
        self.c = constants or KkLConstants()
        self.k_inf, self.phi_inf = float(k_inf), float(phi_inf)
        self.relax, self.relax_mu_t, self.rtol, self.log = relax, relax_mu_t, rtol, log
        mesh, N = solver.mesh, solver.N
        self.N = N
        self.rho = solver.fluid.rho
        # walls, inlets, pressure zones
        wall = np.zeros(N, bool)
        inlet = np.zeros(N, bool)
        tris = []
        for zid, (f, S) in solver.sub.items():
            kind = solver.bcs[zid]["kind"]
            nodes = np.unique(f[f >= 0])
            if kind == "wall":
                wall[nodes] = True
                z = mesh.zones[zid]
                tris.append(mesh.nodes[z.triangles()])
            elif kind == "velocity":
                inlet[nodes] = True
        inlet &= ~wall
        if not tris:
            raise ValueError("the k-kL model needs at least one wall zone")
        self.d = cp.asarray(np.maximum(wall_distance(mesh.nodes, np.concatenate(tris)), 0.0))
        self.wall_d = cp.asarray(wall)
        self.fixed = cp.asarray(wall | inlet)
        fk, fp = np.full(N, self.k_inf), np.full(N, self.phi_inf)
        if inflow is not None and inlet.any():
            vk, vp = inflow(mesh.nodes[inlet])
            fk[inlet], fp[inlet] = vk, vp
        fk[wall] = fp[wall] = 0.0
        self.fixed_k, self.fixed_phi = cp.asarray(fk), cp.asarray(fp)
        # element matrices' ingredients, slot order, and their CSR positions
        indptr, indices = (cp.asarray(a) for a in solver.pattern)
        self.indptr, self.indices = indptr.astype(cp.int64), indices.astype(cp.int64)
        rows = cp.asarray(np.repeat(np.arange(N, dtype=np.int64),
                                    np.diff(cp.asnumpy(self.indptr))))
        self._keys = rows * N + self.indices                       # sorted (CSR order)
        self._rows = rows
        self.diag_pos = cp.searchsorted(self._keys, cp.arange(N, dtype=cp.int64) * (N + 1))
        self.parts = []
        for K in solver.asm.kinds:
            topo = topology(K["kind"]) if "topo" not in K else K["topo"]
            n, E = topo.n, int(K["E"])
            conn = cp.asnumpy(K["conn"]).reshape(E, n).astype(np.int64)
            x = mesh.nodes[conn]
            P = element_points(x, topo)
            A = ip_areas(P, topo)                                  # (E, nip, 3), node 0 -> 1
            G, _ = gradients(x, topo)                              # (E, nip, n, 3)
            # diffusion through sub-face s on edge a -> b: grad(phi) . A_s split into the
            # two-point part c_s (phi_b - phi_a), c_s = max(A_s . e, 0)/|e|^2 (implicit), and
            # the remainder grad(phi) . (A_s - c_s e) from the shape functions (lagged), which
            # vanishes where the sub-face is normal to its edge
            e = x[:, topo.edges[:, 1]] - x[:, topo.edges[:, 0]]    # (E, nip, 3)
            c2 = np.maximum((A * e).sum(-1), 0.0) / (e * e).sum(-1)
            DT = np.einsum("eskd,esd->esk", G, A - c2[..., None] * e)
            inc = np.zeros((topo.n_ip, n))
            inc[np.arange(topo.n_ip), topo.edges[:, 0]] = 1.0
            inc[np.arange(topo.n_ip), topo.edges[:, 1]] = -1.0
            rr = np.repeat(conn[:, :, None], n, 2)
            cc = np.repeat(conn[:, None, :], n, 1)
            pos = cp.searchsorted(self._keys, cp.asarray(rr * N + cc))
            self.parts.append({
                "kind": K["kind"], "E": E, "n": n, "conn": cp.asarray(conn), "c2": cp.asarray(c2),
                "DT": cp.asarray(DT),
                "inc": cp.asarray(inc), "Nip": cp.asarray(topo.N_ip),
                "ea": cp.asarray(topo.edges[:, 0]), "eb": cp.asarray(topo.edges[:, 1]),
                "pos": pos})
        self.V = cp.asarray(solver.Vd)
        # pressure-zone nodes: the lagged boundary mass flow carries values in or out
        pz = np.zeros(N, bool)
        for zid, (f, S) in solver.sub.items():
            if solver.bcs[zid]["kind"] == "pressure":
                pz[f[f >= 0]] = True
        self.pz = cp.asarray(pz & ~(wall | inlet))
        # state
        self.k = cp.where(self.wall_d, 0.0, self.k_inf)
        self.phi = cp.where(self.wall_d, 0.0, self.phi_inf)
        self.k_floor, self.phi_floor = 1e-6 * self.k_inf, 1e-6 * self.phi_inf
        self.linear_failures = 0
        self._amg = None                                           # AmgX, built on first use
        self.mu_t = self._eddy_viscosity()
        self._t = None
        self._transient = False
        self._old = None
        self.history = []

    # ------------------------------------------------------------ the model

    def _eddy_viscosity(self):
        cp = self.cp
        k = cp.maximum(self.k, self.k_floor)
        mut = self.c.c_mu ** 0.25 * self.rho * self.phi / cp.sqrt(k)
        return cp.where(self.wall_d, 0.0, cp.maximum(mut, 0.0))

    def _laminar_viscosity(self, gradU):
        s = self.s
        if s.fluid.viscosity is None:
            return self.cp.full(self.N, s.fluid.mu)
        return s._node_viscosity_d(gradU)

    def attach(self):
        """Give the solver this model and the initial eddy viscosity."""
        self.s.turbulence = self
        self.s.mu_t = self.mu_t
        return self

    def update(self, solver) -> float:
        """One update of ``k`` and ``Φ`` from the current velocity; returns the relative change."""
        cp, c = self.cp, self.c
        U = solver.U
        if solver.dt is not None:
            if solver.t != self._t:                                # a new physical time step
                self._transient = self._t is not None
                self._old = (self.k.copy(), self.phi.copy())
                self._t = solver.t
            if not self._transient:
                # steady solve with a false time step: march k and kL with the same pseudo-time
                # step as the flow, from the last iterate (as CFX does); without it the
                # turbulence jumps to the steady state of a far-from-converged velocity field
                self._old = (self.k.copy(), self.phi.copy())
        gU = solver.asm.gradient(U)                                # (N, 3, 3): du_i/dx_j
        S = 0.5 * (gU + cp.swapaxes(gU, 1, 2))
        S2 = 2.0 * cp.einsum("nij,nij->n", S, S)                   # S² = 2 S:S
        H = solver.asm.gradient(gU.reshape(self.N, 9)).reshape(self.N, 3, 3, 3)
        lap = cp.einsum("nijj->ni", H)                             # ∇²u_i
        U2 = cp.sqrt(cp.einsum("ni,ni->n", lap, lap))
        mu = self._laminar_viscosity(gU)
        # wall nodes are Dirichlet rows: give them a harmless distance so that no term there
        # is infinite (an inf times 0 would leave NaN in the matrix before the rows are reset)
        d = cp.where(self.wall_d, 1.0, cp.maximum(self.d, 1e-300))
        k0, p0 = self.k, self.phi
        # k first, then Φ with the new k (Gauss–Seidel): each equation settles within an update,
        # and Φ solved against the old k (Jacobi) can lock the two into an oscillation
        t = self._sources(cp.maximum(k0, self.k_floor), cp.maximum(p0, self.phi_floor),
                          S2, U2, mu, d)
        self.last = t["last"]
        k_new = self._solve(k0, mu + c.sigma_k * t["mut"], t["ap_k"], t["b_k"], self.fixed_k,
                            self.k_inf, 0 if self._old is None else self._old[0])
        # positivity: no node's k or kL may fall below a tenth of its previous value in one
        # update (an undershoot of the linear solve, e.g. near a leading-edge singularity,
        # would otherwise send k to the floor and mu_t = C kL / sqrt(k) off to infinity);
        # a converged steady solution is unaffected
        self.k = cp.maximum(k_new, cp.where(self.wall_d, 0.0, cp.maximum(self.k_floor, 0.1 * k0)))
        t = self._sources(cp.maximum(self.k, self.k_floor), cp.maximum(p0, self.phi_floor),
                          S2, U2, mu, d)
        phi_new = self._solve(p0, mu + c.sigma_phi * t["mut"], t["ap_p"], t["b_p"],
                              self.fixed_phi, self.phi_inf,
                              0 if self._old is None else self._old[1])
        self.phi = cp.maximum(phi_new, cp.where(self.wall_d, 0.0,
                                                cp.maximum(self.phi_floor, 0.1 * p0)))
        dk = float(cp.abs(self.k - k0).max() / max(float(cp.abs(self.k).max()), 1e-300))
        dp = float(cp.abs(self.phi - p0).max() / max(float(cp.abs(self.phi).max()), 1e-300))
        mut_new = self._eddy_viscosity()
        mut_new = self.mu_t + self.relax_mu_t * (mut_new - self.mu_t)
        dm = float(cp.abs(mut_new - self.mu_t).max() / max(float(cp.abs(mut_new).max()), 1e-300))
        self.mu_t = mut_new
        solver.mu_t = self.mu_t
        change = max(dk, dp, dm)
        self.history.append({"k": dk, "phi": dp, "mu_t": dm,
                             "linear_failures": self.linear_failures})
        return change

    def _sources(self, k, phi, S2, U2, mu, d) -> dict:
        """The linearised sources of both equations at ``(k, Φ)`` (floored), per unit volume
        times the control volumes: ``ap_*`` the diagonal, ``b_*`` the right side."""
        cp, c, rho, V = self.cp, self.c, self.rho, self.V
        mut = cp.where(self.wall_d, 0.0, c.c_mu ** 0.25 * rho * phi / cp.sqrt(k))
        cmu34 = c.c_mu ** 0.75
        P = mut * S2
        Dk = cmu34 * rho * k ** 2.5 / phi                          # destruction of k
        limited = P > 20.0 * Dk
        Pk = cp.where(limited, 20.0 * Dk, P)
        fp = cp.minimum(cp.maximum(Pk / Dk, 0.5), 1.0)
        Lvk = c.kappa * cp.sqrt(S2) / cp.maximum(U2, 1e-300)
        lower = phi / (k * c.c11)
        low = Lvk <= lower
        Lvk = cp.maximum(Lvk, lower)
        upper = c.c12 * c.kappa * d * fp
        low &= lower < upper
        Lvk = cp.minimum(Lvk, upper)
        r2 = (phi / (k * Lvk)) ** 2
        cphi1 = c.zeta1 - c.zeta2 * r2
        xi = rho * d * cp.sqrt(0.3 * k) / (20.0 * mu)
        fphi = (1.0 + c.cd1 * xi) / (1.0 + xi ** 4)
        # Sources are linearised about (k, Φ): Newton for the sinks (a sink g ∝ φ^m contributes
        # m g/φ to the diagonal and (m - 1) g to the right side), the positive parts explicit,
        # so that every coefficient stays positive and the update contracts (the plain
        # (k^3/2/Φ) k form of k^5/2/Φ gives an update gain of about -2 near equilibrium;
        # Newton's about -0.2).
        # k: P_k - C_mu^3/4 rho k^5/2/Φ - 2 mu k/d^2. Unlimited, P_k ∝ k^-1/2: a falling
        # source, treated as a sink of order -1/2; limited, P_k = 20 Dk is explicit.
        ap_k = V * (2.5 * Dk / k + 2.0 * mu / d ** 2 + cp.where(limited, 0.0, 0.5 * Pk / k))
        b_k = V * (1.5 * Dk + cp.where(limited, Pk, 1.5 * Pk))
        # Φ: (ζ1 - ζ2 r²) Q - ζ3 rho k^3/2 - 6 mu f_φ Φ/d², Q = (Φ/k) P_k. Q ∝ Φ² unless P_k
        # is limited (Φ⁰); r² ∝ Φ² unless L_vK sits on its lower bound Φ/(k C11) (Φ⁰). The
        # destruction ζ3 rho k^3/2 (Φ⁰) keeps the positive Picard form (ζ3 rho k^3/2/Φ) Φ.
        Q = phi / k * Pk
        g = c.zeta2 * r2 * Q
        m = cp.maximum(cp.where(limited, 0.0, 2.0) + cp.where(low, 0.0, 2.0), 1.0)
        sink = c.zeta3 * rho * k ** 1.5 / phi + 6.0 * mu * fphi / d ** 2 + m * g / phi
        # the production ζ1 Q ∝ Φ² explicit would give the update a gain above 1; written
        # (ζ1 Q/Φ) Φ on the diagonal, as far as the sinks leave the diagonal at least half its
        # size, the gain is about 0.5
        prod = c.zeta1 * Q / phi
        theta = cp.minimum(1.0, 0.5 * sink / cp.maximum(prod, 1e-300))
        ap_p = V * (sink - theta * prod)
        b_p = V * ((m - 1.0) * g + (1.0 - theta) * c.zeta1 * Q)
        return {"mut": mut, "ap_k": ap_k, "b_k": b_k, "ap_p": ap_p, "b_p": b_p,
                "last": {"Lvk": Lvk, "cphi1": cphi1, "Pk": Pk, "S2": S2, "U2": U2}}

    # ------------------------------------------------------------ one scalar equation

    def _solve(self, phi0, gamma_node, ap_src, b_src, fixed_val, inflow_val, old):
        """Assemble and solve ``conv + diff + ap_src φ = b_src`` (+ time, relaxation, BCs)."""
        cp = self.cp
        s = self.s
        nnz = len(self.indices)
        data = cp.zeros(nnz)
        b = b_src.copy()
        net = cp.zeros(self.N)                                     # mass outflow of each node
        for p in self.parts:
            mdot = s.mdot[p["kind"]]                               # (E, nip), slot order
            g_ip = cp.einsum("sc,ec->es", p["Nip"], gamma_node[p["conn"]])
            # diffusion, implicit two-point part: outflow of a = Γ c (φ_a - φ_b); rows a, b
            Md = cp.einsum("sa,es,sc->eac", p["inc"], g_ip * p["c2"], p["inc"])
            # and the lagged remainder Γ grad(φ).(A - c e) as a source (into a, out of b)
            q = g_ip * cp.einsum("esc,ec->es", p["DT"], phi0[p["conn"]])
            cp.add.at(b, p["conn"].reshape(-1), cp.einsum("sa,es->ea", p["inc"], q).reshape(-1))
            # upwind advection: outflow of a = m φ_up
            up = cp.where(mdot >= 0, p["ea"][None, :], p["eb"][None, :])     # (E, nip)
            onehot = (up[..., None] == cp.arange(p["n"])[None, None, :]).astype(cp.float64)
            Mc = cp.einsum("sa,es,esc->eac", p["inc"], mdot, onehot)
            cp.add.at(data, p["pos"].reshape(-1), (Md + Mc).reshape(-1))
            cp.add.at(net, p["conn"].reshape(-1),
                      cp.einsum("sa,es->ea", p["inc"], mdot).reshape(-1))
        diag = self.diag_pos
        ap = ap_src.copy()
        # pressure zones: lagged boundary mass flow (out > 0) carries phi out, inflow carries in
        mb = s._mb if getattr(s, "_mb", None) is not None else cp.zeros(self.N)
        out = cp.where(self.pz, cp.maximum(mb, 0.0), 0.0)
        inn = cp.where(self.pz, cp.maximum(-mb, 0.0), 0.0)
        ap += out
        b += inn * inflow_val
        # bounded advection, ∇·(ρuφ) - φ ∇·(ρu): the same once continuity is met, but while the
        # coupled solve is unconverged a node taking in more mass than it gives out would
        # otherwise lose its diagonal dominance (and the solve its bound on φ)
        ap -= net + out - inn
        if s.dt is not None and not isinstance(old, int):
            ct = self.rho * self.V / s.dt
            ap += ct
            b += ct * old
        data[diag] += ap
        # under-relaxation of the sources only: a pseudo-time step of about relax/(1 - relax)
        # turbulent time scales (ap_src/V is the sinks' rate). Relaxing the whole diagonal
        # would scale it by the diffusion across the thinnest cells (wall layers, one-cell
        # slabs), which is large and does not limit anything: the step would be orders of
        # magnitude shorter than the turbulence's own time scale.
        r = (1.0 - self.relax) / self.relax * ap_src
        data[diag] += r
        b += r * phi0
        # Dirichlet rows
        fixed = self.fixed
        data = cp.where(fixed[self._rows], 0.0, data)
        data[diag] = cp.where(fixed, 1.0, data[diag])
        b = cp.where(fixed, fixed_val, b)
        x, ok = self._linear_solve(data, b, phi0)
        if not ok:
            self.linear_failures += 1
        return x

    def _linear_solve(self, data, b, x0):
        """Solve the scalar system (CSR ``data`` on the solver's pattern), rows scaled to a unit
        diagonal: FGMRES with an AmgX V-cycle when AmgX is available, else GMRES. The thin
        cells of wall layers (and of one-cell slabs) couple their nodes far more strongly
        across the thin direction than along it; Jacobi-preconditioned Krylov methods stall
        on that anisotropy, aggregation AMG with a Gauss–Seidel smoother does not."""
        csp = self.csp
        dinv = 1.0 / data[self.diag_pos]
        ds = data * dinv[self._rows]
        bs = dinv * b
        A = csp.csr_matrix((ds, self.indices, self.indptr), shape=(self.N, self.N))
        if self._amg is None:
            from zvcfd.fv.linear import AMGX, amgx_library

            self._amg = False
            if amgx_library():
                self._amg = AMGX(smoother="gs", selector="SIZE_2", strength=None,
                                 coarse="smooth")
        if self._amg:
            from types import SimpleNamespace

            from zvcfd.fv.linear import fgmres

            self._amg.upload(self.indptr, self.indices, ds)
            x, _, _, ok = fgmres(SimpleNamespace(matvec=lambda v: A @ v), bs, x0.copy(),
                                 self._amg.vcycle, rtol=self.rtol, restart=30, maxiter=300)
            return x, bool(ok)
        x, info = _gmres_scaled(A, bs, x0, self.rtol)
        return x, info == 0


def _gmres_scaled(A, b, x0, rtol, restart: int = 40, cycles: int = 25):
    """GMRES on a system whose rows are already scaled to a unit diagonal, from ``x0``;
    returns ``(x, info)``, ``info`` 0 when the residual fell by ``rtol`` relative to ``b``.

    No preconditioner is passed: cupyx's GMRES (CuPy 13) takes ``x0`` as its
    right-preconditioned variable, so with ``M`` the start would be ``M x0``, and its ``info``
    does not report a stop at ``maxiter``.
    """
    import cupy as cp
    from cupyx.scipy.sparse.linalg import gmres

    x, _ = gmres(A, b, x0=x0, tol=rtol, atol=0.0, restart=restart, maxiter=restart * cycles)
    res = float(cp.linalg.norm(b - A @ x)) / max(float(cp.linalg.norm(b)), 1e-300)
    return x, 0 if res <= rtol * 1.0001 else 1


__all__ = ["KkLConstants", "KkLModel", "tmr_freestream", "wall_distance"]
