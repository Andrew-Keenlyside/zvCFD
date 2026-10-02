"""The GPU finite-volume solver: CFX-style outer loop over GPU assembly and a block linear solve.

:class:`GPUSolver` has the interface of :class:`zvcfd.fv.reference.ReferenceSolver`
(mesh, :class:`~zvcfd.fv.reference.Fluid`, ``{zone: spec}`` boundary conditions,
``solve``, ``initialise``, ``solve_transient``, ``zone_flows``) and solves the
same discrete equations. The differences are those CFX makes:

- the Rhie–Chow interpolated gradient ``∇̄p`` is **lagged** (taken from the
  previous outer iteration), which keeps the matrix pattern to the element
  couplings. The CPU reference shows the lagged and implicit forms reach
  the same solution;
- each outer iteration solves the linearised block system only to a
  relative tolerance (``linear_rtol``, 0.1 by default: one order of
  magnitude, as CFX's coupled solver does), with AmgX or zvCFD's own
  additive-correction multigrid (:mod:`zvcfd.fv.linear`);
- convergence is reported as CFX does: RMS and maximum normalised
  residuals per equation, and the global mass imbalance. The
  normalisation here is ``|r| / (a_P U_ref)`` for momentum and
  ``|r| / (ρ U_ref V^{2/3})`` for continuity, with ``U_ref`` the largest
  velocity. CFX's exact form is in its Theory Guide, not yet in hand.

Everything per element runs on the GPU (:mod:`zvcfd.fv.gpu`). The host
keeps the boundary classification and boundary values
(:class:`~zvcfd.fv.reference.BoundaryConditions`), which are uploaded when
they change.

Pressure zones beyond the reference solver's
--------------------------------------------
``"lumped": model`` (a :class:`zvcfd.lumped.LumpedOutlet`)
    a 0-D outlet coupled **implicitly**: the zone's pressure is uniform and
    equal to ``a + r Q``, with ``(a, r)`` the model's coefficients for the
    step (or its steady pair) and ``Q`` the zone's volume outflow. ``Q`` is
    linear in the unknowns through the zone nodes' continuity rows, so the
    constraint is a rank-one term ``u wᵀ`` in the linear system
    (:class:`~zvcfd.fv.linear.BlockMatrix`), and the pressure force on the
    zone is ``p_j A_j`` in the matrix. Many outlets with large distal
    resistances stay stable, unlike a lagged pressure update. After each
    time step the model's state is committed with the step's final flow.
``"profile": "average"``
    average static pressure (CFX): the zone's pressures follow the profile
    extrapolated from the upstream neighbours (lagged), shifted so their
    area average is ``value``; ``"blend"`` (default 0.05) mixes in a
    uniform profile.
``"opening": True``
    an opening: ``value`` is the static pressure where flow leaves and the
    total pressure where it enters (``p = value − ½ρ|u|²`` at inflowing
    nodes, lagged).
"""

from __future__ import annotations

import time

import numpy as np

from zvcfd.fv.boundary_advection import (
    boundary_advection_correction,
    boundary_advection_tables,
    boundary_moments,
    inflow_cross_stream,
    inflow_offsets,
    neighbour_minmax,
)
from zvcfd.fv.geometry import boundary_subface_weights, dual_geometry
from zvcfd.fv.gpu import GPUAssembler
from zvcfd.fv.linear import BlockMatrix, make
from zvcfd.fv.pattern import node_graph
from zvcfd.fv.reference import (
    BDF,
    BoundaryConditions,
    Fluid,
    SolveReport,
    _call,
    bdf_coefficients,
    time_steps,
)
from zvcfd.mesh.core import UnstructuredMesh


class GPUSolver(BoundaryConditions):
    """Coupled finite-volume solve on one GPU.

    Args:
        mesh, fluid, bcs, source, advection, dt, reference_pressure, stokes,
            freeze_limiter, rhie_chow, transpose, t: as :class:`ReferenceSolver`.
            ``fluid.viscosity`` may be None (Newtonian) or a
            :class:`zvcfd.rheology.CarreauYasuda`.
        linear: ``"acm"``, ``"amgx"`` or ``"host-direct"``.
        linear_rtol: relative residual reduction of each linear solve.
        linear_options: passed to the linear solver.
        store_geometry: as :class:`zvcfd.fv.gpu.GPUAssembler` (``"auto"``, ``True``,
            ``"tets"``, ``False``): trade device memory for recomputing element geometry.
        kernels: ``"fast"`` (:class:`zvcfd.fv.fast.FastAssembler`: setup on the device,
            per-block threads, node-based gradients) or ``"classic"``
            (:class:`zvcfd.fv.gpu.GPUAssembler`). Both evaluate the same equations.
    """

    def __init__(self, mesh: UnstructuredMesh, fluid: Fluid, bcs: dict, *, source=None,
                 advection="high-resolution", dt: float | None = None,
                 reference_pressure=None, stokes: bool = False, rhie_chow: float = 1.0,
                 transpose="auto", freeze_limiter: int | None = 10, t: float = 0.0,
                 linear: str = "acm", linear_rtol: float = 0.1,
                 linear_options: dict | None = None, store_geometry="auto",
                 kernels: str = "fast"):
        t0 = time.time()
        self.store_geometry = store_geometry
        self.kernels = kernels
        self._setup(mesh, fluid, bcs, source=source, advection=advection, dt=dt, stokes=stokes,
                    rhie_chow=rhie_chow, transpose=transpose, freeze_limiter=freeze_limiter,
                    t=t)
        self._classify(reference_pressure)
        self._update_boundary(self.t)
        self._build()
        self.linear = make(linear, **({"rtol": linear_rtol} | (linear_options or {}))) \
            if linear != "host-direct" else make(linear)
        self.linear_rtol = linear_rtol
        self._initial_pressure()
        self._apply_fixed()
        self.setup_seconds = time.time() - t0

    def _setup(self, mesh, fluid, bcs, *, source, advection, dt, stokes, rhie_chow, transpose,
               freeze_limiter, t, n_own=None):
        """Options and host geometry (``n_own``: nodes owned, the rest halo; default all)."""
        import cupy as cp

        from zvcfd.rheology import CarreauYasuda

        self.cp = cp
        self.mesh, self.fluid, self.bcs, self.source = mesh, fluid, bcs, source
        self.advection, self.dt, self.stokes = advection, dt, stokes
        self.rhie_chow, self.freeze_limiter = rhie_chow, freeze_limiter
        self.t = float(t)
        self.N = mesh.n_nodes
        self._n_own = self.N if n_own is None else int(n_own)
        if fluid.viscosity is not None and not isinstance(fluid.viscosity, CarreauYasuda):
            raise ValueError("the GPU solver takes Newtonian or CarreauYasuda viscosity")
        self.transpose = (fluid.viscosity is not None) if transpose == "auto" else bool(transpose)
        if getattr(self, "kernels", "fast") == "fast":
            # the whole-mesh work on the device: colouring, pattern, geometry, node volumes;
            # the host keeps only the boundary sub-faces
            from zvcfd.fv.fast import FastAssembler
            from zvcfd.fv.geometry import DualGeometry, boundary_subfaces

            self.asm = FastAssembler(mesh, rho=fluid.rho, mu=fluid.mu,
                                     rheology=fluid.viscosity, transpose=self.transpose,
                                     stokes=stokes,
                                     store_geometry=getattr(self, "store_geometry", "auto"))
            self.geom = DualGeometry(cp.asnumpy(self.asm.node_volume), boundary={
                zid: boundary_subfaces(mesh.nodes, z.faces) for zid, z in mesh.zones.items()})
        else:
            self.geom = dual_geometry(mesh, keep_areas=False)
        self.V = self.geom.node_volume

    @property
    def fast(self) -> bool:
        return getattr(self, "kernels", "fast") == "fast"

    def _build(self):
        """Device structures: pattern, assembler, state, special zones, boundary data."""
        cp = self.cp
        if self.fast:
            self.pattern = self.asm.pattern
            indptr = indices = None                     # host copies only if a zone needs them
        else:
            indptr, indices = node_graph(self.mesh)
            self.pattern = (cp.asarray(indptr, dtype=cp.int64),
                            cp.asarray(indices, dtype=cp.int32))
            self.asm = GPUAssembler(self.mesh, rho=self.fluid.rho, mu=self.fluid.mu,
                                    rheology=self.fluid.viscosity, transpose=self.transpose,
                                    stokes=self.stokes,
                                    store_geometry=getattr(self, "store_geometry", "auto"))
        self.Vd = cp.asarray(self.V)
        self.U = cp.zeros((self.N, 3))
        self.P = cp.zeros(self.N)
        if self.fast:
            self.mdot = self.asm.zero_mdot()            # slot order
            self._adv = cp.zeros(self.N)                # advective diagonal of self.mdot
        else:
            self.mdot = {K["kind"]: cp.zeros((int(K["elem"].shape[0]), K["topo"].n_ip))
                         for K in self.asm.kinds}
        self._mb = cp.zeros(self.N)
        self._frozen_beta = None
        self.iteration = 0
        self._time = None
        A0 = BlockMatrix(self.pattern[0], self.pattern[1],
                         cp.zeros((int(self.pattern[1].size), 4, 4)))
        self._rows = A0.row_of_block()
        self._diag = A0.diagonal_positions()
        self._setup_special(indptr, indices)
        self._upload_boundary()
        T = boundary_advection_tables(self.mesh.nodes, self.sub, self.bcs)
        self._badv = None if T is None else {k: cp.asarray(v) for k, v in T.items()}
        self._dd = None if T is None else cp.asarray(inflow_offsets(self.mesh, T, sym=self.sym))
        self._oonly_d = None if T is None else cp.asarray(
            np.bincount(T["node"], T["outflow_only"], self.N) > 0)
        if self._badv is not None and self.fast:
            # pattern positions of the boundary nodes' rows, for the implicit form
            ip = cp.asnumpy(self.pattern[0])
            rows = np.unique(T["node"])
            cnt = ip[rows + 1] - ip[rows]
            pos = np.concatenate([np.arange(ip[r], ip[r + 1]) for r in rows]) \
                if len(rows) else np.zeros(0, np.int64)
            self._badv_rows = cp.asarray(np.repeat(rows, cnt))
            self._badv_pos = cp.asarray(pos)
        # rows kept before boundary replacement: fixed-velocity nodes (reactions, wall
        # shear) and lumped-zone nodes (their flow); a full copy would double the matrix
        keep = self.vel_fixed_comp.any(1)
        for z in self._special.values():
            if z["kind"] == "lumped":
                keep[cp.asnumpy(z["nodes"])] = True
        self._raw_blocks = cp.flatnonzero(cp.asarray(keep)[self._rows])
        for z in self._special.values():
            if z["kind"] == "lumped":
                z["blk_raw"] = cp.searchsorted(self._raw_blocks, z["blk"])
        if self.fast and getattr(self, "_fusable", True):
            self._setup_fused(keep)

    def _setup_fused(self, keep):
        """Per-node arrays of the fused boundary/scaling kernel (:mod:`zvcfd.fv._cuda_bc`)."""
        from zvcfd.fv import _cuda_bc
        from zvcfd.fv.fast import _raw

        cp, N, rho = self.cp, self.N, self.fluid.rho
        counts = cp.diff(self.pattern[0])
        kd = cp.asarray(keep)
        off = cp.cumsum(cp.where(kd, counts, 0)) - cp.where(kd, counts, 0)
        self._rawoff = cp.where(kd, off, -1).astype(cp.int64)
        self._nraw = int(cp.where(kd, counts, 0).sum())
        self._snl = cp.zeros((N, 3))
        self._lscale = cp.zeros(N)
        lumped_p = np.zeros(N, bool)
        for z in self._special.values():
            if z["kind"] == "lumped":
                self._snl[z["nodes"]] = z["Sn"]
                n = int(z["nodes"].size)
                self._lscale[z["nodes"]] = 1.0 / (rho * float(z["area"].sum()) * np.sqrt(n))
                lumped_p[cp.asnumpy(z["nodes"])] = True
        fixed = np.concatenate([self.vel_fixed_comp, self.p_fixed[:, None]], 1)
        self._fixed4 = cp.asarray(fixed.astype(np.uint8))
        known = fixed.copy()
        known[lumped_p, 3] = False
        self._known4 = cp.asarray(known.astype(np.uint8))
        self._fin_k = _raw("finalise", _cuda_bc.FINALISE)
        self._upd_k = _raw("update", _cuda_bc.UPDATE)
        # free rows per equation, for the RMS of the residual report
        n = self._n_own
        self._nfree = np.r_[(~self.vel_fixed_comp[:n]).sum(0), (~self.p_fixed[:n]).sum()]
        self.eliminate = getattr(self, "eliminate", True)

    def _apply_fixed(self):
        self.U[self._fixed_d] = self._vel_d[self._fixed_d]
        self.P[self._pfix_d] = self._pval_d[self._pfix_d]

    # ------------------------------------------------------------ boundary data

    def _setup_special(self, indptr, indices):
        """Lumped, average-pressure and opening zones: nodes, areas, stencils (once)."""
        cp, N = self.cp, self.N
        self._special = {}
        rows = self._rows
        seen = np.zeros(N, bool)
        for zid, spec in self.bcs.items():
            if spec["kind"] != "pressure":
                continue
            kind = "lumped" if spec.get("lumped") is not None else \
                "average" if spec.get("profile") == "average" else \
                "opening" if spec.get("opening") else None
            f, S = self.sub[zid]
            ok = f >= 0
            nodes = np.unique(f[ok])
            if kind is None:
                seen[nodes] = True
                continue
            if seen[nodes].any():
                raise ValueError(f"zone {zid}: a {kind} pressure zone may not share nodes with "
                                 "another pressure zone")
            seen[nodes] = True
            nodes = nodes[nodes < self._n_own]            # rows this solver owns
            if spec.get("grad") is not None:
                raise ValueError(f"zone {zid}: 'grad' is for fixed pressure zones")
            Sn = np.stack([np.bincount(f[ok], S[ok][:, k], N)[nodes] for k in range(3)], 1)
            area = np.bincount(f[ok], np.linalg.norm(S, axis=2)[ok], N)[nodes]
            z = {"kind": kind, "nodes": cp.asarray(nodes), "Sn": cp.asarray(Sn),
                 "area": cp.asarray(area)}
            if kind == "lumped":
                on = cp.zeros(N, bool)
                on[z["nodes"]] = True
                blk = cp.flatnonzero(on[rows])
                ucol, inv = cp.unique(self.pattern[1][blk], return_inverse=True)
                z.update(blk=blk, ucol=ucol.astype(cp.int64), inv=inv.reshape(-1),
                         model=spec["lumped"], p=None)
            elif kind == "average":
                if indptr is None:
                    indptr, indices = (cp.asnumpy(a) for a in self.pattern)
                # upstream neighbour: the interior neighbour most along the inward normal
                n_in = -Sn / np.linalg.norm(Sn, axis=1, keepdims=True)
                x = self.mesh.nodes
                up = np.empty(len(nodes), np.int64)
                for i, j in enumerate(nodes):
                    nb = indices[indptr[j]:indptr[j + 1]]
                    nb = nb[~self.outlet[nb]]
                    if nb.size == 0:
                        raise ValueError(f"zone {zid}: node {j} has no interior neighbour")
                    d = x[nb] - x[j]
                    up[i] = nb[np.argmax(d @ n_in[i] / np.linalg.norm(d, axis=1))]
                z.update(up=cp.asarray(up), dx=cp.asarray(x[nodes] - x[up]),
                         blend=float(spec.get("blend", 0.05)))
            self._special[zid] = z

    def _upload_boundary(self):
        """Boundary values, flows and outlet forces for the current time, on the device."""
        cp, N = self.cp, self.N
        self._fixed_d = cp.asarray(self.vel_fixed_comp)
        self._vel_d = cp.asarray(self.vel_value)
        self._pfix_d = cp.asarray(self.p_fixed)
        self._pval_d = cp.asarray(self.p_value)
        for z in self._special.values():       # these zones' pressures are the solver's own
            self._pval_d[z["nodes"]] = self.P[z["nodes"]]
        self._inflow_d = cp.asarray(self.inflow)
        self._fV_d = cp.asarray(self.f * self.V[:, None])
        self._outlet_d = cp.asarray(self.outlet)
        force = np.zeros((N, 3))
        a_out = np.zeros((N, 3))
        w_cons = np.zeros(N)
        w_only = np.zeros(N)
        w_stab = np.zeros(N)
        for zid, (f, S) in self.sub.items():
            spec = self.bcs[zid]
            if spec["kind"] != "pressure":
                continue
            ok = f >= 0
            fz = np.where(ok, f, 0)
            p_sub, grad = self._outlet[zid]
            # special zones: the pressure force p_j A_j is added in assemble()
            flux = p_sub[..., None] * S * (zid not in self._special)
            if grad is not None:
                if self.fluid.viscosity is None:
                    mu = np.full(ok.shape, self.fluid.mu)
                else:
                    S2 = 0.5 * (grad + np.swapaxes(grad, -1, -2))
                    mu = self.fluid.viscosity(np.sqrt(2 * np.einsum("fikj,fikj->fi", S2, S2)))
                tau = grad + np.swapaxes(grad, -1, -2) if self.transpose else grad
                flux = flux - mu[..., None] * np.einsum("fikj,fij->fik", tau, S)
            elif self.transpose:
                for k in range(3):
                    a_out[:, k] += np.bincount(f[ok], S[ok][:, k], N)
            for k in range(3):
                force[:, k] += np.bincount(f[ok], flux[ok][:, k], N)
            share = np.linalg.norm(S, axis=2) * ok
            tot = np.bincount(f[ok], share[ok], N)
            w = np.bincount(f[ok], share[ok] / tot[f[ok]], N)
            if spec.get("backflow", "consistent") == "consistent":
                w_cons += w
            else:
                w_only += w
            w_stab += spec.get("backflow_stabilisation", 0.0) * w
            del fz
        self._force_d = cp.asarray(force)
        self._aout_d = cp.asarray(a_out)
        self._has_aout = bool(np.abs(a_out).max() > 0) if a_out.size else False
        self._wcons_d = cp.asarray(w_cons)
        self._wonly_d = cp.asarray(w_only)
        self._wstab_d = cp.asarray(w_stab)

    def _initial_pressure(self):
        """Start from the pressure level of the boundaries, not from zero.

        The lagged Rhie–Chow gradient of a zero interior next to outlets at
        ``p_out`` is a spurious jump of ``p_out / Δx``, which the outer
        iterations then take many steps to forget (the implicit reference is
        level-invariant; the lagged form is only at convergence). The level is
        the area-weighted mean of the pressure zones' values; a lumped outlet
        counts as ``a + r Q`` with the inflow shared in proportion to ``1/r``.
        """
        rho = self.fluid.rho
        q_in = -sum(float(v.sum()) for v in self.sub_flow.values()) / rho
        lumped = {z: self.lumped_coefficients(z) for z, v in self._special.items()
                  if v["kind"] == "lumped"}
        g = sum(1.0 / r for _, r in lumped.values() if r > 0)
        num = den = 0.0
        for zid, (f, S) in self.sub.items():
            if self.bcs[zid]["kind"] != "pressure":
                continue
            ok = f >= 0
            area = float((np.linalg.norm(S, axis=2) * ok).sum())
            if zid in lumped:
                a, r = lumped[zid]
                val = a + (max(q_in, 0.0) / (r * g) * r if r > 0 and g > 0 else 0.0)
                self._pval_d[self._special[zid]["nodes"]] = val
            else:
                nodes = np.unique(f[ok])
                val = float(self.p_value[nodes].mean())
            num += val * area
            den += area
        if den > 0:
            self.P[:] = num / den

    def _mirror_d(self, g):
        """Mirror symmetry of nodal gradients at symmetry-plane nodes, on the device."""
        cp = self.cp
        if not self.sym.any():
            return g
        for ax in range(3):
            idx = cp.asarray(np.flatnonzero(self.sym[:, ax]))
            if idx.size == 0:
                continue
            if g.ndim == 2:
                g[idx, ax] = 0.0
            else:
                for j in range(3):
                    for k in range(3):
                        if (j == ax) != (k == ax):
                            g[idx, j, k] = 0.0
        return g

    # ------------------------------------------------------------ time terms

    def _time_terms(self):
        """``(diag (N,), rhs (N, 3), levels)`` of the time derivative on the device."""
        rho = self.fluid.rho
        if self._time is not None:
            scheme, dt, old, dt_prev = self._time
            c0, c1, c2 = bdf_coefficients(scheme, dt, dt_prev, len(old))
            w = rho * self.Vd / dt
            rhs = w[:, None] * (c1 * old[0][0] + (c2 * old[1][0] if len(old) > 1 and c2 else 0.0))
            levels = [(c1 / c0, old[0][0], old[0][1])]
            if len(old) > 1 and c2:
                levels.append((c2 / c0, old[1][0], old[1][1]))
            return c0 * w, rhs, levels
        if self.dt:
            w = rho * self.Vd / self.dt
            return w, w[:, None] * self.U, [(1.0, self.U.copy(), {k: v.copy()
                                                                  for k, v in self.mdot.items()})]
        return None, None, []

    # ------------------------------------------------------------ one outer iteration

    def _beta(self, gradU):
        cp = self.cp
        if self.stokes or self.advection == "upwind":
            return cp.zeros((self.N, 3))
        if self.advection != "high-resolution":
            return cp.full((self.N, 3), float(self.advection))
        if self._frozen_beta is not None:
            return self._frozen_beta
        beta = self.asm.limiter(self.U, gradU)
        if self.freeze_limiter is not None and self.iteration >= self.freeze_limiter:
            self._frozen_beta = beta
        return beta

    def _own_inflow(self):
        """Nodes where the flow enters through a pressure boundary (``ṁ_b < 0``, lagged):
        their rows leave out the deferred corrections of the faces they are upwind of.

        Momentum crossing a pressure boundary is ``ṁ_b u_node``, while the faces
        leaving the node carry ``u_node + β ∇u_node · (x_ip − x_node)``. In the
        first row of control volumes those corrections had no counterpart, the
        advection operator's truncation error there was first order (0.0101 of the
        momentum throughput against 0.0005 inside, on a pipe in developed flow), and
        a pressure-driven pipe's flow came out 3–4 % high, not shrinking with the
        mesh. Dropping the corrections at those nodes altogether only moved the
        mismatch one row in. Here the faces keep them for the downstream row, and
        the node's boundary flux carries exactly the momentum they take out: the
        scheme stays conservative, and on prism and hexahedral inflow rows it is
        consistent. On tetrahedral ones :meth:`_boundary_advection` adds the
        cross-stream part of the difference (docs/spec/fv_numerics.md). ``None``
        with the full second-order boundary reconstruction
        (``boundary_reconstruction``), which is unstable on coarse meshes. Nodes with fixed
        velocity are left out: their rows are replaced, and their reactions keep the
        corrections. So are zones that count only outflowing momentum
        (``backflow: outflow``).
        """
        if (self.stokes or self._badv is None
                or getattr(self, "boundary_reconstruction", False)):
            return None
        return (self._mb < 0.0) & ~self._fixed_d.all(axis=1) & ~self._oonly_d

    def assemble(self):
        """The linearised block system about the current state: ``(BlockMatrix, b (N, 4))``."""
        A, b = self._assemble_stage2(self._assemble_stage1())
        self._finish_lumped(A, b, self._lumped_pieces())
        return A, b

    def _assemble_stage1(self) -> dict:
        """Nodal gradients, limiter blend and momentum diagonal (valid at owned nodes).

        A partitioned solver exchanges these at halo nodes before stage 2.
        """
        g = self.asm.gradient(self.cp.concatenate([self.U, self.P[:, None]], 1))   # one pass
        gradU = self._mirror_d(self.cp.ascontiguousarray(g[:, :3]))
        gradP = self._mirror_d(self.cp.ascontiguousarray(g[:, 3]))
        if self.fast and self.fluid.viscosity is None:
            # static viscous part + the advective part the last mass-flow pass left
            diag = self.asm.dvisc if self.stokes else self.asm.dvisc + self._adv
        else:
            diag = self.asm.diagonal(self.mdot, self.U)
        return {"gradU": gradU, "gradP": gradP, "beta": self._beta(gradU), "diag": diag}

    def _assemble_stage2(self, pre: dict):
        """Assembly from stage 1's (exchanged) fields; lumped rows are finished separately."""
        cp = self.cp
        gradU, gradP, beta, diag = pre["gradU"], pre["gradP"], pre["beta"], pre["diag"]
        tdiag, trhs, levels = self._time_terms()
        t = tdiag if tdiag is not None else 0.0
        dnode = self.rhie_chow * self.Vd / (diag + t)
        snode = self.rhie_chow * self.Vd / diag
        if self.fast:
            data, b = self.asm.assemble(self.U, self.mdot, gradU, beta, gradP, dnode, snode,
                                        levels=levels, own=self._own_inflow())
        else:
            data, b = self.asm.assemble(self.pattern, self.U, self.mdot, gradU, beta, gradP,
                                        dnode, snode, levels=levels, own=self._own_inflow())
        self._dnode, self._snode, self._levels = dnode, snode, levels
        self.aP = diag + t
        self._boundary_advection(data, b, gradU, beta)
        if tdiag is not None:
            for k in range(3):
                data[self._diag, k, k] += tdiag
            b[:, :3] += trhs
        b[:, :3] += self._fV_d
        b[:, 3] -= self._inflow_d
        # pressure outlets: pressure force (and traction), outflowing momentum
        b[:, :3] -= self._force_d
        if self.transpose and float(cp.abs(self._aout_d).max()) > 0:
            mu = self._node_viscosity_d(gradU)
            b[:, :3] += mu[:, None] * cp.einsum("njk,nj->nk", gradU, self._aout_d)
        if not self.stokes:
            acc = self._wcons_d * self._mb + self._wonly_d * cp.maximum(self._mb, 0.0) \
                + self._wstab_d * cp.maximum(-self._mb, 0.0)
            for k in range(3):
                data[self._diag, k, k] += acc
        for zid, z in self._special.items():
            d = self._diag[z["nodes"]]
            if z["kind"] == "lumped":           # implicit pressure force p_j A_j
                for k in range(3):
                    data[d, k, 3] += z["Sn"][:, k]
            else:
                pb = self._dynamic_pressure(zid, z, gradP)
                self._pval_d[z["nodes"]] = pb
                b[z["nodes"], :3] -= pb[:, None] * z["Sn"]
        A = BlockMatrix(self.pattern[0], self.pattern[1], data)
        self._raw = RawRows(self._raw_blocks, self._rows[self._raw_blocks],
                            self.pattern[1][self._raw_blocks], data[self._raw_blocks], b.copy())
        self._dirichlet(A, b)
        return A, b

    def _boundary_advection(self, data, b, gradU, beta) -> None:
        """Momentum entering through pressure boundaries beyond ``ṁ_b u_node``.

        By default the cross-stream consistency term at the nodes of
        :meth:`_own_inflow`, on the right-hand side. Opt-in
        (``boundary_reconstruction``): the second-order, limited boundary
        reconstruction (:mod:`zvcfd.fv.boundary_advection`), implicit through the
        nodal-gradient operator with the fast kernels, on the right-hand side with the
        classic ones."""
        if self.stokes or self._badv is None or self.advection == "upwind":
            return
        cp = self.cp
        if not getattr(self, "boundary_reconstruction", False):
            own = self._own_inflow()
            if own is not None:
                G = inflow_cross_stream(cp, self._dd, self._mb, gradU, beta)
                b[:, :3] -= cp.where(own[:, None], G, 0.0)
            return
        if getattr(self, "_sym_d", None) is None:
            self._sym_d = cp.asarray(self.sym)
        frozen = getattr(self, "_frozen_bsf", None)
        lo = hi = None
        if frozen is None:
            lo, hi = neighbour_minmax(cp, self.pattern[0], self.pattern[1], self.U)
        D, bsf = boundary_moments(cp, self._badv, self.U, self._mb, gradU, beta, lo, hi,
                                  self.fluid.rho, self.N, sym=self._sym_d, bsf=frozen)
        if frozen is None and self._frozen_beta is not None:
            self._frozen_bsf = bsf             # frozen with High Resolution's limiter
        if not self.fast:
            b[:, :3] -= boundary_advection_correction(cp, D, gradU)
            return
        # component k at node i: sum_j g_ij . D[i, k] u_j,k, on the diagonal of block (i, j)
        r, p = self._badv_rows, self._badv_pos
        g = self.asm.gop.reshape(-1, 3)[p]
        d4 = data.reshape(-1, 4, 4)
        for k in range(3):
            d4[p, k, k] += cp.einsum("pj,pj->p", D[r, k], g)

    def _assemble_fused(self):
        """Assembly, boundary rows, residual statistics and row scaling in one pass.

        Returns ``(A, b, device scalars)``: the scaled system, and on the device
        ``[U_ref, p_ref, max |u|]`` followed by the residual partials; nothing is
        read back here.
        """
        cp = self.cp
        self._raw = None              # the last iteration's rows: freed before the new ones
        pre = self._assemble_stage1()
        gradU, gradP, beta, diag = pre["gradU"], pre["gradP"], pre["beta"], pre["diag"]
        tdiag, trhs, levels = self._time_terms()
        t = tdiag if tdiag is not None else 0.0
        dnode = self.rhie_chow * self.Vd / (diag + t)
        snode = self.rhie_chow * self.Vd / diag
        data, b = self.asm.assemble(self.U, self.mdot, gradU, beta, gradP, dnode, snode,
                                    levels=levels, own=self._own_inflow())
        self._dnode, self._snode, self._levels = dnode, snode, levels
        self.aP = aP = diag + t
        self._boundary_advection(data, b, gradU, beta)
        if self.transpose and self._has_aout:
            mu = self._node_viscosity_d(gradU)
            b[:, :3] += mu[:, None] * cp.einsum("njk,nj->nk", gradU, self._aout_d)
        for zid, z in self._special.items():
            if z["kind"] != "lumped":
                pb = self._dynamic_pressure(zid, z, gradP)
                self._pval_d[z["nodes"]] = pb
                b[z["nodes"], :3] -= pb[:, None] * z["Sn"]
        umax = cp.abs(self.U).max()
        uref = cp.maximum(cp.maximum(umax, cp.abs(self._vel_d).max()), 1e-30)
        pref = cp.maximum(cp.maximum(cp.ptp(self.P), self.fluid.rho * uref * uref), 1e-30)
        scal = cp.stack([uref, pref, cp.maximum(umax, 1e-30)])
        rawdata = cp.empty((max(self._nraw, 1), 4, 4))
        rawb = cp.zeros((self.N, 4))
        rs = cp.empty((self.N, 4))
        x = self._state()
        nth = self.N * 4
        nb = (nth + 255) // 256
        partial = cp.empty(nb * 8)
        dummy = cp.zeros(1)
        self._fin_k((nb,), (256,), (
            np.int64(self.N), self.pattern[0], self.pattern[1], self._diag, data, b,
            np.int32(tdiag is not None), dummy if tdiag is None else tdiag,
            dummy if trhs is None else cp.ascontiguousarray(trhs),
            self._fV_d, self._force_d, self._inflow_d, self._mb, self._wcons_d, self._wonly_d,
            self._wstab_d, np.int32(self.stokes), self._snl, self._fixed4, self._known4,
            self._vel_d, self._pval_d, np.int32(bool(self.eliminate)), self._rawoff,
            rawdata, rawb, x, aP, self.Vd, np.float64(self.fluid.rho), scal, self._lscale,
            rs, partial))
        self._raw = RawRows(self._raw_blocks, self._rows[self._raw_blocks],
                            self.pattern[1][self._raw_blocks], rawdata[:self._nraw], rawb,
                            copy=False)
        A = BlockMatrix(self.pattern[0], self.pattern[1], data)
        pieces = self._lumped_pieces(host=False)
        self._finish_lumped(A, b, pieces, rs=rs)
        return A, b, cp.concatenate([scal, partial])

    def _residuals_from(self, host) -> dict:
        """The CFX-style residual report from :meth:`_assemble_fused`'s scalars (on the host)."""
        uref = host[2]
        part = host[3:].reshape(-1, 8)
        out = {}
        for k, name in enumerate(("u", "v", "w", "p")):
            n = self._nfree[k]
            out[f"rms_{name}"] = float(np.sqrt(part[:, k].sum() / n) / uref) if n else 0.0
            out[f"max_{name}"] = float(part[:, 4 + k].max() / uref) if n else 0.0
        return out

    def _dynamic_pressure(self, zid, z, gradP):
        """This iteration's (lagged) pressures on an average-pressure or opening zone."""
        cp = self.cp
        from zvcfd.lumped import _Signal

        p0 = _Signal(self.bcs[zid].get("value", 0.0))(self.t)
        if z["kind"] == "average":
            up = z["up"]
            pt = self.P[up] + cp.einsum("nk,nk->n", gradP[up], z["dx"])
            mean = float((pt * z["area"]).sum() / z["area"].sum())
            return p0 + (1.0 - z["blend"]) * (pt - mean)
        nodes = z["nodes"]
        inflow = self._mb[nodes] < 0
        return p0 - cp.where(inflow, 0.5 * self.fluid.rho * (self.U[nodes] ** 2).sum(1), 0.0)

    def lumped_coefficients(self, zid) -> tuple[float, float]:
        """``(a, r)`` of a lumped zone for the current step (steady pair outside a step)."""
        model = self._special[zid]["model"]
        if self._time is None:
            return model.steady_coefficients(self.t)
        return model.coefficients(self.t, self._time[1])

    def _lumped_pieces(self, host: bool = True) -> dict:
        """This solver's part of each lumped zone's flow ``Q = w·x + c``: ``{zone: (wi, w, c)}``.

        ``w`` sums the raw continuity rows of the zone's (owned) nodes, so
        ``Q`` is linear in the unknowns; ``wi`` are its flat dof indices.
        ``c`` is a float, or with ``host=False`` a device scalar (no
        synchronisation per zone).
        """
        cp = self.cp
        out = {}
        if not self._special:
            return out
        raw, braw = self._raw
        rho = self.fluid.rho
        for zid, z in self._special.items():
            if z["kind"] != "lumped":
                continue
            w = cp.zeros((z["ucol"].size, 4))
            cp.add.at(w, z["inv"], raw.data[z["blk_raw"], 3, :])
            wi = (4 * z["ucol"][:, None] + cp.arange(4)).reshape(-1)
            c = braw[z["nodes"], 3].sum() / rho
            out[zid] = (wi, (-w / rho).reshape(-1), float(c) if host else c)
        return out

    def _finish_lumped(self, A: BlockMatrix, b, pieces: dict, c_total: dict | None = None,
                       coefficients: dict | None = None, rs=None) -> None:
        """Pressure rows of lumped zones: ``p_j − r Q(x) = a``, ``Q = w·x + c`` (rank one).

        Each row is scaled by ``s = min(1, ρ/r)``, which makes it a mass-flow
        residual like the continuity rows: unscaled, rows of size ``r Q``
        dominate the norm FGMRES minimises, and it stops with the momentum
        rows unconverged (gate B, two outlets at 1000× the domain resistance).
        A partitioned solver passes the zone totals ``c_total`` and adds the
        global rank-one term itself; here ``w`` is this solver's own.
        """
        cp = self.cp
        rho = self.fluid.rho
        for zid, (wi, w, c) in pieces.items():
            z = self._special[zid]
            a, r = (coefficients or {}).get(zid) or self.lumped_coefficients(zid)
            c = c if c_total is None else c_total[zid]
            nodes = z["nodes"]
            sc = min(1.0, rho / r) if r > 0 else 1.0
            s_row = rs[nodes, 3] if rs is not None else 1.0      # rows already scaled
            A.data[self._diag[nodes], 3, 3] = sc * s_row
            b[nodes, 3] = sc * (a + r * c) * s_row
            if c_total is None:
                A.lowrank.append((4 * nodes + 3, -sc * r * s_row * cp.ones(nodes.size), wi, w))
            z["a"], z["r"], z["sc"] = a, r, sc

    def _node_viscosity_d(self, gradU):
        cp = self.cp
        if self.fluid.viscosity is None:
            return cp.full(self.N, self.fluid.mu)
        s = 0.5 * (gradU + cp.swapaxes(gradU, 1, 2))
        g = cp.sqrt(2.0 * cp.einsum("ijk,ijk->i", s, s))
        v = self.fluid.viscosity
        return v.mu_inf + (v.mu_0 - v.mu_inf) * (1 + (v.lam * g) ** v.a) ** ((v.n - 1) / v.a)

    def _dirichlet(self, A: BlockMatrix, b):
        cp = self.cp
        rows = self._rows
        for k in range(3):
            m = self._fixed_d[rows, k]
            A.data[m, k, :] = 0.0
        m = self._pfix_d[rows]
        A.data[m, 3, :] = 0.0
        for k in range(3):
            fixed = self._fixed_d[:, k]
            A.data[self._diag[fixed], k, k] = 1.0
            b[fixed, k] = self._vel_d[fixed, k]
        A.data[self._diag[self._pfix_d], 3, 3] = 1.0
        b[self._pfix_d, 3] = self._pval_d[self._pfix_d]
        del cp

    def _row_scales(self) -> tuple[float, float, dict]:
        """``(U_ref, p_ref, {lumped zone: (area, nodes)})`` for :meth:`_scale_rows`."""
        cp = self.cp
        uref = max(float(cp.abs(self.U).max()), float(cp.abs(self._vel_d).max()), 1e-30)
        pref = max(float(cp.ptp(self.P)), self.fluid.rho * uref * uref, 1e-30)
        zones = {zid: (float(z["area"].sum()), int(z["nodes"].size))
                 for zid, z in self._special.items() if z["kind"] == "lumped"}
        return uref, pref, zones

    def _scale_rows(self, A: BlockMatrix, b, scales=None):
        """Make every row dimensionless before the linear solve (in place).

        Momentum rows are divided by ``a_P U_ref``, continuity rows by
        ``ρ U_ref V^{2/3}`` (the normalisation of the reported residuals),
        lumped-outlet rows by the patch flux ``ρ U_ref A_patch √n``, fixed
        rows by their variable's scale. FGMRES then
        minimises a residual in which no equation's units dominate: with
        rows in SI units a lumped outlet's rows (kg/s) outweigh momentum rows
        (N) by orders of magnitude, one Krylov step removes them, and the
        solver stops with momentum unsolved. The solution is unchanged; the
        block smoothers are invariant to row scaling.
        """
        cp = self.cp
        uref, pref, zones = scales or self._row_scales()
        rs = cp.empty((self.N, 4))
        rs[:, :3] = (1.0 / (self.aP * uref))[:, None]
        rs[:, :3] = cp.where(self._fixed_d, 1.0 / uref, rs[:, :3])
        rs[:, 3] = 1.0 / (self.fluid.rho * uref * self.Vd ** (2 / 3))
        pinned = self._pfix_d.copy()
        for z in self._special.values():
            if z["kind"] == "lumped":
                pinned[z["nodes"]] = False
        rs[:, 3] = cp.where(pinned, 1.0 / pref, rs[:, 3])
        # a lumped zone's rows repeat one patch-level flow equation at each of its n
        # nodes: scale by the patch flux, and by 1/√n for the repetition
        for zid, z in self._special.items():
            if z["kind"] == "lumped":
                area, n = zones[zid]
                rs[z["nodes"], 3] = 1.0 / (self.fluid.rho * uref * area * np.sqrt(n))
        A.data *= rs[self._rows][:, :, None]
        flat = rs.reshape(-1)
        A.lowrank = [(ui, uv * flat[ui], wi, wv) for ui, uv, wi, wv in A.lowrank]
        b *= rs
        return rs

    def _residuals(self, A: BlockMatrix, b):
        """CFX-style RMS and maximum normalised residuals of the current state."""
        r = (b.reshape(-1) - A.matvec(self._state())).reshape(-1, 4)
        return residual_summary([self._residual_sums(r, max(float(self.cp.abs(self.U).max()),
                                                                1e-30))])

    def _state(self):
        return self.cp.concatenate([self.U, self.P[:, None]], 1).reshape(-1)

    def _residual_sums(self, r, uref: float) -> dict:
        """``{equation: (Σ r², count, max)}`` of normalised residuals at owned free rows."""
        cp = self.cp
        n = self._n_own
        mom = cp.abs(r[:n, :3]) / (self.aP[:n, None] * uref)
        cont = cp.abs(r[:n, 3]) / (self.fluid.rho * uref * self.Vd[:n] ** (2 / 3))
        free = ~self._fixed_d[:n]
        pfree = ~self._pfix_d[:n]
        out = {}
        for k, name in enumerate(("u", "v", "w")):
            v = mom[free[:, k], k]
            out[name] = (float((v ** 2).sum()), int(v.size), float(v.max()) if v.size else 0.0)
        v = cont[pfree]
        out["p"] = (float((v ** 2).sum()), int(v.size), float(v.max()) if v.size else 0.0)
        return out

    def update_mass_flows(self, U_old=None):
        self._massflow_from(self._mirror_d(self.asm.gradient(self.P)))

    def _massflow_from(self, gradP):
        if self.fast:
            new, imb, self._adv = self.asm.massflow(self.U, self.P, gradP, self._dnode,
                                                    self._snode, levels=self._levels)
        else:
            new, imb = self.asm.massflow(self.U, self.P, gradP, self._dnode, self._snode,
                                         levels=self._levels)
        self.mdot = new
        self._interior_out = imb
        self._mb = self.cp.where(self._outlet_d, -(imb + self._inflow_d), 0.0)

    def _iterate(self, max_iterations, tol, log, hist, residual_target=None):
        if self.fast and getattr(self, "_fin_k", None) is not None:
            return self._iterate_fused(max_iterations, tol, log, hist, residual_target)
        return self._iterate_staged(max_iterations, tol, log, hist, residual_target)

    def _iterate_fused(self, max_iterations, tol, log, hist, residual_target=None):
        """Outer iterations with the fused kernels: one host read-back per iteration."""
        cp = self.cp
        converged = False
        it = 0
        for it in range(1, max_iterations + 1):
            self.iteration = it - 1
            t0 = time.time()
            x0 = self._state()
            A, b, scal = self._assemble_fused()
            cp.cuda.Device().synchronize()
            t_asm = time.time() - t0
            x, info = self.linear.solve(A, b.reshape(-1), x0, rtol=self.linear_rtol)
            del A, b                  # the matrix is not needed again: one copy at a time
            nb = (self.N + 255) // 256
            part = cp.empty(nb * 5)
            self._upd_k((nb,), (256,), (np.int64(self.N), x, self.U, self.P, part))
            self.update_mass_flows()
            pr = part.reshape(-1, 5)
            small = cp.concatenate([scal, pr[:, 0].max()[None], pr[:, 1].max()[None],
                                    pr[:, 2].max()[None], pr[:, 3].min()[None],
                                    pr[:, 4].max()[None], self._zone_contrib()]).get()
            ns = scal.size
            res = self._residuals_from(small[:ns])
            du_, um, dp_, pmin, pmax = small[ns:ns + 5]
            du = float(du_) / max(float(um), 1e-300)
            dp = float(dp_) / max(float(pmax - pmin), 1e-300)
            q = self._zone_flows_from(small[ns + 5:])
            rec = {"iteration": it, "du": du, "dp": dp, "linear_iterations": info.iterations,
                   "linear_residual": info.residual, "linear_converged": bool(info.converged),
                   "linear_solver": getattr(self.linear, "active", self.linear.name),
                   "assemble_s": t_asm, "linear_setup_s": info.setup_seconds,
                   "linear_solve_s": info.seconds, "seconds": time.time() - t0, **res}
            gross = sum(abs(v) for v in q.values()) or 1.0
            rec["imbalance"] = sum(q.values()) / gross
            rec["flows"] = q
            hist.append(rec)
            if log:
                log(f"  it {it:4d}  rms u {res['rms_u']:.2e} p {res['rms_p']:.2e}  "
                    f"du {du:.2e} dp {dp:.2e}  lin {info.iterations} ({info.residual:.1e})  "
                    f"imb {rec['imbalance']:+.1e}  {rec['seconds']:.2f}s")
            rms = max(res["rms_u"], res["rms_v"], res["rms_w"], res["rms_p"])
            done = (du < tol and dp < tol) or \
                (residual_target is not None and rms < residual_target)
            if it > 1 and done and info.converged:
                converged = True
                break
        return it, converged

    def _iterate_staged(self, max_iterations, tol, log, hist, residual_target=None):
        cp = self.cp
        converged = False
        it = 0
        for it in range(1, max_iterations + 1):
            self.iteration = it - 1
            t0 = time.time()
            A, b = self.assemble()
            res = self._residuals(A, b)
            x0 = cp.concatenate([self.U, self.P[:, None]], 1).reshape(-1)
            self._scale_rows(A, b)
            cp.cuda.Device().synchronize()
            t_asm = time.time() - t0
            x, info = self.linear.solve(A, b.reshape(-1), x0, rtol=self.linear_rtol)
            del A, b
            X = x.reshape(-1, 4)
            du = float(cp.abs(X[:, :3] - self.U).max()) / max(float(cp.abs(X[:, :3]).max()), 1e-300)
            dp = float(cp.abs(X[:, 3] - self.P).max()) / max(float(cp.ptp(X[:, 3])), 1e-300)
            self.U, self.P = X[:, :3].copy(), X[:, 3].copy()
            self.update_mass_flows()
            rec = {"iteration": it, "du": du, "dp": dp, "linear_iterations": info.iterations,
                   "linear_residual": info.residual, "linear_converged": bool(info.converged),
                   "linear_solver": getattr(self.linear, "active", self.linear.name),
                   "assemble_s": t_asm, "linear_setup_s": info.setup_seconds,
                   "linear_solve_s": info.seconds, "seconds": time.time() - t0, **res}
            q = self.zone_flows()
            gross = sum(abs(v) for v in q.values()) or 1.0
            rec["imbalance"] = sum(q.values()) / gross
            rec["flows"] = q
            hist.append(rec)
            if log:
                log(f"  it {it:4d}  rms u {res['rms_u']:.2e} p {res['rms_p']:.2e}  "
                    f"du {du:.2e} dp {dp:.2e}  lin {info.iterations} ({info.residual:.1e})  "
                    f"imb {rec['imbalance']:+.1e}  {rec['seconds']:.2f}s")
            # a stalled linear solve also gives small du, dp: not convergence
            rms = max(res["rms_u"], res["rms_v"], res["rms_w"], res["rms_p"])
            done = (du < tol and dp < tol) or \
                (residual_target is not None and rms < residual_target)
            if it > 1 and done and info.converged:
                converged = True
                break
        return it, converged

    # ------------------------------------------------------------ public interface

    def solve(self, *, max_iterations: int = 200, tol: float = 1e-10, log=None,
              residual_target: float | None = None) -> SolveReport:
        """Steady solve (false time step if ``dt``): outer iterations to ``tol`` (relative
        change of ``u`` and ``p``) or to ``residual_target`` (largest RMS normalised residual).

        Lumped outlets are then put at rest at the converged flows, ready for
        a transient run to start from the steady state.
        """
        t0 = time.time()
        hist = []
        it, conv = self._iterate(max_iterations, tol, log, hist, residual_target)
        q = self.patch_flows()
        for zid, z in self._special.items():
            if z["kind"] == "lumped":
                z["model"].initialise(q[zid], self.t)
        return SolveReport(it, conv, hist, time.time() - t0)

    def patch_flows(self) -> dict[int, float]:
        """Volume outflow (m³/s) through each zone; inflow negative."""
        return {z: v / self.fluid.rho for z, v in self.zone_flows().items()}

    def patch_pressures(self) -> dict[int, float]:
        """Area-mean pressure (Pa) on each zone."""
        P = self.cp.asnumpy(self.P)
        out = {}
        for zid, (f, S) in self.sub.items():
            ok = f >= 0
            a = np.linalg.norm(S, axis=2) * ok
            out[zid] = float((P[np.where(ok, f, 0)] * a).sum() / a.sum())
        return out

    def initialise(self, U=None, P=None) -> None:
        cp = self.cp
        x = self.mesh.nodes
        if U is not None:
            self.U = cp.asarray(np.array(_call(U, x, self.t), float).reshape(self.N, 3))
        if P is not None:
            self.P = cp.asarray(np.array(_call(P, x, self.t), float).reshape(self.N))
        self.U[self._fixed_d] = self._vel_d[self._fixed_d]
        self.P[self._pfix_d] = self._pval_d[self._pfix_d]
        self.assemble()
        self.update_mass_flows()

    def solve_transient(self, dt, steps: int | None = None, *, scheme: str = "bdf2",
                        loops: int = 5, tol: float = 1e-8, callback=None,
                        log=None, residual_target: float | None = None) -> SolveReport:
        """March time steps (BDF1/BDF2, variable step) with coefficient loops per step.

        ``dt`` is a constant (with ``steps``), a sequence of step sizes, or a
        schedule ``dt(t)`` (with ``steps``). Each step's coefficient loops stop
        when the relative changes fall below ``tol`` or, with
        ``residual_target``, when the largest RMS normalised residual does
        (CFX's criterion). Lumped outlets are committed
        with each step's final flow. ``callback(solver, dt)`` (or
        ``callback(solver)``) runs after each step. ``converged`` in the
        report is whether every step's coefficient loops met ``tol``.
        """
        if scheme not in BDF:
            raise ValueError(f"scheme {scheme!r}")
        t0 = time.time()
        hist = []
        old = [(self.U.copy(), {k: v.copy() for k, v in self.mdot.items()})]
        total = 0
        dt_prev = None
        all_ok = True
        nargs = _nargs(callback)
        for dt in time_steps(dt, steps, self.t):
            self.t += dt
            self._time = (scheme, dt, old, dt_prev)
            dt_prev = dt
            self._update_boundary(self.t)
            self._upload_boundary()
            self.U[self._fixed_d] = self._vel_d[self._fixed_d]
            self.P[self._pfix_d] = self._pval_d[self._pfix_d]
            n, ok = self._iterate(loops, tol, log, hist, residual_target)
            all_ok &= ok
            total += n
            hist[-1]["t"] = self.t
            q = self.patch_flows()
            for zid, z in self._special.items():
                if z["kind"] == "lumped":
                    z["model"].advance(q[zid], self.t, dt)
            old = [(self.U.copy(), {k: v.copy() for k, v in self.mdot.items()})] + old[:1]
            if callback is not None:
                callback(self, dt) if nargs >= 2 else callback(self)
        self._time = None
        return SolveReport(total, all_ok, hist, time.time() - t0)

    # ------------------------------------------------------------ forces

    def zone_forces(self) -> dict[int, np.ndarray]:
        """Force of the fluid on each zone (N), from consistent reactions (owned nodes' share).

        As :meth:`ReferenceSolver.zone_forces`: at nodes with fixed velocity
        components, minus the momentum residual before the boundary rows
        replaced it, shared among the zones fixing each component by
        sub-face area, less the momentum an inlet carries in; at pressure
        zones, the pressure force on the zone (``p A``; the lagged viscous part
        of the zero-gradient outlet is left out).
        """
        cp = self.cp
        N, n = self.N, self._n_own
        r = cp.asnumpy(self._raw.residual(self._state())[:, :3])
        fixed = self.vel_fixed_comp
        react = np.where(fixed, -r, 0.0)
        react[n:] = 0.0
        per, tot = {}, np.zeros((N, 3))
        for zid, (f, S) in self.sub.items():
            kind = self.bcs[zid]["kind"]
            if kind == "pressure":
                continue
            ok = f >= 0
            a = np.bincount(f[ok], (np.linalg.norm(S, axis=2) * ok)[ok], N)
            comp = np.ones(3, bool)
            if kind == "symmetry":
                sn = np.abs(S.reshape(-1, 3).sum(0))
                comp = sn == sn.max()
            per[zid] = a[:, None] * comp[None, :]
            tot += per[zid]
        P = cp.asnumpy(self.P)
        out = {}
        for zid, (f, S) in self.sub.items():
            ok = f >= 0
            own = ok & (f < n)
            if self.bcs[zid]["kind"] == "pressure":
                z = self._special.get(zid)
                p_sub = P[np.where(ok, f, 0)] if z is not None else self._outlet[zid][0]
                out[zid] = (p_sub[..., None] * S * own[..., None]).reshape(-1, 3).sum(0)
                continue
            share = np.divide(per[zid], tot, out=np.zeros_like(tot), where=tot > 0)
            F = (share * react).sum(0)
            if self.bcs[zid]["kind"] == "velocity" and not self.stokes:
                F -= np.einsum("fi,fik->k", self.sub_flow[zid] * own, self.sub_vel[zid])
            out[zid] = F
        return out

    # ------------------------------------------------------------ wall shear

    def _wall_geometry(self):
        """Wall nodes, their outward unit normals and areas (the wall zones' sub-faces)."""
        if getattr(self, "_wallgeo", None) is None:
            N = self.N
            Sn = np.zeros((N, 3))
            area = np.zeros(N)
            for zid, (f, S) in self.sub.items():
                if self.bcs[zid]["kind"] != "wall":
                    continue
                ok = f >= 0
                for k in range(3):
                    Sn[:, k] += np.bincount(f[ok], S[ok][:, k], N)
                area += np.bincount(f[ok], np.linalg.norm(S, axis=2)[ok], N)
            nodes = np.flatnonzero(area[:self._n_own] > 0)
            n = Sn[nodes] / np.linalg.norm(Sn[nodes], axis=1, keepdims=True)
            share = self._wall_share()[nodes]
            self._wallgeo = (nodes, n, area[nodes], share)
        return self._wallgeo

    def _wall_share(self):
        """Each node's wall fraction of the sub-face area of the zones fixing its velocity."""
        N = self.N
        wall = np.zeros(N)
        tot = np.zeros(N)
        for zid, (f, S) in self.sub.items():
            kind = self.bcs[zid]["kind"]
            if kind not in ("wall", "velocity"):
                continue
            ok = f >= 0
            a = np.bincount(f[ok], np.linalg.norm(S, axis=2)[ok], N)
            tot += a
            if kind == "wall":
                wall += a
        return np.divide(wall, tot, out=np.zeros(N), where=tot > 0)

    def wall_shear(self, method: str = "gradient"):
        """Wall shear stress (Pa) at wall nodes: ``(nodes, τ (n, 3))``, the force per area of
        the fluid on the wall.

        ``method="gradient"`` (default) evaluates it as Ansys CFX does for
        laminar walls: the tangential viscous traction ``μ (∇u + ∇uᵀ) · n`` from
        the owning element's shape-function gradients at each wall sub-face's
        integration point, area-averaged to the node
        (:class:`zvcfd.fv.wall.WallGradientShear`). It needs near-wall
        resolution: on linear tetrahedra the gradient is the element's constant
        (P1) one and reads low where the boundary layer spans few cells;
        prism layers fix that.

        ``method="reaction"`` takes it from consistent reactions instead
        (below). Those close the momentum balance exactly and are what
        :meth:`zone_forces` uses, but locally they absorb the discretisation
        error of the advective flux into the wall control volumes and read
        high where near-wall cells are coarse (+45 % in Poiseuille flow at
        R/h = 2.8; docs/validation/simvascular.md). Use them for forces, not
        for the distribution of wall shear stress.
        """
        if method == "gradient":
            return self._wall_shear_gradient()
        if method != "reaction":
            raise ValueError(f"wall_shear method {method!r}: 'gradient' or 'reaction'")
        return self._wall_shear_reaction()

    def _wall_shear_gradient(self):
        from zvcfd.fv.wall import WallGradientShear

        nodes = self._wall_geometry()[0]
        if getattr(self, "_wallgrad", None) is None:
            walls = [z for z in self.mesh.zones if self.bcs[z]["kind"] == "wall"]
            self._wallgrad = WallGradientShear(self.mesh, walls, xp=self.cp)
        tau = self._wallgrad(self.U, self.fluid.viscosity, self.fluid.mu)
        return nodes, self.cp.asnumpy(tau[self.cp.asarray(nodes)])

    def _wall_shear_reaction(self):
        """Wall shear from consistent reactions.

        The force of the fluid on each wall node is minus the residual of that
        node's momentum equation before the no-slip condition replaced it
        (the boundary flux the discrete equations need; no differentiation at
        the wall). Where the node also borders an inlet, that residual also
        holds the inlet sub-faces' pressure force ``p_j A_in`` (tangential to
        the wall, and large where the pressure level is high) and the
        momentum the inlet carries, so both are removed; the inlet's viscous
        traction, small for inflow profiles, stays. Pressure-zone forces are
        in the assembled rows already. The tangential part over the node's
        wall area is ``τ``.
        """
        nodes, n, area, share = self._wall_geometry()
        r = self.cp.asnumpy(self._raw.residual(self._state())[:, :3])
        F = -r
        P = self.cp.asnumpy(self.P)
        N = self.N
        for zid, (f, S) in self.sub.items():
            if self.bcs[zid]["kind"] != "velocity":
                continue
            ok = f >= 0
            for k in range(3):
                F[:, k] -= np.bincount(f[ok], S[ok][:, k], N) * P
                if not self.stokes:
                    mom = self.sub_flow[zid] * self.sub_vel[zid][..., k]
                    F[:, k] -= np.bincount(f[ok], mom[ok],
                                           N)
        F = F[nodes]
        tau = F - np.einsum("nk,nk->n", F, n)[:, None] * n
        return nodes, tau / area[:, None]

    def _zone_tables(self):
        """Per pressure zone, the (node, weight) pairs of its outflow ``Σ ṁ_b w`` (once)."""
        if getattr(self, "_ztab", None) is None:
            idx, wts, bounds, names = [], [], [0], []
            for zid, (f, S) in self.sub.items():
                if self.bcs[zid]["kind"] != "pressure":
                    continue
                ok = f >= 0
                own = ok & (f < self._n_own)
                share = np.linalg.norm(S, axis=2) * ok
                nodes = f[ok]
                tot = np.bincount(nodes, share[ok], self.N)
                w = np.where(own[ok], share[ok] / np.where(tot[nodes] > 0, tot[nodes], 1), 0)
                idx.append(nodes)
                wts.append(w)
                bounds.append(bounds[-1] + len(nodes))
                names.append(zid)
            cp = self.cp
            self._ztab = (cp.asarray(np.concatenate(idx) if idx else np.zeros(0, np.int64)),
                          cp.asarray(np.concatenate(wts) if wts else np.zeros(0)),
                          np.asarray(bounds), names)
        return self._ztab

    def _zone_contrib(self):
        idx, w, _, _ = self._zone_tables()
        return self._mb[idx] * w

    def _zone_flows_from(self, contrib_host) -> dict[int, float]:
        _, _, bounds, names = self._zone_tables()
        out = {}
        for k, zid in enumerate(names):
            out[zid] = float(contrib_host[bounds[k]:bounds[k + 1]].sum())
        for zid, (f, S) in self.sub.items():
            if self.bcs[zid]["kind"] != "pressure":
                ok = f >= 0
                out[zid] = float((self.sub_flow[zid] * (ok & (f < self._n_own))).sum())
        return {z: out[z] for z in self.sub}

    def zone_flows(self) -> dict[int, float]:
        """Mass outflow (kg/s) through each zone; inflow negative (owned nodes' share)."""
        if self.fast:
            return self._zone_flows_from(self._zone_contrib().get())
        mb = self.cp.asnumpy(self._mb)
        out = {}
        for zid, (f, S) in self.sub.items():
            ok = f >= 0
            own = ok & (f < self._n_own)
            if self.bcs[zid]["kind"] == "pressure":
                share = np.linalg.norm(S, axis=2) * ok
                nodes = f[ok]
                tot = np.bincount(nodes, share[ok], self.N)
                w = np.where(own[ok], share[ok] / np.where(tot[nodes] > 0, tot[nodes], 1), 0)
                out[zid] = float(np.sum(mb[nodes] * w))
            else:
                out[zid] = float((self.sub_flow[zid] * own).sum())
        return out

    def fields(self) -> dict:
        """Node fields on the host: ``{"U": (N, 3), "P": (N,)}``."""
        return {"U": self.cp.asnumpy(self.U), "P": self.cp.asnumpy(self.P)}


class RawRows:
    """The rows of the assembled system kept before boundary conditions replaced them.

    Only the blocks of kept rows are stored (``data`` ``(k, 4, 4)`` at pattern
    positions ``pos``), with the whole right-hand side. Iterating as a
    ``(matrix, rhs)`` pair gives ``(self, b)`` for code that unpacks it.
    """

    def __init__(self, pos, rows, cols, data, b, copy=True):
        self.pos, self.rows, self.cols = pos, rows, cols
        self.data, self.b = (data.copy() if copy else data), b

    def __iter__(self):
        return iter((self, self.b))

    def residual(self, x):
        """``(N, 4)``: ``A x − b`` at the kept rows, zero elsewhere."""
        import cupy as cp

        X = x.reshape(-1, 4)
        r = cp.zeros_like(self.b)
        cp.add.at(r, self.rows, cp.einsum("kij,kj->ki", self.data, X[self.cols]))
        kept = cp.zeros(len(self.b), bool)
        kept[self.rows] = True
        return cp.where(kept[:, None], r - self.b, 0.0)


def residual_summary(parts: list[dict]) -> dict:
    """Combine :meth:`GPUSolver._residual_sums` of one or more partitions: RMS and max."""
    out = {}
    for name in ("u", "v", "w", "p"):
        sq = sum(p[name][0] for p in parts)
        n = sum(p[name][1] for p in parts)
        out[f"rms_{name}"] = float(np.sqrt(sq / n)) if n else 0.0
        out[f"max_{name}"] = max(p[name][2] for p in parts)
    return out


def _nargs(fn) -> int:
    import inspect

    if fn is None:
        return 0
    try:
        return len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return 1


def subface_centroids(mesh: UnstructuredMesh, faces: np.ndarray) -> np.ndarray:
    """Area centroids of each node's share of each face ``(F, 4, 3)`` (for sampling data)."""
    W = boundary_subface_weights(faces)
    return np.einsum("fij,fjk->fik", W, mesh.nodes[np.where(faces >= 0, faces, 0)])


__all__ = ["GPUSolver", "subface_centroids"]
