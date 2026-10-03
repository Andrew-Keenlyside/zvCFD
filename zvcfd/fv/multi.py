"""The finite-volume solver on several GPUs (or several partitions of one).

:class:`MultiGPUSolver` has the interface of :class:`zvcfd.fv.solver.GPUSolver`
and solves the same discrete equations on partitions of the mesh
(:mod:`zvcfd.fv.partition`: whole store chunks along a Morton curve, each
with a one-element halo). Each partition is a
:class:`~zvcfd.fv.solver.GPUSolver` on its local mesh, on its device; this
class runs the outer iteration across them:

1. **Stage 1** on every partition: nodal gradients of ``u`` and ``p``, the
   limiter blend and the momentum diagonal, valid at owned nodes. Then a
   **halo exchange** of all four, since flux points of halo elements need
   them at halo nodes.
2. **Stage 2**: assembly of the owned rows, boundary conditions; the lumped
   outlets' flow ``Q = Σ_p (w_p·x + c_p)`` is summed over partitions, so the
   rank-one pressure–flow coupling is global.
3. **Linear solve**: one FGMRES over the distributed vector, with global
   dot products, halo exchange in every product, and a block-Jacobi
   preconditioner: a V-cycle of zvCFD's ACM (or of AmgX) on each
   partition's owned block. Rows are made dimensionless with global scales.
4. **Update and mass flows**: owned values from the solution, halo
   exchange of ``u`` and ``p``, then of ``∇p`` for the Rhie–Chow mass
   flows, which every partition then evaluates identically on shared
   elements.

Reductions (residuals, imbalance, patch flows, convergence) sum over
owned nodes. With one partition this is the single-GPU solver with its ACM
back end, to round-off; with several, the converged solution is the same
to the solver tolerance, because only the preconditioner changes.

Devices are ``devices[p % len(devices)]``; halo copies between devices are
peer copies through CuPy. Not supported across partitions:
average-static-pressure zones (their mean couples a zone globally).
"""

from __future__ import annotations

import time

import numpy as np

from zvcfd.fv.geometry import dual_geometry
from zvcfd.fv.linear import ACM, AMGX, BlockMatrix, SolveInfo, fgmres
from zvcfd.fv.partition import halo_plan, local_meshes, morton_owner, partition_report
from zvcfd.fv.reference import BDF, BoundaryConditions, Fluid, SolveReport, _call, time_steps
from zvcfd.fv.solver import GPUSolver, _nargs, residual_summary
from zvcfd.mesh.core import UnstructuredMesh


class DVec:
    """A vector split over partitions: one CuPy array per partition, on its device."""

    __array_ufunc__ = None                  # numpy scalars defer to DVec's operators

    def __init__(self, arrays, devices):
        self.a, self.devs = list(arrays), devices

    def _map(self, other, fn):
        import cupy as cp

        out = []
        for i, d in enumerate(self.devs):
            with cp.cuda.Device(d):
                out.append(fn(self.a[i], other.a[i] if isinstance(other, DVec) else other))
        return DVec(out, self.devs)

    def __add__(self, o):
        return self._map(o, lambda a, b: a + b)

    def __sub__(self, o):
        return self._map(o, lambda a, b: a - b)

    def __mul__(self, s):
        return self._map(float(s), lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, s):
        return self._map(float(s), lambda a, b: a / b)

    def __iadd__(self, o):
        self.a = (self + o).a
        return self

    def __isub__(self, o):
        self.a = (self - o).a
        return self

    def copy(self):
        return self._map(None, lambda a, b: a.copy())

    def dot(self, o) -> float:
        import cupy as cp

        tot = 0.0
        for i, d in enumerate(self.devs):
            with cp.cuda.Device(d):
                tot += float(cp.dot(self.a[i], o.a[i]))
        return tot

    def norm(self) -> float:
        return float(np.sqrt(max(self.dot(self), 0.0)))


class _Part(GPUSolver):
    """One partition: a GPU solver on a local mesh, fed boundary data by the global solver."""

    _fusable = False            # the global solver exchanges halos between the stages

    def __init__(self, G, lm, device):
        import cupy as cp

        self.G, self.lm, self.device = G, lm, device
        with cp.cuda.Device(device):
            self._setup(lm.mesh, G.fluid, G.bcs, source=None, advection=G.advection, dt=G.dt,
                        stokes=G.stokes, rhie_chow=G.rhie_chow, transpose=G.transpose,
                        freeze_limiter=G.freeze_limiter, t=G.t, n_own=lm.n_own)
            # whole control volumes: a halo node's local one misses the elements this
            # partition lacks, and Rhie–Chow's V/a_P at shared elements must agree
            self.V = G.V[lm.nodes]
            self._classify(None)
            self._update_boundary(self.t)
            self._build()

    def _classify(self, reference_pressure):
        G, g = self.G, self.lm.nodes
        self.sub = {zid: (z.faces, self.geom.boundary[zid]) for zid, z in self.mesh.zones.items()}
        self._zone_nodes = {zid: np.unique(z.faces[z.faces >= 0])
                            for zid, z in self.mesh.zones.items()}
        for name in ("wall", "vel_fixed", "vel_fixed_comp", "sym", "outlet", "p_fixed"):
            setattr(self, name, getattr(G, name)[g])
        self._pin = None

    def _update_boundary(self, t):
        """Slices of the global boundary data (the global solver updates it first)."""
        G, g, fm = self.G, self.lm.nodes, self.lm.faces
        self.vel_value = G.vel_value[g]
        self.p_value = G.p_value[g]
        self.inflow = G.inflow[g]
        self.f = G.f[g]
        self.sub_flow = {z: G.sub_flow[z][fm[z]] for z in G.sub_flow}
        self.sub_vel = {z: G.sub_vel[z][fm[z]] for z in G.sub_vel}
        self._outlet = {z: (p[fm[z]], None if gr is None else gr[fm[z]])
                        for z, (p, gr) in G._outlet.items()}

    def lumped_coefficients(self, zid):
        return self.G.lumped_coefficients(zid)


class MultiGPUSolver(BoundaryConditions):
    """Coupled finite-volume solve on partitions of the mesh (see the module docstring).

    Args:
        mesh, fluid, bcs, source, advection, dt, reference_pressure, stokes,
            rhie_chow, transpose, freeze_limiter, t: as :class:`GPUSolver`.
        parts: number of partitions.
        devices: CUDA device ids (partition ``p`` on ``devices[p % len]``).
        chunk: partition chunk edge (mesh units); default about 64 chunks per part.
        linear: ``"acm"`` or ``"amgx"``: the per-partition preconditioner.
        linear_rtol, restart, maxiter: of the global FGMRES.
    """

    def __init__(self, mesh: UnstructuredMesh, fluid: Fluid, bcs: dict, *, parts: int = 2,
                 devices=None, chunk: float | None = None, source=None,
                 advection="high-resolution", dt: float | None = None,
                 reference_pressure=None, stokes: bool = False, rhie_chow: float = 1.0,
                 transpose="auto", freeze_limiter: int | None = 10, t: float = 0.0,
                 linear: str = "acm", linear_rtol: float = 0.1, restart: int = 30,
                 maxiter: int = 200, owner: np.ndarray | None = None):
        import cupy as cp

        t0 = time.time()
        for zid, spec in bcs.items():
            if spec.get("profile") == "average":
                raise NotImplementedError(f"zone {zid}: average-pressure zones need one partition")
        self.cp = cp
        self.mesh, self.fluid, self.bcs, self.source = mesh, fluid, bcs, source
        self.advection, self.dt, self.stokes = advection, dt, stokes
        self.rhie_chow, self.freeze_limiter = rhie_chow, freeze_limiter
        self.transpose = (fluid.viscosity is not None) if transpose == "auto" else bool(transpose)
        self.t = float(t)
        self.N = mesh.n_nodes
        self.geom = dual_geometry(mesh, keep_areas=False)
        self.V = self.geom.node_volume
        self._classify(reference_pressure)
        self._update_boundary(self.t)
        self.owner = morton_owner(mesh.nodes, parts, chunk) if owner is None else owner
        self.local = local_meshes(mesh, self.owner)
        self.plan = halo_plan(self.local, self.owner)
        devices = list(devices) if devices is not None else [cp.cuda.Device().id]
        self.devs = [devices[p % len(devices)] for p in range(len(self.local))]
        self.parts = [_Part(self, lm, d) for lm, d in zip(self.local, self.devs)]
        self._time = None
        self.iteration = 0
        self.linear_name, self.linear_rtol = linear, linear_rtol
        self.restart, self.maxiter = restart, maxiter
        self._pre = []
        for p in self.parts:
            with cp.cuda.Device(p.device):
                p.linear = ACM() if linear == "acm" else AMGX() if linear == "amgx" else None
                if p.linear is None:
                    raise ValueError(f"linear {linear!r}: acm or amgx per partition")
                self._pre.append(_owned_block(p))
        self.linear = type("L", (), {"name": f"{linear}×{len(self.parts)}"})()
        self._initial_pressure()
        for p in self.parts:
            with cp.cuda.Device(p.device):
                p._apply_fixed()
        self.setup_seconds = time.time() - t0
        self.report = partition_report(self.local)

    # ------------------------------------------------------------ plumbing

    def _each(self, fn):
        import cupy as cp

        out = []
        for p in self.parts:
            with cp.cuda.Device(p.device):
                out.append(fn(p))
        return out

    def exchange(self, arrays: list) -> None:
        """Fill each partition's halo rows of ``arrays[p]`` from their owners (in place)."""
        import cupy as cp

        for p, plan in enumerate(self.plan):
            with cp.cuda.Device(self.devs[p]):
                for q, send, recv in plan:
                    src = arrays[q][send] if self.devs[q] == self.devs[p] else \
                        cp.asarray(arrays[q][send])
                    arrays[p][recv] = src

    def _local_x(self, x: DVec):
        """Local vectors (owned + halo, flat 4 dofs per node) of a distributed owned vector."""
        import cupy as cp

        xl = []
        for p, part in enumerate(self.parts):
            with cp.cuda.Device(part.device):
                v = cp.zeros((part.N, 4))
                v[:part._n_own] = x.a[p].reshape(-1, 4)
                xl.append(v)
        self.exchange(xl)
        return [v.reshape(-1) for v in xl]

    def lumped_coefficients(self, zid) -> tuple[float, float]:
        model = self.bcs[zid]["lumped"]
        if self._time is None:
            return model.steady_coefficients(self.t)
        return model.coefficients(self.t, self._time[1])

    def _lumped_zones(self):
        return [z for z, s in self.bcs.items() if s["kind"] == "pressure" and s.get("lumped")]

    def _initial_pressure(self):
        """The boundary pressure level, as :meth:`GPUSolver._initial_pressure`, globally."""
        rho = self.fluid.rho
        q_in = -sum(float(v.sum()) for v in self.sub_flow.values()) / rho
        lumped = {z: self.lumped_coefficients(z) for z in self._lumped_zones()}
        g = sum(1.0 / r for _, r in lumped.values() if r > 0)
        num = den = 0.0
        vals = {}
        for zid, (f, S) in self.sub.items():
            if self.bcs[zid]["kind"] != "pressure":
                continue
            ok = f >= 0
            area = float((np.linalg.norm(S, axis=2) * ok).sum())
            if zid in lumped:
                a, r = lumped[zid]
                val = a + (max(q_in, 0.0) / g if r > 0 and g > 0 else 0.0)
                vals[zid] = val
            else:
                val = float(self.p_value[np.unique(f[ok])].mean())
            num += val * area
            den += area
        for p in self.parts:
            with self.cp.cuda.Device(p.device):
                if den > 0:
                    p.P[:] = num / den
                for zid, v in vals.items():
                    if zid in p._special:
                        p._pval_d[p._special[zid]["nodes"]] = v

    # ------------------------------------------------------------ one outer iteration

    def _assemble_all(self):
        """Assemble every partition; returns the distributed system (unscaled)."""
        pre = self._each(lambda p: p._assemble_stage1())
        for key in ("gradU", "gradP", "beta", "diag"):
            arrs = [d[key] for d in pre]
            if all(hasattr(a, "shape") and a.shape[0] == p.N for a, p in zip(arrs, self.parts)):
                self.exchange(arrs)
        sys_ = self._each(lambda p: p._assemble_stage2(pre[self.parts.index(p)]))
        pieces = self._each(lambda p: p._lumped_pieces())
        c_total = {z: sum(pc[z][2] for pc in pieces) for z in self._lumped_zones()}
        coef = {z: self.lumped_coefficients(z) for z in self._lumped_zones()}
        for p, (A, b), pc in zip(self.parts, sys_, pieces):
            with self.cp.cuda.Device(p.device):
                p._finish_lumped(A, b, pc, c_total, coef)
        # the global rank-one term of each lumped zone: rows in every part, w summed over parts
        lowrank = []
        for z in self._lumped_zones():
            terms = []
            for p, pc in zip(self.parts, pieces):
                sp = p._special[z]
                with self.cp.cuda.Device(p.device):
                    terms.append((4 * sp["nodes"] + 3,
                                  self.cp.full(sp["nodes"].size, -sp["sc"] * sp["r"]),
                                  pc[z][0], pc[z][1]))
            lowrank.append(terms)
        return [s[0] for s in sys_], [s[1] for s in sys_], lowrank

    def _matvec(self, As, lowrank, x: DVec) -> DVec:
        xl = self._local_x(x)
        ys = []
        for p, (part, A) in enumerate(zip(self.parts, As)):
            with self.cp.cuda.Device(part.device):
                ys.append(A.matvec(xl[p])[:4 * part._n_own])
        for terms in lowrank:
            s = 0.0
            for p, (ui, uv, wi, wv) in enumerate(terms):
                with self.cp.cuda.Device(self.devs[p]):
                    s += float(self.cp.dot(wv, xl[p][wi]))
            for p, (ui, uv, wi, wv) in enumerate(terms):
                with self.cp.cuda.Device(self.devs[p]):
                    ys[p][ui] += uv * s
        return DVec(ys, self.devs)

    def _state(self) -> DVec:
        return DVec(self._each(lambda p: p._state()[:4 * p._n_own]), self.devs)

    def _iterate(self, max_iterations, tol, log, hist, residual_target=None):
        cp = self.cp
        converged = False
        it = 0
        for it in range(1, max_iterations + 1):
            self.iteration = it - 1
            for p in self.parts:
                p.iteration = it - 1
            t0 = time.time()
            As, bs, lowrank = self._assemble_all()
            x0 = self._state()
            b = DVec([bb.reshape(-1)[:4 * p._n_own] for bb, p in zip(bs, self.parts)], self.devs)
            r = b - self._matvec(As, lowrank, x0)
            sums = []
            for p, rp in zip(self.parts, r.a):
                with cp.cuda.Device(p.device):
                    rr = cp.zeros((p.N, 4))
                    rr[:p._n_own] = rp.reshape(-1, 4)
                    sums.append(p._residual_sums(rr, self._uref()))
            res = residual_summary(sums)
            scales = self._scales()
            for p, A, bb, terms in zip(self.parts, As, bs, zip(*lowrank) if lowrank else
                                       [()] * len(self.parts)):
                with cp.cuda.Device(p.device):
                    rs = p._scale_rows(A, bb, scales).reshape(-1)
                    for term in terms:
                        term[1][...] = term[1] * rs[term[0]]
            b = DVec([bb.reshape(-1)[:4 * p._n_own] for bb, p in zip(bs, self.parts)], self.devs)
            t_asm = time.time() - t0
            x, info = self._solve(As, lowrank, b, x0)
            du = dp = 0.0
            umax, pmin, pmax = 1e-300, np.inf, -np.inf
            for p, xp in zip(self.parts, x.a):
                with cp.cuda.Device(p.device):
                    X = xp.reshape(-1, 4)
                    n = p._n_own
                    du = max(du, float(cp.abs(X[:, :3] - p.U[:n]).max()))
                    dp = max(dp, float(cp.abs(X[:, 3] - p.P[:n]).max()))
                    umax = max(umax, float(cp.abs(X[:, :3]).max()))
                    pmin, pmax = min(pmin, float(X[:, 3].min())), max(pmax, float(X[:, 3].max()))
                    p.U[:n] = X[:, :3]
                    p.P[:n] = X[:, 3]
            du /= umax
            dp /= max(pmax - pmin, 1e-300)
            self.exchange([p.U for p in self.parts])
            self.exchange([p.P for p in self.parts])
            self.update_mass_flows()
            rec = {"iteration": it, "du": du, "dp": dp, "linear_iterations": info.iterations,
                   "linear_residual": info.residual, "linear_converged": bool(info.converged),
                   "assemble_s": t_asm, "linear_setup_s": info.setup_seconds,
                   "linear_solve_s": info.seconds, "seconds": time.time() - t0, **res}
            q = self.zone_flows()
            gross = sum(abs(v) for v in q.values()) or 1.0
            rec["imbalance"] = sum(q.values()) / gross
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

    def _uref(self) -> float:
        return max(max(self._each(lambda p: float(self.cp.abs(p.U[:p._n_own]).max()))), 1e-30)

    def _scales(self):
        cp = self.cp
        uref = max(self._uref(), max(self._each(lambda p: float(cp.abs(p._vel_d).max()))), 1e-30)
        lo = min(self._each(lambda p: float(p.P[:p._n_own].min())))
        hi = max(self._each(lambda p: float(p.P[:p._n_own].max())))
        pref = max(hi - lo, self.fluid.rho * uref * uref, 1e-30)
        zones = {}
        for z in self._lumped_zones():
            zones[z] = (sum(float(p._special[z]["area"].sum()) for p in self.parts),
                        sum(int(p._special[z]["nodes"].size) for p in self.parts))
        return uref, pref, zones

    def _solve(self, As, lowrank, b: DVec, x0: DVec):
        """Global FGMRES, block-Jacobi preconditioned by per-partition V-cycles."""
        cp = self.cp
        t = time.time()
        for p, (pos, indptr, indices), A in zip(self.parts, self._pre, As):
            with cp.cuda.Device(p.device):
                own = BlockMatrix(indptr, indices, A.data[pos])
                if isinstance(p.linear, ACM):
                    p.linear.setup(own)
                else:
                    p.linear._upload(own)
                    p.linear._solver("precond")

        def precond(r: DVec) -> DVec:
            out = []
            for p, rp in zip(self.parts, r.a):
                with cp.cuda.Device(p.device):
                    out.append(p.linear.vcycle(rp))
            return DVec(out, self.devs)

        ts = time.time() - t
        t = time.time()
        x, it, res, ok = fgmres(_DistOp(self, As, lowrank), b, x0, precond,
                                rtol=self.linear_rtol, restart=self.restart,
                                maxiter=self.maxiter, norm=DVec.norm, dot=DVec.dot)
        return x, SolveInfo(it, res, time.time() - t, ts, ok)

    def update_mass_flows(self):
        grads = self._each(lambda p: p._mirror_d(p.asm.gradient(p.P)))
        self.exchange(grads)
        for p, g in zip(self.parts, grads):
            with self.cp.cuda.Device(p.device):
                p._massflow_from(g)

    # ------------------------------------------------------------ public interface

    turbulence = None          # eddy viscosity and turbulence models: one GPU for now
    mu_t = None

    def _laminar_only(self):
        if self.turbulence is not None or self.mu_t is not None:
            raise NotImplementedError("eddy viscosity and turbulence models run on one GPU "
                                      "(GPUSolver) for now")

    def solve(self, *, max_iterations: int = 200, tol: float = 1e-10, log=None,
              residual_target: float | None = None) -> SolveReport:
        self._laminar_only()
        t0 = time.time()
        hist = []
        it, conv = self._iterate(max_iterations, tol, log, hist, residual_target)
        q = self.patch_flows()
        for z in self._lumped_zones():
            self.bcs[z]["lumped"].initialise(q[z], self.t)
        return SolveReport(it, conv, hist, time.time() - t0)

    def initialise(self, U=None, P=None) -> None:
        cp = self.cp
        for p in self.parts:
            with cp.cuda.Device(p.device):
                x = p.mesh.nodes
                if U is not None:
                    p.U = cp.asarray(np.array(_call(U, x, self.t), float).reshape(p.N, 3))
                if P is not None:
                    p.P = cp.asarray(np.array(_call(P, x, self.t), float).reshape(p.N))
                p._apply_fixed()
        self._assemble_all()
        self.update_mass_flows()

    def solve_transient(self, dt, steps: int | None = None, *, scheme: str = "bdf2",
                        loops: int = 5, tol: float = 1e-8, callback=None,
                        log=None, residual_target: float | None = None) -> SolveReport:
        """As :meth:`GPUSolver.solve_transient`, over the partitions."""
        self._laminar_only()
        cp = self.cp
        if scheme not in BDF:
            raise ValueError(f"scheme {scheme!r}")
        t0 = time.time()
        hist = []

        def level(p):
            return (p.U.copy(), {k: v.copy() for k, v in p.mdot.items()})

        old = [[lv] for lv in self._each(level)]
        total, dt_prev, all_ok = 0, None, True
        nargs = _nargs(callback)
        for dt in time_steps(dt, steps, self.t):
            self.t += dt
            self._time = (scheme, dt, None, dt_prev)
            self._update_boundary(self.t)
            for p, o in zip(self.parts, old):
                with cp.cuda.Device(p.device):
                    p.t = self.t
                    p._time = (scheme, dt, o, dt_prev)
                    p._update_boundary(self.t)
                    p._upload_boundary()
                    p._apply_fixed()
            dt_prev = dt
            n, ok = self._iterate(loops, tol, log, hist, residual_target)
            all_ok &= ok
            total += n
            hist[-1]["t"] = self.t
            q = self.patch_flows()
            for z in self._lumped_zones():
                self.bcs[z]["lumped"].advance(q[z], self.t, dt)
            old = [[lv] + o[:1] for lv, o in zip(self._each(level), old)]
            if callback is not None:
                callback(self, dt) if nargs >= 2 else callback(self)
        self._time = None
        for p in self.parts:
            p._time = None
        return SolveReport(total, all_ok, hist, time.time() - t0)

    def zone_flows(self) -> dict[int, float]:
        out = {}
        for d in self._each(lambda p: p.zone_flows()):
            for z, v in d.items():
                out[z] = out.get(z, 0.0) + v
        return out

    def patch_flows(self) -> dict[int, float]:
        return {z: v / self.fluid.rho for z, v in self.zone_flows().items()}

    def fields(self) -> dict:
        """Node fields on the host, gathered: ``{"U": (N, 3), "P": (N,)}``."""
        U, P = np.zeros((self.N, 3)), np.zeros(self.N)
        for p in self.parts:
            f = p.fields()
            g = p.lm.nodes[:p._n_own]
            U[g], P[g] = f["U"][:p._n_own], f["P"][:p._n_own]
        return {"U": U, "P": P}

    def patch_pressures(self) -> dict[int, float]:
        P = self.fields()["P"]
        out = {}
        for zid, (f, S) in self.sub.items():
            ok = f >= 0
            a = np.linalg.norm(S, axis=2) * ok
            out[zid] = float((P[np.where(ok, f, 0)] * a).sum() / a.sum())
        return out

    def zone_forces(self) -> dict[int, np.ndarray]:
        out = {}
        for d in self._each(lambda p: p.zone_forces()):
            for z, v in d.items():
                out[z] = out.get(z, 0.0) + v
        return out

    def wall_shear(self, method: str = "gradient"):
        """``(global wall nodes, τ (n, 3))``, gathered from the partitions
        (:meth:`GPUSolver.wall_shear`)."""
        nodes, taus = [], []
        for p in self.parts:
            with self.cp.cuda.Device(p.device):
                n, tau = p.wall_shear(method)
            nodes.append(p.lm.nodes[n])
            taus.append(tau)
        nodes, tau = np.concatenate(nodes), np.concatenate(taus)
        order = np.argsort(nodes)
        return nodes[order], tau[order]


class _DistOp:
    """``A`` for :func:`zvcfd.fv.linear.fgmres`: the distributed matrix-vector product."""

    def __init__(self, solver, As, lowrank):
        self.s, self.As, self.lowrank = solver, As, lowrank

    def matvec(self, x):
        return self.s._matvec(self.As, self.lowrank, x)


def _owned_block(p) -> tuple:
    """``(block positions, indptr, indices)`` of a partition's owned-rows × owned-columns block."""
    cp = p.cp
    n = p._n_own
    rows, cols = p._rows, p.pattern[1].astype(cp.int64)
    pos = cp.flatnonzero((rows < n) & (cols < n))
    counts = cp.bincount(rows[pos], minlength=n)
    indptr = cp.concatenate([cp.zeros(1, cp.int64), cp.cumsum(counts)]).astype(cp.int64)
    return pos, indptr, p.pattern[1][pos]


__all__ = ["DVec", "MultiGPUSolver"]
