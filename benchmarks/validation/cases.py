"""The validation cases: each builds a lattice problem with a known answer, runs it, and scores it.

Every case function returns a dict of plain numbers (JSON-ready) with one
row per resolution and the observed order of convergence where it is
meaningful. Lattice units throughout; TRT with Lambda = 3/16, fp32.
"""

from __future__ import annotations

import time

import exact
import numpy as np

RHO0 = 1.0


# ---------------------------------------------------------------- helpers

def _pad8(n: int) -> int:
    return int(np.ceil(n / 8) * 8)


def to_bricks(domain, vol: np.ndarray) -> np.ndarray:
    """Dense ``(z, y, x)`` values to ``(n_bricks, 512)`` rows in domain order."""
    b = domain.brick
    bz, by, bx = domain.brick_grid
    v = np.asarray(vol).reshape(bz, b, by, b, bx, b).transpose(0, 2, 4, 1, 3, 5)
    c = domain.coords
    return v[c[:, 0], c[:, 1], c[:, 2]].reshape(len(c), b ** 3)


def equilibrium_init(domain, rho, ux, uy=None, uz=None) -> np.ndarray:
    """``(Q, n)`` equilibrium populations from dense fields, for ``SparseLBM(init=...)``."""
    from zvcfd.lbm.solver import equilibrium

    z = np.zeros(domain.shape)
    rows = [to_bricks(domain, np.broadcast_to(v if v is not None else z, domain.shape)).reshape(-1)
            for v in (rho, ux, uy, uz)]
    return equilibrium(*rows)


def dense(sim, domain, key: str) -> np.ndarray:
    return domain.to_dense(sim.fields()[key].get())


def drho_dense(sim, domain) -> np.ndarray:
    """Density deviation rho - 1 summed in float64 from the shifted populations.

    ``fields()["rho"]`` is ``1 + drho`` in fp32, which quantises a 10^-6
    density step at 1.2e-7; the populations themselves hold it exactly.
    """
    from zvcfd.lbm._cuda import Q

    g = sim.f0.reshape(Q, -1).astype("float64").sum(0).get()
    return domain.to_dense(g.reshape(-1, domain.brick ** 3))


def order(ns, errs) -> float:
    """Least-squares slope of log(error) against log(resolution), sign flipped."""
    ns, errs = np.asarray(ns, float), np.asarray(errs, float)
    return float(-np.polyfit(np.log(ns), np.log(errs), 1)[0])


def run_steady(sim, monitor, *, check=500, tol=1e-6, max_steps=2_000_000, min_steps=0):
    """Step until ``monitor`` changes by less than ``tol`` (relative) between checks.

    fp32 fields resolve ~1e-7 relative, so ``tol`` alone can stop on a stall:
    ``min_steps`` (several diffusion times) is the other half of the criterion.
    """
    prev, t0 = None, time.time()
    while sim.steps < max_steps:
        sim.step(check)
        m = monitor()
        if prev is not None and sim.steps >= min_steps and abs(m - prev) <= tol * abs(m):
            break
        prev = m
    return {"steps": int(sim.steps), "wall_s": time.time() - t0}


def _l2(a, b):
    return float(np.sqrt(((a - b) ** 2).sum() / (b ** 2).sum()))


def _linf(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


# ---------------------------------------------------------------- 1. plane Poiseuille

def poiseuille(widths=(6, 14, 30), taus=(0.55, 0.8, 2.0)) -> dict:
    """Force-driven plane channel, periodic in x and y: exact for TRT, Lambda = 3/16, at any tau."""
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM

    rows = []
    for H in widths:
        nz = _pad8(H + 2)
        flags = np.ones((nz, 8, 8), np.uint8)
        flags[1:H + 1] = 0
        dom = BrickDomain.from_flags(flags, periodic=(False, True, True))
        zeta = np.arange(1, H + 1) - 0.5
        for tau in taus:
            nu = (tau - 0.5) / 3
            F = 0.01 * 2 * nu / (H / 2) ** 2          # centre-line velocity 0.01
            sim = SparseLBM(dom, tau=tau, force=(F, 0.0, 0.0), collision="trt")
            info = run_steady(sim, lambda: float(dense(sim, dom, "ux")[H // 2 + 1, 4, 4]),
                              check=max(200, int(H * H / nu / 20)),
                              min_steps=int(15 * H * H / (np.pi ** 2 * nu)))
            u = dense(sim, dom, "ux")[1:H + 1, 4, 4]
            ex = exact.plane_poiseuille(zeta, H, F, nu)
            rows.append({"H": H, "tau": tau, "linf": _linf(u, ex), "l2": _l2(u, ex), **info})
    return {"case": "poiseuille", "rows": rows}


def poiseuille_pressure(H=30, nx=64, taus=(0.6, 1.0, 2.0), drho=1e-4) -> dict:
    """Plane channel driven by pressure patches at the x faces (non-equilibrium extrapolation)."""
    from zvcfd.boundary import Patch, face_patches
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM

    nz = _pad8(H + 2)
    flags = np.ones((nz, 8, nx), np.uint8)
    flags[1:H + 1] = 0
    rows = []
    for tau in taus:
        dom = BrickDomain.from_flags(flags, periodic=(False, True, False))
        b = face_patches(dom, {"xmin": Patch("in", "pressure"), "xmax": Patch("out", "pressure")})
        sim = SparseLBM(dom, tau=tau, collision="trt", boundary=b)
        sim.set_patch(0, rho=1 + drho / 2)
        sim.set_patch(1, rho=1 - drho / 2)
        nu = (tau - 0.5) / 3
        info = run_steady(sim, lambda: float(sim.patch_flux()[1]), check=1000,
                          min_steps=int(15 * H * H / (np.pi ** 2 * nu)))
        ux = dense(sim, dom, "ux")
        dr = drho_dense(sim, dom)
        mid = nx // 2
        u = ux[1:H + 1, 4, mid]
        # local gradient from the density field (weakly compressible: rho u is constant along x)
        G = -(dr[1:H + 1, 4, mid + 1] - dr[1:H + 1, 4, mid - 1]).mean() / 2 / 3
        ex = exact.plane_poiseuille(np.arange(1, H + 1) - 0.5, H,
                                    G / (1 + dr[1:H + 1, 4, mid].mean()), nu)
        G_nom = drho / 3 / (nx - 1)
        rows.append({"H": H, "tau": tau, "linf": _linf(u, ex), "l2": _l2(u, ex),
                     "grad_vs_nominal": float(G / G_nom), **info})
    return {"case": "poiseuille_pressure", "rows": rows}


# ---------------------------------------------------------------- 2. square duct

def duct(sides=(6, 14, 30, 62), tau=0.8) -> dict:
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM

    nu = (tau - 0.5) / 3
    rows = []
    for h in sides:
        n = _pad8(h + 2)
        flags = np.ones((n, n, 8), np.uint8)
        flags[1:h + 1, 1:h + 1] = 0
        dom = BrickDomain.from_flags(flags, periodic=(False, False, True))
        a = h / 2
        F = 0.01 * nu / a ** 2 * 3.4       # centre-line velocity ~0.01
        sim = SparseLBM(dom, tau=tau, force=(F, 0.0, 0.0), collision="trt")
        info = run_steady(sim, lambda: float(dense(sim, dom, "ux")[h // 2 + 1, h // 2 + 1, 4]),
                          check=max(200, int(h * h / nu / 20)),
                          min_steps=int(15 * h * h / (2 * np.pi ** 2 * nu)))
        u = dense(sim, dom, "ux")[1:h + 1, 1:h + 1, 4]
        c = np.arange(1, h + 1) - 0.5 - a
        Z, Y = np.meshgrid(c, c, indexing="ij")
        ex = exact.square_duct(Y, Z, a, F, nu)
        q = float(u.sum()) / exact.square_duct_flux(a, F, nu)
        rows.append({"side": h, "l2": _l2(u, ex), "linf": _linf(u, ex), "flux_ratio": q, **info})
    return {"case": "duct", "tau": tau, "rows": rows,
            "order_l2": order([r["side"] for r in rows], [r["l2"] for r in rows])}


# ---------------------------------------------------------------- 3. Hagen-Poiseuille

def _pipe_flags(R: int, nx: int = 8):
    n = _pad8(2 * R + 2)
    c = n / 2
    j = np.arange(n) + 0.5
    Zc, Yc = np.meshgrid(j - c, j - c, indexing="ij")
    r = np.sqrt(Zc ** 2 + Yc ** 2)
    disk = r <= R
    flags = np.ones((n, n, nx), np.uint8)
    flags[disk] = 0
    return flags, r, disk


def pipe(radii=(4, 8, 16, 32), tau=0.8) -> dict:
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM

    nu = (tau - 0.5) / 3
    rows = []
    for R in radii:
        flags, r, disk = _pipe_flags(R)
        dom = BrickDomain.from_flags(flags, periodic=(False, False, True))
        F = 0.01 * 4 * nu / R ** 2
        sim = SparseLBM(dom, tau=tau, force=(F, 0.0, 0.0), collision="trt")
        n = flags.shape[0]
        info = run_steady(sim, lambda: float(dense(sim, dom, "ux")[n // 2, n // 2, 4]),
                          check=max(200, int(R * R / nu / 10)),
                          min_steps=int(15 * R * R / (5.78 * nu)))
        u = dense(sim, dom, "ux")[:, :, 4][disk]
        ex = exact.hagen_poiseuille(r[disk], R, F, nu)
        q_exact = np.pi * R ** 4 * F / (8 * nu)
        rows.append({"R": R, "l2": _l2(u, ex), "linf": _linf(u, ex),
                     "flux_ratio": float(u.sum() / q_exact), **info})
    return {"case": "pipe", "tau": tau, "rows": rows,
            "order_l2": order([r["R"] for r in rows], [r["l2"] for r in rows])}


# ---------------------------------------------------------------- 4. Womersley

def _womersley_run(sim, dom, profile_fn, exact_fn, F0, omega, T, tau_decay, *, samples=16,
                   keep=False):
    """Drive with F0 cos(omega t); measure over the last period after the transient has decayed.

    ``keep`` stores the sampled profiles (simulated and exact) for figures.
    """
    periods = int(np.ceil(max(6 * T, 12 * tau_decay) / T))
    n_total = periods * T
    sample_at = set(n_total - T + (np.arange(samples) * T) // samples)
    num = den = 0.0
    worst = 0.0
    kept = []
    t0 = time.time()
    for step in range(n_total):
        sim.force = (np.float32(F0 * np.cos(omega * step)), np.float32(0.0), np.float32(0.0))
        sim.step(1)
        if step in sample_at:
            u = profile_fn()
            ex = exact_fn(step)
            num += ((u - ex) ** 2).sum()
            den += (ex ** 2).sum()
            worst = max(worst, float(np.abs(u - ex).max()))
            if keep:
                kept.append({"phase": float((step - (n_total - T)) / T), "u": u.tolist(),
                             "exact": ex.tolist()})
    amp = max(abs(exact_fn(n_total - T + k * T // samples)).max() for k in range(samples))
    out = {"l2": float(np.sqrt(num / den)), "linf": worst / float(amp), "steps": n_total,
           "period": int(T), "wall_s": time.time() - t0}
    if keep:
        out["profiles"] = kept
    return out


def womersley(half_widths=(8, 16, 32), radii=(8, 16, 32), alphas=(4.0, 12.0), tau=0.8) -> dict:
    """Pulsatile flow: plane channel (cosh) and pipe (Bessel J0), exact; diffusive scaling."""
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM

    nu = (tau - 0.5) / 3
    plane, piperows = [], []
    for alpha in alphas:
        for h in half_widths:
            H = 2 * h
            nz = _pad8(H + 2)
            flags = np.ones((nz, 8, 8), np.uint8)
            flags[1:H + 1] = 0
            dom = BrickDomain.from_flags(flags, periodic=(False, True, True))
            omega = alpha ** 2 * nu / h ** 2
            T = int(round(2 * np.pi / omega))
            omega = 2 * np.pi / T                     # whole number of steps per period
            F0 = 0.01 / max(1 / omega, h * h / (2 * nu))
            sim = SparseLBM(dom, tau=tau, force=(F0, 0.0, 0.0), collision="trt")
            zeta = np.arange(1, H + 1) - 0.5
            r = _womersley_run(sim, dom, lambda: dense(sim, dom, "ux")[1:H + 1, 4, 4],
                               lambda s: exact.womersley_plane(zeta, H, F0, omega, nu, s),
                               F0, omega, T, 4 * h * h / (np.pi ** 2 * nu), keep=h == 16)
            plane.append({"alpha": alpha, "h": h, "zeta": zeta.tolist() if h == 16 else None,
                          "F0": F0, "omega": omega, **r})
        for R in radii:
            flags, rr, disk = _pipe_flags(R)
            dom = BrickDomain.from_flags(flags, periodic=(False, False, True))
            omega = alpha ** 2 * nu / R ** 2
            T = int(round(2 * np.pi / omega))
            omega = 2 * np.pi / T
            F0 = 0.01 / max(1 / omega, R * R / (4 * nu))
            sim = SparseLBM(dom, tau=tau, force=(F0, 0.0, 0.0), collision="trt")
            r = _womersley_run(sim, dom, lambda: dense(sim, dom, "ux")[:, :, 4][disk],
                               lambda s: exact.womersley_pipe(rr[disk], R, F0, omega, nu, s),
                               F0, omega, T, R * R / (5.78 * nu))
            piperows.append({"alpha": alpha, "R": R, **r})
    out = {"case": "womersley", "tau": tau, "plane": plane, "pipe": piperows, "order": {}}
    for alpha in alphas:
        p = [r for r in plane if r["alpha"] == alpha]
        q = [r for r in piperows if r["alpha"] == alpha]
        if len(p) > 1:
            out["order"][f"plane_alpha{alpha:g}"] = order([r["h"] for r in p], [r["l2"] for r in p])
        if len(q) > 1:
            out["order"][f"pipe_alpha{alpha:g}"] = order([r["R"] for r in q], [r["l2"] for r in q])
    return out


# ---------------------------------------------------------------- 5. Taylor-Green vortex

def taylor_green(sizes=(16, 32, 64, 128, 256), tau=0.8, u16=0.04) -> dict:
    """Decaying 2-D Taylor–Green vortex, diffusive scaling (Re fixed); errors at one decay time.

    Two starts: plain equilibrium, and a consistent one (Mei, Luo, Lallemand
    & d'Humières, Comput. Fluids 35, 855, 2006) that adds

    - the first-order non-equilibrium part ``-3 tau w rho (c c : grad u)``,
      times ``1 - 1/tau`` because the buffers hold post-collision populations;
    - the small divergence a weakly compressible solver needs while the
      pressure decays: continuity asks ``div u = -(1/rho) d rho/dt =
      12 nu k^2 p``, met by ``u += -3 nu grad p``. Without it the start
      launches a sound wave in the pressure mode of relative size
      ``~2 nu k / c_s``, a first-order pressure error.
    """
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM
    from zvcfd.lbm._cuda import CX, CY, W

    nu = (tau - 0.5) / 3
    rows = []
    for start in ("equilibrium", "consistent"):
        for N in sizes:
            U = u16 * 16 / N
            flags = np.zeros((8, N, N), np.uint8)
            dom = BrickDomain.from_flags(flags, periodic=True)
            c = np.arange(N) + 0.5
            Y, X = np.meshgrid(c, c, indexing="ij")
            ux, uy, p = exact.taylor_green(X, Y, N, U, nu, 0.0)
            rho = RHO0 + 3 * p
            if start == "consistent":
                k = 2 * np.pi / N
                ux = ux - 3 * nu * (U * U * k / 2) * np.sin(2 * k * X)     # -3 nu dp/dx
                uy = uy - 3 * nu * (U * U * k / 2) * np.sin(2 * k * Y)
            init = equilibrium_init(dom, rho[None], ux[None], uy[None], None)
            if start == "consistent":
                dxux = U * k * np.sin(k * X) * np.sin(k * Y)
                dyuy = -dxux
                dyux = -U * k * np.cos(k * X) * np.cos(k * Y)
                dxuy = -dyux
                for q in range(len(W)):
                    s_q = CX[q] ** 2 * dxux + CY[q] ** 2 * dyuy + CX[q] * CY[q] * (dxuy + dyux)
                    # the buffer holds post-collision populations: (1 - omega) f_neq
                    fneq = (1 - 1 / tau) * -3 * tau * W[q] * rho * s_q
                    init[q] += to_bricks(dom, np.broadcast_to(fneq[None], dom.shape)).reshape(-1)
            sim = SparseLBM(dom, tau=tau, collision="trt", init=init)
            k = 2 * np.pi / N
            steps = int(round(1 / (2 * nu * k * k)))
            t0 = time.time()
            sim.step(steps)
            wall = time.time() - t0
            ex_x, ex_y, ex_p = exact.taylor_green(X, Y, N, U, nu, steps)
            gx, gy = dense(sim, dom, "ux")[4], dense(sim, dom, "uy")[4]
            err_u = float(np.sqrt((((gx - ex_x) ** 2 + (gy - ex_y) ** 2).sum())
                                  / ((ex_x ** 2 + ex_y ** 2).sum())))
            pr = drho_dense(sim, dom)[4] / 3
            pr -= pr.mean()
            rows.append({"start": start, "N": N, "U": U, "steps": steps, "l2_u": err_u,
                         "l2_p": _l2(pr, ex_p), "re": U * N / nu, "wall_s": wall})
    # N = 256 reaches the fp32 round-off floor (~1e-5 in u at U = 0.0025): fit below it
    out = {"case": "taylor_green", "tau": tau, "rows": rows, "fit_range_N": [16, 128]}
    for start in ("equilibrium", "consistent"):
        r = [x for x in rows if x["start"] == start and x["N"] <= 128]
        out[f"order_u_{start}"] = order([x["N"] for x in r], [x["l2_u"] for x in r])
        out[f"order_p_{start}"] = order([x["N"] for x in r], [x["l2_p"] for x in r])
    return out


# ---------------------------------------------------------------- 6. Carreau-Yasuda channel

def carreau_channel(widths=(14, 30, 62), cu=50.0, a=2.0, n=0.3568, nu0=0.3, ratio=16.23) -> dict:
    """Shear-thinning (Carreau–Yasuda, blood exponents) channel against the exact stress balance.

    Diffusive scaling from H = 14: lattice viscosities fixed, lambda ~ H^2,
    force ~ H^-3, so the wall Carreau number lambda * shear rate is ``cu`` at
    every resolution.
    """
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM
    from zvcfd.lbm.solver import CarreauYasuda

    nu_inf = nu0 / ratio
    rows = []
    for H in widths:
        h = H / 2
        g_wall = 0.02 * 7 / h ** 2
        lam = cu / g_wall

        def nu_of(g, lam=lam):
            return exact.carreau_yasuda_nu(g, nu0, nu_inf, lam, a, n)

        F = nu_of(g_wall) * g_wall / h
        nz = _pad8(H + 2)
        flags = np.ones((nz, 8, 8), np.uint8)
        flags[1:H + 1] = 0
        dom = BrickDomain.from_flags(flags, periodic=(False, True, True))
        cy = CarreauYasuda(nu_0=nu0, nu_inf=nu_inf, lam=lam, a=a, n=n)
        sim = SparseLBM(dom, tau=3 * nu0 + 0.5, force=(F, 0.0, 0.0), collision="trt", rheology=cy)
        info = run_steady(sim, lambda: float(dense(sim, dom, "ux")[H // 2 + 1, 4, 4]),
                          check=max(500, int(H * H / nu_inf / 50)),
                          min_steps=int(15 * H * H / (np.pi ** 2 * nu_inf)))
        u = dense(sim, dom, "ux")[1:H + 1, 4, 4]
        zeta = np.arange(1, H + 1) - 0.5
        ex = exact.gn_channel(zeta - h, h, F, nu_of)
        newt = exact.plane_poiseuille(zeta, H, F, nu0)
        rows.append({"H": H, "lambda": lam, "wall_cu": cu, "l2": _l2(u, ex), "linf": _linf(u, ex),
                     "thinning": float(ex.max() / newt.max()),
                     "profile": {"zeta": zeta.tolist(), "u": u.tolist(), "exact": ex.tolist(),
                                 "newtonian_nu0": newt.tolist()} if H == 30 else None, **info})
    return {"case": "carreau_channel", "rows": rows,
            "order_l2": order([r["H"] for r in rows], [r["l2"] for r in rows])}


# ---------------------------------------------------------------- 7. sphere array

def sphere_array(sizes=(32, 64), chis=tuple(exact.SC_CHI), tau=1.0, re_p=0.01) -> dict:
    """Stokes flow through a simple cubic array of spheres (one sphere per periodic cell).

    Body force ``a`` on the fluid; at balance the total force per sphere is
    ``rho a L^3`` (Bogner et al. 2015, eq. 14), so C = a L^3 / (3 pi nu d u_bar)
    with u_bar the superficial velocity. Compared with Sangani & Acrivos (1982).
    """
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM

    nu = (tau - 0.5) / 3
    rows = []
    for L in sizes:
        c = np.arange(L) + 0.5 - L / 2
        Z, Y, X = np.meshgrid(c, c, c, indexing="ij")
        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        for chi, c_ref in zip(chis, exact.SC_DRAG[np.searchsorted(exact.SC_CHI, chis)]):
            d = chi * L
            flags = np.where(r <= d / 2, 1, 0).astype(np.uint8)
            dom = BrickDomain.from_flags(flags, periodic=True)
            u_target = re_p * nu / d
            acc = u_target * 3 * np.pi * nu * d * c_ref / L ** 3
            sim = SparseLBM(dom, tau=tau, force=(acc, 0.0, 0.0), collision="trt")
            fluid = dom.flags.reshape(-1) == 0

            def ubar():
                return float(sim.fields()["ux"].get().reshape(-1)[fluid].sum()) / L ** 3

            info = run_steady(sim, ubar, check=500, min_steps=int(3 * L * L / (np.pi ** 2 * nu)))
            u = ubar()
            C = acc * L ** 3 / (3 * np.pi * nu * d * u)
            rows.append({"L": L, "chi": float(chi), "phi": float(np.pi * chi ** 3 / 6),
                         "phi_voxel": float(1 - fluid.sum() / L ** 3), "C": float(C),
                         "C_ref": float(c_ref), "err": float(C / c_ref - 1),
                         "bogner_err": float(exact.BOGNER_ERR[L][list(exact.SC_CHI).index(chi)])
                         if L in exact.BOGNER_ERR else None, **info})
    return {"case": "sphere_array", "tau": tau, "rows": rows}


# ---------------------------------------------------------------- 8. cylinder (DFG 2D-1)

def cylinder(diameters=(20, 40, 80), u_mean=0.04, re=20.0, nz=8) -> dict:
    """Schäfer–Turek 2D-1: steady flow past a cylinder in a channel, Re = 20.

    Quasi-2-D (periodic in z, 8 voxels thick). Parabolic velocity inlet,
    pressure outlet, half-way bounce-back on the walls and the (staircase)
    cylinder; forces by momentum exchange.
    """
    from zvcfd.boundary import Patch, cell_coords, cell_ids, face_patches
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM
    from zvcfd.lbm._cuda import CX, CY, CZ, Q

    rows = []
    for D in diameters:
        Hc = int(round(4.1 * D))                  # fluid rows 1..Hc; walls at 0.5 and Hc + 0.5
        ny = _pad8(Hc + 2)
        nx = _pad8(22 * D)
        nu = u_mean * D / re
        tau = 3 * nu + 0.5
        xc, yc = 2.0 * D, 2.0 * D + 0.5           # inlet node x = 0; y_phys = (row - 0.5)
        xi = np.arange(nx)
        yj = np.arange(ny)
        Yg, Xg = np.meshgrid(yj, xi, indexing="ij")
        cyl = (Xg - xc) ** 2 + (Yg - yc) ** 2 <= (D / 2) ** 2
        flags2 = np.ones((ny, nx), np.uint8)
        flags2[1:Hc + 1] = 0
        flags2[cyl] = 1
        flags = np.repeat(flags2[None], nz, axis=0)
        dom = BrickDomain.from_flags(flags, periodic=(True, False, False))
        b = face_patches(dom, {"xmin": Patch("in", "velocity"), "xmax": Patch("out", "pressure")})
        rows_in = cell_coords(dom, b.cells[b.patch == 0])[:, 1]
        yh = (rows_in - 0.5) / Hc
        b.scale[b.patch == 0] = (6.0 * yh * (1 - yh)).astype(np.float32)   # U_max = 1.5 U_mean
        prof = np.where((yj >= 1) & (yj <= Hc), 6.0 * ((yj - 0.5) / Hc) * (1 - (yj - 0.5) / Hc), 0)
        ux0 = np.broadcast_to((u_mean * prof)[None, :, None], flags.shape).copy()
        ux0[flags == 1] = 0.0
        init = equilibrium_init(dom, RHO0, ux0)
        sim = SparseLBM(dom, tau=tau, collision="trt", boundary=b, init=init)
        sim.set_patch(0, u_zyx=(0.0, 0.0, u_mean))
        sim.set_patch(1, rho=RHO0)

        # momentum exchange: fluid cell x, link q with x + c_q inside the cylinder
        cyl3 = np.repeat(cyl[None], nz, axis=0)
        fl3 = flags == 0
        link_cells, link_q = [], []
        for q in range(1, Q):
            sh = np.zeros_like(cyl3)
            sz, sy, sx = CZ[q], CY[q], CX[q]
            src = cyl3[:, max(sy, 0):ny + min(sy, 0), max(sx, 0):nx + min(sx, 0)]
            sh[:, max(-sy, 0):ny + min(-sy, 0), max(-sx, 0):nx + min(-sx, 0)] = src
            sh = np.roll(sh, -sz, axis=0)
            hit = np.argwhere(fl3 & sh)
            link_cells.append(cell_ids(dom, hit))
            link_q.append(np.full(len(hit), q))
        import cupy as cp

        lc = np.concatenate(link_cells)
        lq = np.concatenate(link_q)
        idx = cp.asarray(lq * sim.n + lc)
        w = np.asarray([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)[lq]
        cx = cp.asarray(np.asarray(CX)[lq], cp.float64)
        cy_ = cp.asarray(np.asarray(CY)[lq], cp.float64)
        wq = cp.asarray(w)

        def forces():
            f = sim.f0[idx].astype(cp.float64) + wq
            return float(2 * (cx * f).sum()) / nz, float(2 * (cy_ * f).sum()) / nz

        pa = cell_ids(dom, np.array([[nz // 2, int(yc - 0.5), int(1.5 * D)],
                                     [nz // 2, int(yc + 0.5), int(1.5 * D)]]))
        pe = cell_ids(dom, np.array([[nz // 2, int(yc - 0.5), int(2.5 * D)],
                                     [nz // 2, int(yc + 0.5), int(2.5 * D)]]))
        norm = RHO0 * u_mean ** 2 * D / 2
        info = run_steady(sim, lambda: forces()[0], check=1000, tol=1e-7,
                          min_steps=int(3 * nx / u_mean))
        fx, fy = forces()
        from zvcfd.lbm._cuda import Q as NQ

        g = sim.f0.reshape(NQ, -1).astype(cp.float64).sum(0).get()
        dp = (g[pa].mean() - g[pe].mean()) / 3 / (RHO0 * u_mean ** 2)
        rows.append({"D": D, "tau": tau, "cells": int(dom.fluid_cells), "cd": fx / norm,
                     "cl": fy / norm, "dp_norm": float(dp),
                     "dp": float(dp * exact.DFG_2D1["dp"] / exact.DFG_2D1["dp_norm"]),
                     "mach": u_mean * 1.5 * np.sqrt(3), **info})
        for key in ("cd", "cl", "dp"):
            rows[-1][f"{key}_err"] = rows[-1][key] / exact.DFG_2D1[key] - 1
        rows[-1]["u_mean"] = u_mean
    return {"case": "cylinder", "reference": exact.DFG_2D1, "rows": rows}


def cylinder_mach(D=40, u_means=(0.04, 0.02)) -> dict:
    """The cylinder at one resolution and two lattice Mach numbers: staircase or compressibility?"""
    rows = []
    for u in u_means:
        rows += cylinder(diameters=(D,), u_mean=u)["rows"]
    return {"case": "cylinder_mach", "reference": exact.DFG_2D1, "rows": rows}


CASES = {"poiseuille": poiseuille, "poiseuille_pressure": poiseuille_pressure, "duct": duct,
         "pipe": pipe, "womersley": womersley, "taylor_green": taylor_green,
         "carreau_channel": carreau_channel, "sphere_array": sphere_array, "cylinder": cylinder,
         "cylinder_mach": cylinder_mach}
