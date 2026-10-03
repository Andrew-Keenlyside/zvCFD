"""The zero-pressure-gradient flat plate with the k-kL model, by boundary-layer marching.

    PYTHONPATH=. python benchmarks/turbulence/plate_bl.py

An independent reference for the NASA TMR flat plate (Re = 5 million per unit
length, the TMR freestream turbulence for M = 0.2): the incompressible
boundary-layer equations

    u du/dx + v du/dy = d/dy[(nu + nu_t) du/dy],   du/dx + dv/dy = 0,
    u dk/dx + v dk/dy = d/dy[(nu + nu_t) dk/dy] + P_k - C_mu^3/4 k^5/2/Phi - 2 nu k/y^2,
    u dPhi/dx + v dPhi/dy = d/dy[(nu + nu_t) dPhi/dy] + C_phi1 (Phi/k) P_k - zeta3 k^3/2
                            - 6 nu Phi f_phi/y^2,

with k-kL-MEAH2015m's closure (U' = |du/dy|, U'' = |d2u/dy2|, d = y), marched in x
implicitly (backward Euler in x, Picard in each station, tridiagonal in y).
It shares no code with zvcfd.fv. Prints cf at x = 0.97 (FUN3D and CFL3D: about
0.0027, NASA/TM-2015-218968 Fig. 2) and cf against Re_theta (Karman-Schoenherr).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

C_MU, KAPPA, Z1, Z2, Z3, C11, C12, CD1 = 0.09, 0.41, 1.2, 0.97, 0.13, 10.0, 1.3, 4.7
RE, U_INF, MACH = 5.0e6, 1.0, 0.2
NU = 1.0 / RE


def tri(lower, diag, upper, b):
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, b)


def ks(re_theta):
    lg = np.log10(re_theta)
    return 1.0 / (17.08 * lg ** 2 + 25.11 * lg + 6.012)


def march(ny: int = 240, y_max: float = 0.12, h1: float = 2e-7, nx: int = 600,
          x0: float = 1e-5, x1: float = 1.0, picard: int = 12):
    # stretched y grid
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        lo, hi = (lo, r) if h1 * (r ** ny - 1) / (r - 1) > y_max else (r, hi)
    y = np.concatenate([[0.0], np.cumsum(h1 * r ** np.arange(ny))])
    y *= y_max / y[-1]
    a = U_INF / MACH
    k_inf, phi_inf = 9e-9 * a * a, 1.5589e-6 * NU * a          # rho = 1, mu = nu
    xs = np.geomspace(x0, x1, nx)
    # laminar-ish start: Blasius-like profile at x0
    delta = 5 * np.sqrt(NU * x0 / U_INF)
    u = U_INF * np.tanh(2 * y / delta)
    k = np.full_like(y, k_inf)
    phi = np.full_like(y, phi_inf)
    u[0] = k[0] = phi[0] = 0.0
    yf = 0.5 * (y[1:] + y[:-1])
    vol = np.empty_like(y)
    vol[1:-1] = yf[1:] - yf[:-1]
    vol[0], vol[-1] = yf[0], y[-1] - yf[-1]
    d = np.where(y > 0, y, 1.0)
    out = []
    x_prev = x0
    for x in xs[1:]:
        dx = x - x_prev
        u0, k0, p0 = u.copy(), k.copy(), phi.copy()
        for _ in range(picard):
            kk, pp = np.maximum(k, 1e-6 * k_inf), np.maximum(phi, 1e-6 * phi_inf)
            nut = C_MU ** 0.25 * pp / np.sqrt(kk)
            nut[0] = 0.0
            # continuity: v from du/dx
            dudx = (u - u0) / dx
            v = np.concatenate([[0.0], -np.cumsum(0.5 * (dudx[1:] + dudx[:-1]) * np.diff(y))])
            gam = NU + nut
            cf_ = 0.5 * (gam[1:] + gam[:-1]) / np.diff(y)

            def operator(phi_old, ap_src, b_src, top):
                """u (phi - phi_old)/dx + v dphi/dy (upwind) - d/dy(gam dphi/dy) + ap phi = b."""
                lower = np.zeros_like(y)
                upper = np.zeros_like(y)
                diag = np.zeros_like(y)
                lower[1:] -= cf_
                upper[:-1] -= cf_
                diag[1:] += cf_
                diag[:-1] += cf_
                diag = diag / vol
                lower, upper = lower / vol, upper / vol
                # advection in y, upwind
                dyl = np.concatenate([[1.0], np.diff(y)])
                dyu = np.concatenate([np.diff(y), [1.0]])
                vp, vm = np.maximum(v, 0), np.minimum(v, 0)
                diag = diag + vp / dyl - vm / dyu
                lower = lower - vp / dyl
                upper = upper + vm / dyu
                diag = diag + np.maximum(u, 1e-12) / dx + ap_src
                b = b_src + np.maximum(u, 1e-12) / dx * phi_old
                # wall: 0; top: freestream
                diag[0], upper[0], b[0] = 1.0, 0.0, 0.0
                diag[-1], lower[-1], b[-1] = 1.0, 0.0, top
                return tri(lower, diag, upper, b)
            u = operator(u0, np.zeros_like(y), np.zeros_like(y), U_INF)
            dU = np.gradient(u, y)
            d2U = np.gradient(dU, y)
            S2 = dU ** 2
            P = nut * S2
            cmu34 = C_MU ** 0.75
            Pk = np.minimum(P, 20 * cmu34 * kk ** 2.5 / pp)
            fp = np.clip(Pk * pp / (cmu34 * kk ** 2.5), 0.5, 1.0)
            Lvk = KAPPA * np.abs(dU) / np.maximum(np.abs(d2U), 1e-300)
            Lvk = np.minimum(np.maximum(Lvk, pp / (kk * C11)), C12 * KAPPA * d * fp)
            c1 = Z1 - Z2 * (pp / (kk * Lvk)) ** 2
            xi = d * np.sqrt(0.3 * kk) / (20 * NU)
            fphi = (1 + CD1 * xi) / (1 + xi ** 4)
            kn = operator(k0, cmu34 * kk ** 1.5 / pp + 2 * NU / d ** 2, Pk, k_inf)
            prod = c1 * Pk / kk
            pn = operator(p0, Z3 * kk ** 1.5 / pp + 6 * NU * fphi / d ** 2 + np.maximum(-prod, 0),
                          np.maximum(prod, 0) * pp, phi_inf)
            k = np.maximum(kn, 0.1 * k)
            phi = np.maximum(pn, 0.1 * phi)
            k[0] = phi[0] = 0.0
        x_prev = x
        cf = 2 * NU * (u[1] - u[0]) / (y[1] - y[0]) / U_INF ** 2
        ue = u[-1]
        theta = np.trapezoid(u / ue * (1 - u / ue), y)
        out.append((x, cf, theta * ue / NU))
    return np.array(out), y, u, k, phi


if __name__ == "__main__":
    res, y, u, k, phi = march()
    x, cf, rt = res.T
    i = np.argmin(np.abs(x - 0.97))
    print(f"x = {x[i]:.3f}: cf {cf[i]:.6f}  Re_theta {rt[i]:.0f}  Karman-Schoenherr {ks(rt[i]):.6f}"
          f"  (FUN3D/CFL3D k-kL: ~0.0027)")
    for xx in (0.1, 0.3, 0.5, 0.97):
        j = np.argmin(np.abs(x - xx))
        print(f"   x {x[j]:.3f}  cf {cf[j]:.6f}  Re_theta {rt[j]:7.0f}  "
              f"KS {ks(max(rt[j], 10)):.6f}")
    ut = np.sqrt(cf[-1] / 2)
    yp, up = y * ut / NU, u / ut
    for t in (10, 30, 100, 300, 1000):
        j = np.argmin(np.abs(yp - t))
        print(f"   y+ {yp[j]:7.1f}  u+ {up[j]:6.2f}  log law {np.log(yp[j]) / 0.41 + 5.0:6.2f}")
