"""Fully developed channel flow with the k-kL model in one dimension: an independent reference.

    PYTHONPATH=. python benchmarks/turbulence/channel_1d.py [Re_tau ...]

Half a channel in wall units (u_tau = 1, half height H = 1, nu = 1/Re_tau, rho = 1),
driven by dp/dx = -1, wall at y = 0 and symmetry at y = 1:

    d/dy[(nu + nu_t) dU/dy] + 1 = 0
    d/dy[(nu + nu_t) dk/dy] + P_k - C_mu^3/4 k^5/2/Phi - 2 nu k/y^2 = 0
    d/dy[(nu + nu_t) dPhi/dy] + C_phi1 (Phi/k) P_k - zeta3 k^3/2 - 6 nu Phi f_phi/y^2 = 0

with k-kL-MEAH2015m's closure exactly as zvcfd.fv.turbulence (sigma_k = sigma_phi = 1).
Finite differences on a stretched grid, pseudo-time with the same implicit
linearisation of the sinks, tridiagonal solves. It shares no code with the
3-D finite-volume implementation, so it checks the model's equations and
their implementation independently.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.linalg import solve_banded

C_MU, KAPPA, Z1, Z2, Z3, C11, C12, CD1 = 0.09, 0.41, 1.2, 0.97, 0.13, 10.0, 1.3, 4.7


def grid(n: int, re_tau: float, y1p: float = 0.3) -> np.ndarray:
    """Geometric stretching from the wall: first spacing y1p wall units, to y = 1."""
    h1 = y1p / re_tau
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):                          # growth ratio r with h1 (r^n - 1)/(r - 1) = 1
        r = 0.5 * (lo + hi)
        if h1 * (r ** n - 1) / (r - 1) > 1:
            hi = r
        else:
            lo = r
    y = np.concatenate([[0.0], np.cumsum(h1 * r ** np.arange(n))])
    return y / y[-1]


def _diffusion(y, gam):
    """Tridiagonal coefficients of -d/dy(gam dphi/dy) (finite volumes around the nodes)."""
    n = len(y)
    yf = 0.5 * (y[1:] + y[:-1])
    gf = 0.5 * (gam[1:] + gam[:-1])
    cf = gf / np.diff(y)                          # face conductances
    vol = np.empty(n)
    vol[1:-1] = yf[1:] - yf[:-1]
    vol[0], vol[-1] = yf[0] - y[0], y[-1] - yf[-1]
    lower = np.zeros(n)                           # coefficient of phi_{i-1}
    upper = np.zeros(n)                           # coefficient of phi_{i+1}
    diag = np.zeros(n)
    lower[1:] = -cf
    upper[:-1] = -cf
    diag[1:] += cf
    diag[:-1] += cf
    return lower, diag, upper, vol


def _solve(lower, diag, upper, b):
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, b)


def solve(re_tau: float, n: int = 160, iterations: int = 20000, dtau: float = 1.0,
          tol: float = 1e-10, verbose: bool = False):
    nu = 1.0 / re_tau
    y = grid(n, re_tau)
    N = len(y)
    d = np.where(y > 0, y, 1.0)                   # distance to the wall (unused at y = 0)
    U = 1.0 / KAPPA * np.log1p(y * re_tau) + 5.0 * (1 - np.exp(-y * re_tau / 11))
    k = np.full(N, 3.3) * np.minimum(y * re_tau / 10, 1) ** 2
    phi = k * np.minimum(KAPPA * y, 0.1)
    k[0] = phi[0] = 0.0
    floor = 1e-14
    for it in range(iterations):
        kk, pp = np.maximum(k, floor), np.maximum(phi, floor)
        nut = C_MU ** 0.25 * pp / np.sqrt(kk)
        nut[0] = 0.0
        # momentum
        lo, di, up, vol = _diffusion(y, nu + nut)
        b = vol * 1.0
        di2 = di + vol / dtau
        b2 = b + vol / dtau * U
        di2[0], up[0], b2[0] = 1.0, 0.0, 0.0
        Un = _solve(lo, di2, up, b2)
        dU = np.gradient(Un, y)
        d2U = np.gradient(dU, y)
        S2 = dU ** 2
        P = nut * S2
        cmu34 = C_MU ** 0.75
        Pk = np.minimum(P, 20 * cmu34 * kk ** 2.5 / pp)
        fp = np.clip(Pk * pp / (cmu34 * kk ** 2.5), 0.5, 1.0)
        Lvk = KAPPA * np.abs(dU) / np.maximum(np.abs(d2U), 1e-300)
        Lvk = np.minimum(np.maximum(Lvk, pp / (kk * C11)), C12 * KAPPA * d * fp)
        c1 = Z1 - Z2 * (pp / (kk * Lvk)) ** 2
        xi = d * np.sqrt(0.3 * kk) / (20 * nu)
        fphi = (1 + CD1 * xi) / (1 + xi ** 4)
        # k
        lo, di, up, vol = _diffusion(y, nu + nut)
        ap = vol * (cmu34 * kk ** 1.5 / pp + 2 * nu / d ** 2) + vol / dtau
        b = vol * Pk + vol / dtau * k
        di2 = di + ap
        di2[0], up[0], b[0] = 1.0, 0.0, 0.0
        kn = np.maximum(_solve(lo, di2, up, b), 0.1 * k)
        kn[0] = 0.0
        # phi
        prod = c1 * Pk / kk
        ap = (vol * (Z3 * kk ** 1.5 / pp + 6 * nu * fphi / d ** 2 + np.maximum(-prod, 0))
              + vol / dtau)
        b = vol * np.maximum(prod, 0) * pp + vol / dtau * phi
        di2 = di + ap
        di2[0], up[0], b[0] = 1.0, 0.0, 0.0
        pn = np.maximum(_solve(lo, di2, up, b), 0.1 * phi)
        pn[0] = 0.0
        ch = max(np.abs(Un - U).max() / np.abs(Un).max(), np.abs(kn - k).max() / np.abs(kn).max(),
                 np.abs(pn - phi).max() / np.abs(pn).max())
        U, k, phi = Un, kn, pn
        if verbose and it % 2000 == 0:
            print(it, ch)
        if ch < tol:
            break
    nut = C_MU ** 0.25 * np.maximum(phi, floor) / np.sqrt(np.maximum(k, floor))
    nut[0] = 0.0
    Ub = np.trapezoid(U, y)
    # steady momentum: the total shear stress is exactly 1 - y
    stress = 0.5 * ((nu + nut)[1:] + (nu + nut)[:-1]) * np.diff(U) / np.diff(y)
    resid = float(np.abs(stress - (1 - 0.5 * (y[1:] + y[:-1]))).max())
    return {"y": y, "U": U, "k": k, "phi": phi, "nut": nut, "Lvk": Lvk, "c1": c1,
            "Ub": Ub, "Uc": U[-1], "cf": 2.0 / Ub ** 2, "re_b": 2 * Ub * re_tau,
            "iterations": it + 1, "change": ch, "stress_residual": resid}


def report(r, re_tau):
    y, U = r["y"], r["U"]
    yp = y * re_tau
    print(f"Re_tau {re_tau:.0f}: {r['iterations']} its (change {r['change']:.1e}, shear-stress "
          f"residual {r['stress_residual']:.1e}); Re_b {r['re_b']:.0f}, "
          f"cf {r['cf']:.5f} (Dean {0.073 * r['re_b'] ** -0.25:.5f}), "
          f"Uc/Ub {r['Uc'] / r['Ub']:.4f} (Dean {1.28 * r['re_b'] ** -0.0116:.4f})")
    sel = (yp > 30) & (y < 0.2)
    slope = np.polyfit(np.log(yp[sel]), U[sel], 1)[0]
    print(f"   log-layer slope 1/kappa_eff = {slope:.3f} -> kappa_eff {1 / slope:.3f};  "
          f"intercept at y+ = 100: u+ {np.interp(100, yp, U):.2f} "
          f"(law: {np.log(100) / 0.41 + 5.0:.2f})")
    L = r["phi"] / np.maximum(r["k"], 1e-300)
    for t in (10, 30, 100, 300, 1000):
        if t < re_tau * 0.5:
            i = np.argmin(np.abs(yp - t))
            print(f"   y+ {yp[i]:7.1f}: u+ {U[i]:6.2f}  k+ {r['k'][i]:5.2f}  "
                  f"L/(kappa y) {L[i] / (KAPPA * y[i]):5.3f}  nut+ {r['nut'][i] * re_tau:7.1f}  "
                  f"Lvk/(kappa y) {r['Lvk'][i] / (KAPPA * y[i]):5.3f}")


if __name__ == "__main__":
    for rt in [float(a) for a in sys.argv[1:]] or [942.0, 5000.0, 20000.0]:
        report(solve(rt), rt)
