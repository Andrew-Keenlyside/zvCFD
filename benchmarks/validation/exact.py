"""Exact and reference solutions for the validation suite.

Every function here is independent of zvCFD: plain NumPy/SciPy evaluations
of closed-form solutions (or, for the sphere array and the cylinder, the
published reference values). Lattice units throughout unless stated.

References
----------
- Plane Poiseuille and Hagen–Poiseuille: e.g. White, *Viscous Fluid Flow*,
  3rd ed. (2006), §3-3.
- Square duct: White (2006), eq. 3-48 (series solution).
- Womersley: J. R. Womersley, J. Physiol. 127, 553 (1955) (pipe); the plane
  channel is the same problem with cosh in place of J0.
- Taylor–Green vortex: G. I. Taylor & A. E. Green, Proc. R. Soc. A 158, 499
  (1937); exact for the full incompressible Navier–Stokes equations.
- Generalised Newtonian channel: the momentum balance fixes the shear
  stress (``nu(g) g = F y``) whatever the rheology; the velocity is its
  integral (Bird, Armstrong & Hassager, *Dynamics of Polymeric Liquids*,
  1987, ch. 4).
- Simple cubic array of spheres: A. S. Sangani & A. Acrivos, Int. J.
  Multiphase Flow 8, 343 (1982), as tabulated by S. Bogner, S. Mohanty &
  U. Rüde, Int. J. Multiphase Flow 68, 71 (2015), arXiv:1401.2025, Table 1.
- Cylinder in a channel, DFG benchmark 2D-1: M. Schäfer & S. Turek, Notes
  Numer. Fluid Mech. 52, 547 (1996); high-accuracy values from the FeatFlow
  benchmark page and V. John & G. Matthies, Int. J. Numer. Meth. Fluids 37,
  885 (2001).
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, special

# ---------------------------------------------------------------- steady channels


def plane_poiseuille(zeta: np.ndarray, H: float, F: float, nu: float) -> np.ndarray:
    """Velocity at distance ``zeta`` from one wall of a channel of width ``H``, body force ``F``."""
    return F / (2 * nu) * zeta * (H - zeta)


def hagen_poiseuille(r: np.ndarray, R: float, F: float, nu: float) -> np.ndarray:
    return np.where(r < R, F / (4 * nu) * (R * R - r * r), 0.0)


def square_duct(y: np.ndarray, z: np.ndarray, a: float, F: float, nu: float,
                terms: int = 200) -> np.ndarray:
    """Stokes flow in the duct ``|y| < a, |z| < a`` driven by ``F`` (White eq. 3-48, b = a)."""
    u = np.zeros(np.broadcast(y, z).shape)
    for i in range(terms):
        n = 2 * i + 1
        k = n * np.pi / (2 * a)
        # cosh(k z)/cosh(k a) without overflow
        az = np.abs(z)
        ratio = np.exp(k * (az - a)) * (1 + np.exp(-2 * k * az)) / (1 + np.exp(-2 * k * a))
        u += (-1) ** i * (1 - ratio) * np.cos(k * y) / n ** 3
    return 16 * a * a * F / (nu * np.pi ** 3) * u


def square_duct_flux(a: float, F: float, nu: float, terms: int = 200) -> float:
    n = np.arange(1, 2 * terms, 2)
    s = (np.tanh(n * np.pi / 2) / n ** 5).sum()
    return 4 * a ** 4 * F / (3 * nu) * (1 - 192 / np.pi ** 5 * s)


# ---------------------------------------------------------------- Womersley


def womersley_plane(zeta: np.ndarray, H: float, F0: float, omega: float, nu: float,
                    t: float) -> np.ndarray:
    """Channel of width ``H`` driven by ``F0 cos(omega t)``: the periodic (post-transient) state."""
    k = np.sqrt(1j * omega / nu)
    y = zeta - H / 2
    shape = 1 - np.cosh(k * y) / np.cosh(k * H / 2)
    return np.real(F0 / (1j * omega) * shape * np.exp(1j * omega * t))


def womersley_pipe(r: np.ndarray, R: float, F0: float, omega: float, nu: float,
                   t: float) -> np.ndarray:
    """Pipe of radius ``R`` driven by ``F0 cos(omega t)`` (Womersley 1955)."""
    alpha = R * np.sqrt(omega / nu)
    arg = 1j ** 1.5 * alpha
    shape = 1 - special.jv(0, arg * r / R) / special.jv(0, arg)
    return np.where(r < R, np.real(F0 / (1j * omega) * shape * np.exp(1j * omega * t)), 0.0)


# ---------------------------------------------------------------- Taylor-Green


def taylor_green(x: np.ndarray, y: np.ndarray, N: int, U: float, nu: float, t: float):
    """2-D Taylor–Green vortex in a periodic box of side ``N``: (ux, uy, p) at time ``t``."""
    k = 2 * np.pi / N
    d = np.exp(-2 * nu * k * k * t)
    ux = -U * np.cos(k * x) * np.sin(k * y) * d
    uy = U * np.sin(k * x) * np.cos(k * y) * d
    p = -U * U / 4 * (np.cos(2 * k * x) + np.cos(2 * k * y)) * d * d
    return ux, uy, p


# ---------------------------------------------------------------- generalised Newtonian


def carreau_yasuda_nu(g, nu_0, nu_inf, lam, a, n):
    return nu_inf + (nu_0 - nu_inf) * (1 + (lam * g) ** a) ** ((n - 1) / a)


def gn_channel(dist: np.ndarray, h: float, F: float, nu_of_g) -> np.ndarray:
    """Plane channel of half-width ``h``, body force ``F``, viscosity ``nu_of_g(shear rate)``.

    ``dist`` is the distance from the centre line. The shear stress is
    ``F * dist`` exactly; the shear rate follows from ``nu(g) g = F dist`` and
    the velocity is its integral from the wall.
    """
    def g_of(y):
        s = F * y
        if s <= 0:
            return 0.0
        hi = s / min(nu_of_g(1e30), nu_of_g(0.0)) * 2 + 1e-30
        return optimize.brentq(lambda g: nu_of_g(g) * g - s, 0.0, hi, xtol=1e-16, rtol=1e-14)

    out = []
    for y in np.atleast_1d(np.abs(dist)):
        v, _ = integrate.quad(g_of, y, h, epsabs=1e-15, epsrel=1e-12, limit=200)
        out.append(v)
    return np.array(out)


# ---------------------------------------------------------------- sphere array

# Sangani & Acrivos (1982), simple cubic array, as tabulated by Bogner et al.
# (2015) Table 1: chi = d/L, and the normalised drag C = f_t / (3 pi mu d u_bar)
# (f_t the total force per sphere, u_bar the superficial velocity) recovered as
# C*/(1 + err) from their two resolutions (L = 32, 64), averaged.
SC_CHI = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
_C32 = np.array([1.37, 1.75, 2.16, 2.93, 4.00, 6.15, 10.29, 20.06])
_E32 = np.array([-1.13, 2.74, 0.45, 2.93, 0.57, 2.49, 2.43, 4.69]) / 100
_C64 = np.array([1.39, 1.70, 2.17, 2.89, 4.00, 6.08, 10.25, 19.52])
_E64 = np.array([0.37, 0.27, 0.92, 1.63, 0.73, 1.27, 1.95, 1.86]) / 100
SC_DRAG = 0.5 * (_C32 / (1 + _E32) + _C64 / (1 + _E64))
# the published TRT results, for comparison with ours at the same resolutions
BOGNER_ERR = {32: _E32, 64: _E64}


def sc_drag_series(phi: np.ndarray) -> np.ndarray:
    """Sangani & Acrivos (1982) small-concentration series for the simple cubic array.

    ``K = 1 / (1 - 1.7601 c^(1/3) + c - 1.5593 c^2 + 3.9799 c^(8/3) - 3.0734 c^(10/3))``;
    within 0.3 % of the full solution up to c ~ 0.2, not beyond.
    """
    c = np.asarray(phi, float)
    x = c ** (1 / 3)
    return 1 / (1 - 1.7601 * x + c - 1.5593 * c ** 2 + 3.9799 * x ** 8 - 3.0734 * x ** 10)


# ---------------------------------------------------------------- cylinder (DFG 2D-1)

# Re = U_mean D / nu = 20, parabolic inflow with U_mean = 2/3 U_max, H = 4.1 D,
# cylinder centre (2 D, 2 D) from the inflow and the bottom wall, rho = 1.
# C_D = 2 F_x / (rho U_mean^2 D), C_L likewise, dp = p(1.5 D, 2 D) - p(2.5 D, 2 D)
# with U_mean = 0.2 m/s, i.e. dp / (rho U_mean^2) = 0.11752016697 / 0.04.
DFG_2D1 = {"cd": 5.57953523384, "cl": 0.010618948146, "dp": 0.11752016697,
           "dp_norm": 0.11752016697 / 0.04,
           "interval": {"cd": (5.57, 5.59), "cl": (0.0104, 0.0110), "dp": (0.1172, 0.1176)}}
