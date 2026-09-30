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
- Manufactured solution (method of manufactured solutions): P. J. Roache,
  *Code verification by the method of manufactured solutions*, J. Fluids
  Eng. 124, 4 (2002); K. Salari & P. Knupp, Sandia report SAND2000-1444
  (2000). The velocity field is divergence-free and each component is an
  eigenfunction of the Laplacian; the source term is derived by hand below.
- Ethier–Steinman flow: C. R. Ethier & D. A. Steinman, *Exact fully 3D
  Navier–Stokes solutions for benchmarking*, Int. J. Numer. Meth. Fluids
  19, 369 (1994): unsteady, fully 3-D, all nonlinear terms active.
- Kovasznay flow: L. I. G. Kovasznay, *Laminar flow behind a
  two-dimensional grid*, Proc. Camb. Phil. Soc. 44, 58 (1948).
- Cylinder in a channel, DFG benchmark 2D-1: M. Schäfer & S. Turek, Notes
  Numer. Fluid Mech. 52, 547 (1996); high-accuracy values from the FeatFlow
  benchmark page and V. John & G. Matthies, Int. J. Numer. Meth. Fluids 37,
  885 (2001).
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, special

# ---------------------------------------------------------------- manufactured solution


def manufactured(x: np.ndarray, a: float = np.pi, rho: float = 1.0, mu: float = 1.0,
                 p0: float = 1.0, stokes: bool = False):
    """A smooth steady 3-D solution and the body force that makes it exact.

    ``u = (sin ax cos ay cos az, cos ax sin ay cos az, -2 cos ax cos ay sin az)``
    (divergence-free, ``∇²u = -3a² u``), ``p = p0 sin ax sin ay sin az``, and
    ``f = ρ (u·∇)u - μ ∇²u + ∇p`` (no advection if ``stokes``). ``x`` is
    ``(N, 3)`` (x, y, z) in physical units. Returns ``(u (N, 3), p (N,), f (N, 3))``.
    """
    X, Y, Z = (a * x[:, 0], a * x[:, 1], a * x[:, 2])
    sx, cx, sy, cy, sz, cz = np.sin(X), np.cos(X), np.sin(Y), np.cos(Y), np.sin(Z), np.cos(Z)
    u = np.stack([sx * cy * cz, cx * sy * cz, -2 * cx * cy * sz], 1)
    # du_k/dx_j, as J[:, k, j]
    J = a * np.stack([
        np.stack([cx * cy * cz, -sx * sy * cz, -sx * cy * sz], 1),
        np.stack([-sx * sy * cz, cx * cy * cz, -cx * sy * sz], 1),
        np.stack([2 * sx * cy * sz, 2 * cx * sy * sz, -2 * cx * cy * cz], 1)], 1)
    p = p0 * sx * sy * sz
    gp = p0 * a * np.stack([cx * sy * sz, sx * cy * sz, sx * sy * cz], 1)
    f = 3 * a * a * mu * u + gp
    if not stokes:
        f = f + rho * np.einsum("nkj,nj->nk", J, u)
    return u, p, f


def _manufactured_fields(z: np.ndarray, a: float):
    """The divergence-free field of :func:`manufactured` and its gradient, complex-safe."""
    X, Y, Z = a * z[:, 0], a * z[:, 1], a * z[:, 2]
    sx, cx, sy, cy, sz, cz = np.sin(X), np.cos(X), np.sin(Y), np.cos(Y), np.sin(Z), np.cos(Z)
    u = np.stack([sx * cy * cz, cx * sy * cz, -2 * cx * cy * sz], 1)
    J = a * np.stack([
        np.stack([cx * cy * cz, -sx * sy * cz, -sx * cy * sz], 1),
        np.stack([-sx * sy * cz, cx * cy * cz, -cx * sy * sz], 1),
        np.stack([2 * sx * cy * sz, 2 * cx * sy * sz, -2 * cx * cy * cz], 1)], 1)
    return u, J


def manufactured_gn(x: np.ndarray, mu_of_g2, a: float = np.pi, rho: float = 1.0,
                    p0: float = 1.0, shift=(0.0, 0.0, 0.0), mean=(0.0, 0.0, 0.0),
                    stokes: bool = False):
    """:func:`manufactured` for a generalised-Newtonian fluid ``μ(γ̇²)``, full stress.

    ``u(x) = u_m(x + shift) + mean`` (still divergence-free), ``p`` as in
    :func:`manufactured`, and ``f = ρ (u·∇)u − ∇·[μ (∇u + ∇uᵀ)] + ∇p``. The
    stress divergence comes from complex-step derivatives of the analytic
    stress, exact to round-off: ``∂τ/∂x_j = Im τ(x + i h e_j) / h``.
    ``mu_of_g2`` takes ``γ̇² = 2 S:S`` and must accept complex arguments.
    """
    s = np.asarray(shift, float)
    z = x + s
    u, J = _manufactured_fields(z, a)
    u = u + np.asarray(mean, float)
    P = a * z
    p = p0 * np.sin(P[:, 0]) * np.sin(P[:, 1]) * np.sin(P[:, 2])
    gp = p0 * a * np.stack([np.cos(P[:, 0]) * np.sin(P[:, 1]) * np.sin(P[:, 2]),
                            np.sin(P[:, 0]) * np.cos(P[:, 1]) * np.sin(P[:, 2]),
                            np.sin(P[:, 0]) * np.sin(P[:, 1]) * np.cos(P[:, 2])], 1)

    def stress(zz):
        _, JJ = _manufactured_fields(zz, a)
        S = 0.5 * (JJ + np.swapaxes(JJ, 1, 2))
        g2 = 2 * np.einsum("nij,nij->n", S, S)
        return mu_of_g2(g2)[:, None, None] * 2 * S

    h = 1e-30
    div = np.zeros((len(x), 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1.0
        div += np.imag(stress(z + 1j * h * e))[:, :, j] / h
    f = -div + gp
    if not stokes:
        f = f + rho * np.einsum("nkj,nj->nk", J, u)
    return u, p, f, J


def manufactured_noslip(x: np.ndarray, rho: float = 1.0, mu: float = 1.0, amp: float = 100.0,
                        p0: float = 1.0, stokes: bool = False):
    """A steady 3-D solution with ``u = 0`` on every face of the unit cube (no-slip walls).

    ``u = amp ∇ × (0, φ, φ)`` with ``φ = g(x) g(y) g(z)``, ``g(s) = s² (1 − s)²``:
    divergence-free, and zero on the boundary with its tangential
    derivatives, like flow at a wall. ``p = p0 cos πx cos πy cos πz`` (not
    zero on the walls). ``f = ρ (u·∇)u − μ ∇²u + ∇p``, derived by hand from
    the polynomial ``g``. Returns ``(u, p, f)``.
    """
    def gd(s, k):
        return [s**2 * (1 - s)**2, 2 * s * (1 - s) * (1 - 2 * s), 2 - 12 * s + 12 * s**2,
                -12 + 24 * s][k]

    X, Y, Z = x[:, 0], x[:, 1], x[:, 2]

    def phi(i, j, k):                     # d^i/dx^i d^j/dy^j d^k/dz^k of g(x) g(y) g(z)
        return gd(X, i) * gd(Y, j) * gd(Z, k)

    # u = amp curl(0, phi, phi) = amp (dphi/dy - dphi/dz, -dphi/dx, dphi/dx)
    def vel(i, j, k):                     # a derivative of u, component-wise
        return amp * np.stack([phi(i, j + 1, k) - phi(i, j, k + 1), -phi(i + 1, j, k),
                               phi(i + 1, j, k)], 1)

    u = vel(0, 0, 0)
    J = np.stack([vel(1, 0, 0), vel(0, 1, 0), vel(0, 0, 1)], 2)        # J[:, k, j] = du_k/dx_j
    lap = vel(2, 0, 0) + vel(0, 2, 0) + vel(0, 0, 2)
    c = np.cos(np.pi * x)
    sn = np.sin(np.pi * x)
    p = p0 * c[:, 0] * c[:, 1] * c[:, 2]
    gp = -p0 * np.pi * np.stack([sn[:, 0] * c[:, 1] * c[:, 2], c[:, 0] * sn[:, 1] * c[:, 2],
                                 c[:, 0] * c[:, 1] * sn[:, 2]], 1)
    f = -mu * lap + gp
    if not stokes:
        f = f + rho * np.einsum("nkj,nj->nk", J, u)
    return u, p, f


def ethier_steinman(x: np.ndarray, t: float, a: float = np.pi / 4, d: float = np.pi / 2,
                    nu: float = 1.0, rho: float = 1.0):
    """Ethier & Steinman (1994): exact unsteady 3-D Navier–Stokes flow, no body force.

    Returns ``(u (N, 3), p (N,))`` at time ``t`` (kinematic pressure × ρ).
    """
    X, Y, Z = x[:, 0], x[:, 1], x[:, 2]
    e = np.exp(-d * d * nu * t)
    u = -a * (np.exp(a * X) * np.sin(a * Y + d * Z) + np.exp(a * Z) * np.cos(a * X + d * Y))
    v = -a * (np.exp(a * Y) * np.sin(a * Z + d * X) + np.exp(a * X) * np.cos(a * Y + d * Z))
    w = -a * (np.exp(a * Z) * np.sin(a * X + d * Y) + np.exp(a * Y) * np.cos(a * Z + d * X))
    p = -0.5 * a * a * (np.exp(2 * a * X) + np.exp(2 * a * Y) + np.exp(2 * a * Z)
                        + 2 * np.sin(a * X + d * Y) * np.cos(a * Z + d * X) * np.exp(a * (Y + Z))
                        + 2 * np.sin(a * Y + d * Z) * np.cos(a * X + d * Y) * np.exp(a * (Z + X))
                        + 2 * np.sin(a * Z + d * X) * np.cos(a * Y + d * Z) * np.exp(a * (X + Y)))
    return np.stack([u, v, w], 1) * e, rho * p * e * e


def kovasznay(x: np.ndarray, re: float = 40.0):
    """Kovasznay (1948): exact steady 2-D Navier–Stokes flow behind a grid, ``ν = 1/Re``.

    ``u = 1 − e^{λx} cos 2πy``, ``v = λ/(2π) e^{λx} sin 2πy``, ``p = (1 − e^{2λx})/2``
    with ``λ = Re/2 − sqrt(Re²/4 + 4π²)``. Returns ``(u (N, 3), p (N,))``, w = 0.
    """
    lam = re / 2 - np.sqrt(re * re / 4 + 4 * np.pi ** 2)
    X, Y = x[:, 0], x[:, 1]
    ex = np.exp(lam * X)
    u = np.stack([1 - ex * np.cos(2 * np.pi * Y), lam / (2 * np.pi) * ex * np.sin(2 * np.pi * Y),
                  0 * X], 1)
    return u, 0.5 * (1 - np.exp(2 * lam * X))


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
