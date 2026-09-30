"""0-D outlet models, shared by every solver: RCR (Windkessel) and open-loop coronary.

An outlet model maps the flow leaving the 3-D domain through a patch to
the pressure on that patch. Each is a linear state-space system

    dx/dt = A x + Bq Q + Bu u(t),        P = Cx x + Dq Q + Du u(t),

with ``Q`` the outflow (m³/s, positive leaving the 3-D domain) and ``u``
the prescribed inputs ``[P_v, P_im(t)]`` (venous and intramyocardial
pressure, Pa). Time steps use the trapezoidal rule (second order,
A-stable) or backward Euler (first order, L-stable).

Coupling to a 3-D solver
------------------------
Over a step ending at ``t``, the discrete model gives the patch pressure
as an affine function of that step's end flow:

    P(t) = a + r Q(t),            (a, r) = model.coefficients(t, dt)

An implicit 3-D solver enters ``r`` into its equations, so the outlet is
coupled inside each coefficient loop (stable with many outlets, where a
lagged explicit coupling is not). An explicit solver (the LBM patch
controller) measures ``Q`` over an interval and calls
:meth:`LumpedOutlet.advance`, which commits the step and returns the
pressure to hold for the next interval.

Models
------
:class:`RCR`
    Three-element Windkessel: ``P = P_d + R_p Q``,
    ``C dP_d/dt = Q − (P_d − P_v)/R_d``.
:class:`Coronary`
    The open-loop coronary model of Kim et al. (2010): ``R_a`` in series
    with ``C_a`` to ground, then ``R_am`` to a node joined to the
    intramyocardial pressure ``P_im(t)`` through ``C_im``, then ``R_v`` to
    the venous pressure. The state is ``[P_1, P_2 − P_im]`` (the pressures
    across ``C_a`` and ``C_im``), so ``P_im`` enters without being
    differentiated. :meth:`Coronary.from_svsolver` inverts svSolver's
    ``cort.dat`` ODE coefficients back to the five circuit parameters.

References: N. Westerhof, J.-W. Lankhaar, B. E. Westerhof, *The arterial
Windkessel*, Med. Biol. Eng. Comput. 47, 131 (2009); H. J. Kim et al.,
*Patient-specific modeling of blood flow and pressure in human coronary
arteries*, Ann. Biomed. Eng. 38, 3195 (2010).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

METHODS = ("trapezoidal", "euler")


class _Signal:
    """A constant, or a periodic ``[(t, value), ...]`` table (period = last t), linear in t."""

    def __init__(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            self.t, self.v = np.asarray(value, float).reshape(-1, 2).T.copy()
            self.const = float(self.v[0]) if self.t[-1] <= 0 else None
        else:
            self.const = float(value)

    def __call__(self, t: float) -> float:
        if self.const is not None:
            return self.const
        return float(np.interp(t % self.t[-1], self.t, self.v))


@dataclass
class LumpedOutlet:
    """A linear 0-D outlet model. Subclasses fill the matrices in ``__post_init__``.

    Attributes:
        method: ``"trapezoidal"`` (default) or ``"euler"`` (backward).
        x: state vector (Pa); set by :meth:`initialise`, or at rest at the
            venous pressure on first use.
        q: the outflow of the last committed step (m³/s).
        t: the time of the last committed step (s).
    """

    method: str = field(default="trapezoidal", kw_only=True)
    x: np.ndarray | None = field(default=None, kw_only=True, repr=False)
    q: float = field(default=0.0, kw_only=True, repr=False)
    t: float = field(default=0.0, kw_only=True, repr=False)

    # filled by subclasses
    A: np.ndarray = field(init=False, repr=False)
    Bq: np.ndarray = field(init=False, repr=False)
    Bu: np.ndarray = field(init=False, repr=False)
    Cx: np.ndarray = field(init=False, repr=False)
    Dq: float = field(init=False, repr=False)
    Du: np.ndarray = field(init=False, repr=False)

    def inputs(self, t: float) -> np.ndarray:
        """``u(t) = [P_v, P_im(t)]`` in Pa."""
        raise NotImplementedError

    # ------------------------------------------------------------ steady state

    def steady_state(self, q: float, t: float | None = None) -> np.ndarray:
        """The state at rest under constant outflow ``q`` and inputs frozen at ``t``."""
        u = self.inputs(self.t if t is None else t)
        return np.linalg.solve(self.A, -(self.Bq * q + self.Bu @ u))

    def resistance(self) -> float:
        """Total steady resistance ``dP/dQ`` at rest (Pa·s/m³)."""
        return float(self.Dq - self.Cx @ np.linalg.solve(self.A, self.Bq))

    def initialise(self, q: float = 0.0, t: float = 0.0) -> None:
        """Start at the steady state for outflow ``q`` at time ``t``."""
        self.x, self.q, self.t = self.steady_state(q, t), float(q), float(t)

    def _ensure(self) -> None:
        if self.x is None:
            self.initialise(0.0, self.t)

    # ------------------------------------------------------------ stepping

    def _step(self, t_new: float, dt: float):
        """``(x0, x1)``: the end state is ``x0 + x1 Q(t_new)``, affine in the end flow."""
        if self.method not in METHODS:
            raise ValueError(f"method {self.method!r}; expected one of {METHODS}")
        self._ensure()
        key = (self.method, float(dt))
        cache = self.__dict__.setdefault("_inv", {})
        if key not in cache:
            h = dt if self.method == "euler" else 0.5 * dt
            minv = np.linalg.inv(np.eye(len(self.x)) - h * self.A)
            cache.clear()
            cache[key] = (h, minv, minv @ (h * self.Bq))
        h, minv, x1 = cache[key]
        u1 = self.inputs(t_new)
        if self.method == "euler":
            rhs = self.x + h * (self.Bu @ u1)
        else:
            u0 = self.inputs(t_new - dt)
            rhs = self.x + h * (self.A @ self.x + self.Bq * self.q + self.Bu @ (u0 + u1))
        return minv @ rhs, x1, u1

    def coefficients(self, t_new: float, dt: float) -> tuple[float, float]:
        """``(a, r)`` with ``P(t_new) = a + r Q(t_new)`` for the step of length ``dt``.

        Does not change the state; call :meth:`advance` once the step's flow is final.
        """
        x0, x1, u1 = self._step(t_new, dt)
        a = float(self.Cx @ x0 + self.Du @ u1)
        r = float(self.Cx @ x1 + self.Dq)
        return a, r

    def advance(self, q_new: float, t_new: float, dt: float) -> float:
        """Commit the step ending at ``t_new`` with end outflow ``q_new``; return ``P(t_new)``."""
        x0, x1, u1 = self._step(t_new, dt)
        self.x = x0 + x1 * q_new
        self.q, self.t = float(q_new), float(t_new)
        return self.pressure()

    def pressure(self, q: float | None = None) -> float:
        """Patch pressure (Pa) for the current state, at outflow ``q`` (default: the last one)."""
        self._ensure()
        q = self.q if q is None else q
        return float(self.Cx @ self.x + self.Dq * q + self.Du @ self.inputs(self.t))

    def impedance(self, omega: float) -> complex:
        """Input impedance ``P/Q`` at angular frequency ``omega`` (rad/s), inputs held."""
        n = len(self.A)
        g = np.linalg.solve(1j * omega * np.eye(n) - self.A, self.Bq)
        return complex(self.Cx @ g + self.Dq)


@dataclass
class RCR(LumpedOutlet):
    """Three-element Windkessel.

    Args:
        rp: proximal resistance R_p (Pa·s/m³).
        c: compliance C (m³/Pa).
        rd: distal resistance R_d (Pa·s/m³).
        pv: venous (distal reference) pressure P_v (Pa), or a periodic
            ``[(t, Pa), ...]`` table.
    """

    rp: float = 0.0
    c: float = 0.0
    rd: float = 0.0
    pv: float | list = 0.0

    def __post_init__(self):
        if self.rd <= 0 or self.c <= 0:
            raise ValueError("RCR needs rd > 0 and c > 0")
        rc = self.rd * self.c
        self.A = np.array([[-1.0 / rc]])
        self.Bq = np.array([1.0 / self.c])
        self.Bu = np.array([[1.0 / rc, 0.0]])
        self.Cx = np.array([1.0])
        self.Dq = float(self.rp)
        self.Du = np.zeros(2)
        self._pv = _Signal(self.pv)

    @classmethod
    def from_tuple(cls, rcr, pv: float | list = 0.0, **kw) -> RCR:
        """From ``(Rp, C, Rd)``, the order :class:`zvcfd.boundary.Patch` uses."""
        rp, c, rd = rcr
        return cls(rp=rp, c=c, rd=rd, pv=pv, **kw)

    def inputs(self, t: float) -> np.ndarray:
        return np.array([self._pv(t), 0.0])


@dataclass
class Coronary(LumpedOutlet):
    """Open-loop coronary outlet (Kim et al. 2010).

    Args:
        ra: arterial resistance R_a (Pa·s/m³).
        ca: arterial compliance C_a (m³/Pa).
        ram: micro-arterial resistance R_am (Pa·s/m³).
        cim: intramyocardial compliance C_im (m³/Pa).
        rv: venous resistance R_v, including any micro-venous part (Pa·s/m³).
        pim: intramyocardial pressure, a constant (Pa) or a periodic
            ``[(t, Pa), ...]`` table.
        pv: venous pressure (Pa).
    """

    ra: float = 0.0
    ca: float = 0.0
    ram: float = 0.0
    cim: float = 0.0
    rv: float = 0.0
    pim: float | list = 0.0
    pv: float = 0.0

    def __post_init__(self):
        if min(self.ca, self.ram, self.cim, self.rv) <= 0 or self.ra < 0:
            raise ValueError("coronary needs ca, ram, cim, rv > 0 and ra >= 0")
        ga, gv = 1.0 / self.ram, 1.0 / self.rv
        self.A = np.array([[-ga / self.ca, ga / self.ca],
                           [ga / self.cim, -(ga + gv) / self.cim]])
        self.Bq = np.array([1.0 / self.ca, 0.0])
        # u = [P_v, P_im]; state y = P_2 - P_im, so P_2 = y + P_im
        self.Bu = np.array([[0.0, ga / self.ca],
                             [gv / self.cim, -(ga + gv) / self.cim]])
        self.Cx = np.array([1.0, 0.0])
        self.Dq = float(self.ra)
        self.Du = np.zeros(2)
        self._pim = _Signal(self.pim)

    def inputs(self, t: float) -> np.ndarray:
        return np.array([float(self.pv), self._pim(t)])

    def ode_coefficients(self) -> dict[str, tuple[float, float, float]]:
        """svSolver's form ``p0 P + p1 P' + p2 P'' = q0 Q + q1 Q' + q2 Q'' + b1 P_im'``
        (P_v = 0), in the units of the parameters."""
        ra, ca, ram, cim, rv = self.ra, self.ca, self.ram, self.cim, self.rv
        p = (1.0, rv * cim + ca * (ram + rv), ca * ram * rv * cim)
        q = (ra + ram + rv, ra * rv * cim + ra * ca * (ram + rv) + ram * rv * cim,
             ra * ca * ram * rv * cim)
        return {"p": p, "q": q, "b": (0.0, rv * cim, 0.0)}

    @classmethod
    def from_svsolver(cls, q, p, b, *, pim=0.0, pv: float = 0.0, check: float = 1e-3,
                      **kw) -> Coronary:
        """Invert svSolver ``cort.dat`` coefficients to (R_a, C_a, R_am, C_im, R_v).

        ``q, p, b`` are ``(q0, q1, q2)``, ``(p0, p1, p2)``, ``(b0, b1, b2)`` in any
        consistent units (svSolver uses cgs); the parameters come out in the same
        units. The five parameters fix ``p1, p2, q0, q2, b1``; ``q1`` is redundant
        and is checked to relative tolerance ``check``.
        """
        q0, q1, q2 = (float(v) for v in q)
        p0, p1, p2 = (float(v) / float(p[0]) for v in p)
        q0, q1, q2 = q0 / float(p[0]), q1 / float(p[0]), q2 / float(p[0])
        b1 = float(b[1]) / float(p[0])
        if abs(b[0]) > 0 or abs(b[2]) > 0:
            raise ValueError("coronary model expects b0 = b2 = 0")
        ra = q2 / p2
        rv_cim = b1
        ca = (p1 - rv_cim) / (q0 - ra)
        ram = p2 / (ca * rv_cim)
        rv = q0 - ra - ram
        cim = rv_cim / rv
        model = cls(ra=ra, ca=ca, ram=ram, cim=cim, rv=rv, pim=pim, pv=pv, **kw)
        q1_model = model.ode_coefficients()["q"][1]
        if abs(q1_model - q1) > check * abs(q1):
            raise ValueError(f"coefficients are not a Kim coronary circuit: q1 {q1:.6g} "
                             f"vs {q1_model:.6g} from the other four")
        return model

    def scaled(self, r: float = 1.0, c: float = 1.0, p: float = 1.0) -> Coronary:
        """The same model in other units: resistances × r, compliances × c, pressures × p."""
        pim = [(t, v * p) for t, v in self.pim] if isinstance(self.pim, (list, tuple)) \
            else self.pim * p
        return Coronary(ra=self.ra * r, ca=self.ca * c, ram=self.ram * r, cim=self.cim * c,
                        rv=self.rv * r, pim=pim, pv=self.pv * p, method=self.method)


# cgs (dyn, cm, s) -> SI, for svSolver files
CGS_RESISTANCE = 1e5          # dyn·s/cm^5 -> Pa·s/m^3
CGS_COMPLIANCE = 1e-5         # cm^5/dyn -> m^3/Pa
CGS_PRESSURE = 0.1            # dyn/cm^2 -> Pa


def read_svsolver_cort(path, *, si: bool = True) -> list[Coronary]:
    """The coronary outlets of an svSolver ``cort.dat``, one per listed surface, in file order.

    With ``si`` the parameters and ``P_im`` tables are converted from cgs to SI.
    """
    vals = [ln.split() for ln in open(path).read().splitlines() if ln.strip()]
    out, i = [], 1
    while i < len(vals):
        n = int(vals[i][0])
        co = [float(v[0]) for v in vals[i + 1:i + 12]]
        table = [(float(t), float(v)) for t, v in vals[i + 12:i + 12 + n]]
        model = Coronary.from_svsolver(co[0:3], co[3:6], co[6:9], pim=table)
        if si:
            model = model.scaled(CGS_RESISTANCE, CGS_COMPLIANCE, CGS_PRESSURE)
        out.append(model)
        i += 12 + n
    return out


def read_svsolver_rcrt(path, *, si: bool = True) -> list[RCR]:
    """The RCR outlets of an svSolver ``rcrt.dat`` (``Rp, C, Rd`` and a ``P_d(t)`` table)."""
    vals = [ln.split() for ln in open(path).read().splitlines() if ln.strip()]
    out, i = [], 1
    while i < len(vals):
        n = int(vals[i][0])
        rp, c, rd = (float(vals[i + k][0]) for k in (1, 2, 3))
        table = [(float(t), float(v)) for t, v in vals[i + 4:i + 4 + n]]
        if si:
            rp, c, rd = rp * CGS_RESISTANCE, c * CGS_COMPLIANCE, rd * CGS_RESISTANCE
            table = [(t, v * CGS_PRESSURE) for t, v in table]
        pv = table[0][1] if len({v for _, v in table}) <= 1 else table
        out.append(RCR(rp=rp, c=c, rd=rd, pv=pv))
        i += 4 + n
    return out


__all__ = ["CGS_COMPLIANCE", "CGS_PRESSURE", "CGS_RESISTANCE", "Coronary", "LumpedOutlet",
           "METHODS", "RCR", "read_svsolver_cort", "read_svsolver_rcrt"]
