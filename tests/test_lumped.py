"""0-D outlet models (zvcfd.lumped): steady states, frequency response, svSolver files."""

from pathlib import Path

import numpy as np
import pytest

from zvcfd.boundary import Patch
from zvcfd.lumped import RCR, Coronary, read_svsolver_cort, read_svsolver_rcrt

VMR_SIM = Path("/hdd/data/zvcfd_vmr/0066_H_CORO_H/Simulations/0002_0001")

# a coronary outlet of VMR 0066 (cort.dat, first surface), SI
KIM = dict(ra=3.7026e10, ca=1.8736e-12, ram=6.0388e10, cim=7.9333e-11, rv=2.1157e10)


def periodic_response(model, q_of_t, period, dt, cycles=30):
    """Pressure over the last cycle with end-of-step flow ``q_of_t`` (implicit coupling)."""
    model.initialise(q_of_t(0.0), 0.0)
    n = int(round(period / dt))
    t, out = 0.0, []
    for k in range(cycles * n):
        t = (k + 1) * dt
        a, r = model.coefficients(t, dt)
        p = model.advance(q_of_t(t), t, dt)
        assert p == pytest.approx(a + r * q_of_t(t), rel=1e-12)
        if k >= (cycles - 1) * n:
            out.append((t, p))
    return np.array(out)


@pytest.mark.parametrize("model", [RCR(rp=1e8, c=1e-10, rd=9e8, pv=100.0),
                                   Coronary(**KIM, pim=2e3, pv=100.0)])
def test_steady_state_is_total_resistance(model):
    q = 2e-6
    model.initialise(0.0)
    for k in range(4000):                       # 40 s: > 20 slowest time constants (1.7 s)
        p = model.advance(q, (k + 1) * 1e-2, 1e-2)
    total = 1e9 if isinstance(model, RCR) else KIM["ra"] + KIM["ram"] + KIM["rv"]
    assert model.resistance() == pytest.approx(total, rel=1e-12)
    assert p == pytest.approx(100.0 + total * q, rel=1e-6)


@pytest.mark.parametrize("method,order", [("trapezoidal", 2.0), ("euler", 1.0)])
def test_frequency_response_converges(method, order):
    """Periodic flow through the coronary model against its exact input impedance."""
    omega = 2 * np.pi
    q0, q1 = 1e-6, 5e-7
    errs = []
    for dt in (4e-3, 2e-3, 1e-3):
        m = Coronary(**KIM, pim=0.0, method=method)
        out = periodic_response(m, lambda t: q0 + q1 * np.sin(omega * t), 1.0, dt)
        wave = q1 * m.impedance(omega) * np.exp(1j * omega * out[:, 0])
        exact = q0 * m.resistance() + wave.imag
        errs.append(np.abs(out[:, 1] - exact).max() / np.ptp(exact))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert errs[-1] < (1e-4 if method == "trapezoidal" else 2e-2)
    assert rates.min() == pytest.approx(order, abs=0.15)


def test_intramyocardial_pressure_response():
    """Zero flow, sinusoidal P_im: P / P_im = s Rv Cim / (1 + p1 s + p2 s^2)."""
    omega = 2 * np.pi
    m = Coronary(**KIM, pim=[(t, 1e3 * np.sin(omega * t)) for t in np.linspace(0, 1, 4001)])
    out = periodic_response(m, lambda t: 0.0, 1.0, 5e-4)
    c = m.ode_coefficients()
    s = 1j * omega
    h = c["b"][1] * s / (c["p"][0] + c["p"][1] * s + c["p"][2] * s * s)
    exact = 1e3 * (h * np.exp(1j * omega * out[:, 0])).imag
    assert np.abs(out[:, 1] - exact).max() / np.ptp(exact) < 1e-3


def test_svsolver_coefficients_round_trip():
    m = Coronary(**KIM)
    c = m.ode_coefficients()
    back = Coronary.from_svsolver(c["q"], c["p"], c["b"])
    for k, v in KIM.items():
        assert getattr(back, k) == pytest.approx(v, rel=1e-10)
    bad = list(c["q"])
    bad[1] *= 1.1
    with pytest.raises(ValueError, match="not a Kim coronary circuit"):
        Coronary.from_svsolver(bad, c["p"], c["b"])


@pytest.mark.skipif(not (VMR_SIM / "cort.dat").exists(), reason="VMR 0066 not downloaded")
def test_read_vmr_outlet_files():
    cor = read_svsolver_cort(VMR_SIM / "cort.dat")
    assert len(cor) == 24
    first = open(VMR_SIM / "cort.dat").read().split()[2]          # q0 of surface 1, cgs
    assert cor[0].resistance() == pytest.approx(float(first) * 1e5, rel=1e-9)
    assert len(cor[0].pim) == 1001
    rcr = read_svsolver_rcrt(VMR_SIM / "rcrt.dat")
    assert len(rcr) == 1 and rcr[0].rp == pytest.approx(105.379837e5)


def test_coronary_patch():
    p = Patch("LAD", "coronary", pressure=0.0, coronary=tuple(KIM.values()), pim=1e3)
    assert p.lumped and p.flag == 3
    for k in range(4000):
        pr = p.outlet_pressure(1e-6, (k + 1) * 1e-2, 1e-2)
    assert pr == pytest.approx(sum((KIM["ra"], KIM["ram"], KIM["rv"])) * 1e-6, rel=1e-6)
    with pytest.raises(ValueError, match="coronary needs"):
        Patch("x", "coronary")
