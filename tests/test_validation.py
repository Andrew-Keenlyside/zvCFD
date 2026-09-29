"""Fast versions of the validation suite (benchmarks/validation): exact solutions and orders."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "validation"))

pytestmark = pytest.mark.gpu


def test_exact_solutions_reduce_to_known_limits():
    import exact

    z = np.linspace(0, 2, 7)
    assert np.allclose(exact.womersley_plane(z, 2.0, 1.0, 1e-7, 1.0, 0.0),
                       exact.plane_poiseuille(z, 2.0, 1.0, 1.0), rtol=1e-5)
    r = np.linspace(0, 1, 7)
    assert np.allclose(exact.womersley_pipe(r, 1.0, 1.0, 1e-7, 1.0, 0.0),
                       exact.hagen_poiseuille(r, 1.0, 1.0, 1.0), rtol=1e-5, atol=1e-12)
    y = np.linspace(-1, 1, 601)
    Y, Z = np.meshgrid(y, y)
    q = np.trapezoid(np.trapezoid(exact.square_duct(Y, Z, 1.0, 1.0, 1.0), y), y)
    assert q == pytest.approx(exact.square_duct_flux(1.0, 1.0, 1.0), rel=1e-4)
    newt = exact.gn_channel(np.array([0.0, 0.5]), 1.0, 1e-3, lambda g: 0.1 + 0 * g)
    assert np.allclose(newt, [5e-3, 3.75e-3], rtol=1e-8)
    # tabulated Sangani-Acrivos drag agrees with their series where the series holds
    phi = np.pi * exact.SC_CHI[:6] ** 3 / 6
    assert np.allclose(exact.sc_drag_series(phi), exact.SC_DRAG[:6], rtol=4e-3)


def test_womersley_channel_second_order():
    import cases

    r = cases.womersley(half_widths=(8, 16), radii=(), alphas=(4.0,))
    e = [x["l2"] for x in r["plane"]]
    assert e[1] < 2e-4
    assert np.log2(e[0] / e[1]) > 1.8


def test_taylor_green_second_order():
    import cases

    r = cases.taylor_green(sizes=(16, 32, 64))
    eq = [x for x in r["rows"] if x["start"] == "equilibrium"]
    cons = [x for x in r["rows"] if x["start"] == "consistent"]
    assert r["order_u_equilibrium"] > 1.9
    assert r["order_p_consistent"] > 1.8
    assert cons[-1]["l2_u"] < 2e-4 and eq[-1]["l2_u"] < 2e-3


def test_carreau_yasuda_channel_converges():
    import cases

    r = cases.carreau_channel(widths=(14, 30))
    e = [x["l2"] for x in r["rows"]]
    assert e[1] < 2e-3                          # 3 fixed-point iterations stalled at 6e-3
    assert r["rows"][0]["thinning"] > 5         # strongly shear-thinning regime


def test_square_duct_second_order():
    import cases

    r = cases.duct(sides=(6, 14, 30))
    assert r["order_l2"] > 1.9
    assert r["rows"][-1]["flux_ratio"] == pytest.approx(1.0, abs=2e-3)


def test_sphere_array_matches_sangani_acrivos():
    import cases

    r = cases.sphere_array(sizes=(32,), chis=(0.4, 0.6))
    for row in r["rows"]:
        assert abs(row["err"]) < 0.01                   # published TRT code: 0.45, 0.57 %
        assert abs(row["err"] - row["bogner_err"]) < 0.003
