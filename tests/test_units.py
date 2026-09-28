import pytest

from zvcfd.units import Lattice, level_factors


def test_round_trips():
    lat = Lattice(dx=10e-6, nu=3.5e-6, tau=0.8)
    assert lat.u_lattice(lat.velocity(0.05)) == pytest.approx(0.05)
    assert lat.drho_lattice(lat.pressure(0.01)) == pytest.approx(0.01)


def test_diffusive_scaling():
    lat = Lattice(dx=10e-6, nu=3.5e-6)
    c = lat.coarsened(2)
    assert c.dt == pytest.approx(4 * lat.dt)
    assert c.u_lattice(0.1) == pytest.approx(2 * lat.u_lattice(0.1))
    assert level_factors(1) == {"velocity": 2.0, "drho": 4.0, "force": 8.0, "dt": 4.0}
