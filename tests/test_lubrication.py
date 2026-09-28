import numpy as np
import pytest

from zvcfd.solvers.lubrication import solve


@pytest.mark.parametrize("method", ["amg", "cg"])
def test_tube_flux_matches_poiseuille(tube_mask, method):
    if method == "amg":
        pytest.importorskip("pyamg")
    fixed = np.zeros_like(tube_mask)
    fixed[:, :, 0] = fixed[:, :, -1] = True
    value = np.zeros(tube_mask.shape)
    value[:, :, 0] = 1.0
    r = solve(tube_mask, fixed & tube_mask, value, method=method, mu=1.0)
    q = r.flux(axis=2, index=32)
    radius = np.sqrt(tube_mask[:, :, 0].sum() / np.pi)
    exact = np.pi * radius ** 4 / 8 * (1.0 / 63)
    assert q == pytest.approx(exact, rel=0.2)
