"""Element colouring for the GPU kernels (CPU only): valid, deterministic, near the bound."""

import numpy as np

from zvcfd.fv.gpu import _colour_first_fit, _colour_greedy, _first_zero_bit, colour_elements
from zvcfd.mesh.generate import delaunay_box, tube


def _valid(e, cols):
    return sum(len(c) for c in cols) == len(e) and \
        all(len(np.unique(e[c].ravel())) == e[c].size for c in cols)


def test_first_zero_bit():
    m = np.array([[0, 0], [1, 0], [0b1011, 0], [np.iinfo(np.uint64).max, 0b11]], np.uint64)
    np.testing.assert_array_equal(_first_zero_bit(m), [0, 1, 2, 66])


def test_first_fit_colouring():
    m = tube(0.5, 2.0, n_core=8, n_ring=8, n_axial=48, kind="mixed", layers=4, growth=0.85)
    for e in m.elements.values():
        cols = colour_elements(e, m.n_nodes, greedy_limit=0)
        assert _valid(e, cols)
        bound = np.bincount(e.ravel()).max()          # elements meeting at one node
        assert bound <= len(cols) <= 1.3 * len(_colour_greedy(e, m.n_nodes))
        again = colour_elements(e, m.n_nodes, greedy_limit=0)
        assert all(np.array_equal(a, b) for a, b in zip(cols, again))


def test_first_fit_matches_greedy_quality():
    m = delaunay_box((8, 8, 8), seed=3)
    e = m.elements["tet"]
    assert _valid(e, _colour_first_fit(e, m.n_nodes, 0))
    assert len(_colour_first_fit(e, m.n_nodes, 0)) <= 1.3 * len(_colour_greedy(e, m.n_nodes))
