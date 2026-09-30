"""Finite-volume geometry: shape functions, median-dual control volumes, coupling pattern."""

import numpy as np
import pytest

from zvcfd.fv.geometry import (
    XI_NODES,
    dual_geometry,
    element_points,
    gradients,
    ip_areas,
    shape,
    topology,
)
from zvcfd.fv.pattern import coupling_stats, node_graph
from zvcfd.mesh.generate import box, tube

KINDS = ("tet", "pyramid", "wedge", "hex")


@pytest.mark.parametrize("kind", KINDS)
def test_shape_functions(kind):
    n = len(XI_NODES[kind])
    N, dN = shape(kind, XI_NODES[kind])
    np.testing.assert_allclose(N, np.eye(n), atol=1e-14)
    rng = np.random.default_rng(1)
    xi = rng.uniform(-0.3, 0.3, (20, 3)) + (0.25 if kind in ("tet", "wedge") else 0.0) * \
        np.array([1, 1, 0 if kind == "wedge" else 1])
    N, dN = shape(kind, xi)
    np.testing.assert_allclose(N.sum(-1), 1.0, atol=1e-14)
    np.testing.assert_allclose(dN.sum(-2), 0.0, atol=1e-14)
    h = 1e-6
    for d in range(3):
        e = np.zeros(3)
        e[d] = h
        num = (shape(kind, xi + e)[0] - shape(kind, xi - e)[0]) / (2 * h)
        np.testing.assert_allclose(dN[..., d], num, atol=1e-8)


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_reproduce_linear_fields(kind):
    m = box((3, 3, 3), kind=kind, perturb=0.25, seed=3)
    g = np.array([0.4, -1.3, 2.1])
    for k, e in m.elements.items():
        G, detJ = gradients(m.nodes[e], topology(k))
        assert (detJ > 0).all()
        grad = np.einsum("eink,en->eik", G, m.nodes[e] @ g)
        np.testing.assert_allclose(grad, np.broadcast_to(g, grad.shape), atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_topology_orientation_and_counts(kind):
    t = topology(kind)
    # two flux points (planar halves) per element edge
    assert t.n_ip == 2 * {"tet": 6, "pyramid": 8, "wedge": 9, "hex": 12}[kind]
    P = element_points(XI_NODES[kind][None], t)
    A = ip_areas(P, t)[0]
    d = XI_NODES[kind][t.edges[:, 1]] - XI_NODES[kind][t.edges[:, 0]]
    assert (np.einsum("ik,ik->i", A, d) > 0).all()


MESHES = {
    **{f"box-{k}": (lambda k=k: box((5, 4, 3), kind=k, perturb=0.25)) for k in KINDS},
    **{f"tube-{k}": (lambda k=k: tube(0.5, 2.0, n_axial=5, kind=k, layers=2, growth=0.7,
                                      perturb=0.1)) for k in ("hex", "wedge", "tet", "mixed")},
}


@pytest.mark.parametrize("name", sorted(MESHES))
def test_dual_volumes_and_closure(name):
    m = MESHES[name]()
    g = dual_geometry(m)
    assert g.report["volume_rel_diff"] < 1e-13
    assert g.report["closure_max_rel"] < 1e-13
    assert (g.node_volume > 0).all()
    assert g.node_volume.sum() == pytest.approx(m.volume(), rel=1e-13)


def _discrete_gradient(m, p):
    """(1/V) sum_ip p_ip A_ip over each control volume (interior nodes only are meaningful)."""
    N = m.n_nodes
    S = np.zeros((N, 3))
    for k, e in m.elements.items():
        t = topology(k)
        A = ip_areas(element_points(m.nodes[e], t), t)
        pip = np.einsum("in,en->ei", t.N_ip, p[e])
        a, b = e[:, t.edges[:, 0]], e[:, t.edges[:, 1]]
        for c in range(3):
            S[:, c] += np.bincount(a.ravel(), (pip * A[..., c]).ravel(), N)
            S[:, c] -= np.bincount(b.ravel(), (pip * A[..., c]).ravel(), N)
    return S / dual_geometry(m).node_volume[:, None]


@pytest.mark.parametrize("kind,perturb", [("tet", 0.25)] + [(k, 0.0) for k in KINDS])
def test_linear_pressure_gradient_exact_on_affine_elements(kind, perturb):
    """Tetrahedra always, other types when undistorted: sum p_ip A_ip = V grad p."""
    m = box((5, 5, 5), kind=kind, perturb=perturb)
    g = np.array([0.3, -1.1, 0.7])
    G = _discrete_gradient(m, m.nodes @ g)
    bnd = np.zeros(m.n_nodes, bool)
    for z in m.zones.values():
        bnd[z.faces[z.faces >= 0]] = True
    np.testing.assert_allclose(G[~bnd], np.broadcast_to(g, G[~bnd].shape), atol=1e-12)


def test_coupling_pattern():
    s = coupling_stats(node_graph(box((6, 6, 6), kind="hex"))[0])
    assert s["max"] == 27 and s["rows"] == 7 ** 3
    s = coupling_stats(node_graph(box((6, 6, 6), kind="tet"))[0])
    assert s["max"] == 15                             # Kuhn-type split: 14 neighbours + self
    indptr, idx = node_graph(tube(0.5, 1.0, kind="mixed", n_axial=3))
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    key = set(zip(rows.tolist(), idx.tolist()))
    assert all((j, i) in key for i, j in key)         # symmetric
    assert all((i, i) in key for i in range(len(indptr) - 1))


def test_renumbering_preserves_geometry_and_narrows_bandwidth():
    from zvcfd.fv.pattern import bandwidth, node_order

    m = tube(0.5, 4.0, kind="mixed", n_axial=12, perturb=0.1)
    scr = m.renumbered(np.random.default_rng(0).permutation(m.n_nodes))
    r = scr.renumbered(node_order(scr))
    assert bandwidth(*node_graph(r)) < bandwidth(*node_graph(scr)) / 5
    assert r.volume() == pytest.approx(m.volume(), rel=1e-14)
    g = dual_geometry(r)
    assert g.report["closure_max_rel"] < 1e-13
    c = r.renumbered(node_order(r, chunk=1.0))
    chunk = np.floor((c.nodes - c.nodes.min(0)) / 1.0).astype(int)
    key = [tuple(k) for k in chunk]
    # nodes of each chunk are contiguous
    seen, last = set(), None
    for k in key:
        if k != last:
            assert k not in seen
            seen.add(k)
            last = k
