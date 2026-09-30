"""Unstructured meshes: generators, conformity, Fluent read/write round trips, VTK order."""

import importlib.util

import numpy as np
import pytest

from zvcfd.mesh import read_fluent_mesh, to_vtk, write_fluent_mesh
from zvcfd.mesh.generate import box, tube

TUBE_KINDS = ("hex", "wedge", "tet", "mixed")


def _polygon_area(n_sides, r):
    return 0.5 * n_sides * r * r * np.sin(2 * np.pi / n_sides)


@pytest.mark.parametrize("kind", ["hex", "wedge", "tet", "pyramid"])
def test_box_is_conforming_and_exact(kind):
    m = box((4, 3, 5), (1.0, 2.0, 3.0), kind=kind, perturb=0.2)
    fx = m.faces()                                  # raises if not conforming
    assert m.volume() == pytest.approx(6.0, rel=1e-13)
    assert all(v > 0 for k in m.elements for v in [m.element_volumes(k).min()])
    zone_faces = sum(z.n_faces for z in m.zones.values())
    assert zone_faces == int((fx["c1"] < 0).sum())
    assert sorted(z.name for z in m.zones.values()) == sorted(
        ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])


@pytest.mark.parametrize("kind", TUBE_KINDS)
def test_tube_is_conforming_and_exact(kind):
    n_core, L, R = 4, 2.0, 0.5
    m = tube(R, L, n_core=n_core, n_ring=4, n_axial=5, kind=kind, layers=2, growth=0.7,
             perturb=0.1)
    m.faces()
    # the wall is the polygon through the 4 n_core wall nodes
    assert m.volume() == pytest.approx(_polygon_area(4 * n_core, R) * L, rel=1e-13)
    names = {z.name: z.kind for z in m.zones.values()}
    assert names == {"inlet": "velocity-inlet", "outlet": "pressure-outlet", "wall": "wall"}
    if kind == "mixed":
        assert set(m.counts()) == {"tet", "wedge"}


def test_mixed_tube_layers_are_radial_wedges():
    """Inflation layers: wedge triangles parallel to the wall, as in the coronary mesh."""
    m = tube(0.5, 2.0, kind="mixed", n_axial=4, layers=2)
    x = m.nodes[m.elements["wedge"]]
    r = np.hypot(x[..., 0], x[..., 1])
    # bottom and top triangles each lie on one radius; the top is outside the bottom
    assert np.ptp(r[:, :3], axis=1).max() > 0          # butterfly rings are not circles...
    assert (r[:, 3:].mean(1) > r[:, :3].mean(1)).all()  # ...but the wedges point outwards


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("which", ["mixed", "pyramid", "hex"])
def test_fluent_round_trip(tmp_path, which, binary):
    m = {"mixed": lambda: tube(0.5, 2.0, kind="mixed", n_axial=5, perturb=0.1),
         "pyramid": lambda: box((3, 3, 3), kind="pyramid"),
         "hex": lambda: box((3, 4, 5), kind="hex")}[which]()
    path = tmp_path / "m.msh"
    write_fluent_mesh(m, path, binary=binary)
    r = read_fluent_mesh(path)
    assert r.counts() == m.counts()
    np.testing.assert_array_equal(r.nodes, m.nodes)
    for k in m.elements:
        a = np.sort(np.sort(r.elements[k], 1), 0)
        b = np.sort(np.sort(m.elements[k], 1), 0)
        np.testing.assert_array_equal(a, b)
        assert (r.element_volumes(k) > 0).all()
    assert {z.name: z.n_faces for z in r.zones.values()} == \
        {z.name: z.n_faces for z in m.zones.values()}
    assert r.volume() == pytest.approx(m.volume(), rel=1e-13)
    assert r.meta["c0_side"] == 1.0                    # Fluent: right-hand normal into c0


def test_reader_orients_from_geometry(tmp_path):
    """Positive element volumes and outward boundary faces, from the geometry alone."""
    from zvcfd.mesh.fluent import _normals

    m = tube(0.5, 2.0, kind="mixed", n_axial=3)
    path = tmp_path / "m.msh"
    write_fluent_mesh(m, path)
    r = read_fluent_mesh(path)
    for k in r.elements:
        assert (r.element_volumes(k) > 0).all()
    cent = r.all_centroids()
    for z in r.zones.values():
        n, fc = _normals(r.nodes, z.faces)
        assert (np.einsum("ij,ij->i", n, fc - cent[z.cells]) > 0).all()


@pytest.mark.skipif(importlib.util.find_spec("pyvista") is None, reason="needs pyvista")
@pytest.mark.parametrize("kind", ["hex", "wedge", "tet", "pyramid"])
def test_vtk_order(kind):
    import pyvista as pv

    m = box((2, 2, 2), kind=kind, perturb=0.1)
    cells, types = to_vtk(m)
    v = pv.UnstructuredGrid(cells, types, m.nodes).compute_cell_sizes()["Volume"]
    assert (v > 0).all()
    assert v.sum() == pytest.approx(m.volume(), rel=1e-6)


@pytest.mark.skipif(importlib.util.find_spec("pyvista") is None, reason="needs pyvista")
def test_vtu_round_trip(tmp_path):
    from zvcfd.mesh.vtk import read_vtu_mesh, write_vtu

    m = tube(0.5, 2.0, kind="mixed", n_axial=4, perturb=0.1)
    write_vtu(m, tmp_path / "m.vtu")
    r = read_vtu_mesh(tmp_path / "m.vtu")
    assert r.counts() == m.counts()
    for k in m.elements:
        np.testing.assert_array_equal(r.elements[k], m.elements[k])


VMR = "/hdd/data/zvcfd_vmr/0066_H_CORO_H/Simulations/0002_0001/mesh-complete/"


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("pyvista") is None
                    or not __import__("os").path.exists(VMR), reason="needs pyvista and VMR 0066")
def test_read_simvascular_mesh():
    from zvcfd.mesh.vtk import read_vtu_mesh

    m = read_vtu_mesh(VMR + "mesh-complete.mesh.vtu", VMR, unit="cm")
    assert m.counts() == {"tet": 3831813} and m.n_nodes == 701795
    kinds = [z.kind for z in m.zones.values()]
    assert kinds.count("wall") == 1 and kinds.count("velocity-inlet") == 1
    assert kinds.count("pressure-outlet") == 25                  # 24 coronary + aorta
    fx = m.faces()
    assert int((fx["c1"] < 0).sum()) == sum(z.n_faces for z in m.zones.values())
