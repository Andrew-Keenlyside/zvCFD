"""Unstructured meshes in Zarr Vectors stores (zvcfd:fv-mesh and the boundary mesh store)."""

import warnings

import numpy as np
import pytest

from zvcfd.mesh.fluent import match_rows
from zvcfd.mesh.generate import box, tube

building = pytest.importorskip("zarr_vectors.building")
pytestmark = pytest.mark.skipif(not hasattr(building, "write_link_attributes"),
                                reason="needs zarr-vectors building links")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parametrize("which", ["mixed", "hex", "pyramid"])
def test_volume_store_round_trip(tmp_path, which):
    from zvcfd.io.mesh_store import read_mesh_store, write_mesh_store

    m = {"mixed": lambda: tube(0.5, 2.0, kind="mixed", n_axial=6, perturb=0.1),
         "hex": lambda: box((3, 3, 3), kind="hex"),
         "pyramid": lambda: box((2, 2, 2), kind="pyramid")}[which]()
    info = write_mesh_store(tmp_path / "v.zarrvectors", m, chunk=0.4)
    assert info["crossing_elements"] > 0 and info["chunks"] > 1
    r = read_mesh_store(tmp_path / "v.zarrvectors")
    assert r.counts() == m.counts()
    np.testing.assert_array_equal(r.nodes, m.nodes)
    for k in m.elements:
        np.testing.assert_array_equal(r.elements[k], m.elements[k])


def test_boundary_store_round_trip(tmp_path):
    from zvcfd.io.mesh_store import (
        read_boundary_store,
        read_mesh_store,
        write_boundary_store,
        write_mesh_store,
        zones_from_boundary,
    )

    m = tube(0.5, 2.0, kind="mixed", n_axial=6, perturb=0.1)
    write_mesh_store(tmp_path / "v.zarrvectors", m, chunk=0.5)
    write_boundary_store(tmp_path / "b.zarrvectors", m, chunk=0.5)
    r = read_mesh_store(tmp_path / "v.zarrvectors")
    r.zones = zones_from_boundary(r, read_boundary_store(tmp_path / "b.zarrvectors"))
    assert {z.name: (z.kind, z.n_faces) for z in r.zones.values()} == \
        {z.name: (z.kind, z.n_faces) for z in m.zones.values()}
    for k in m.zones:
        assert (match_rows(r.zones[k].faces, m.zones[k].faces) >= 0).all()
