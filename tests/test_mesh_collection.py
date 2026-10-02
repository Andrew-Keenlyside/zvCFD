"""Mesh collections (``.zvmesh``): any mesh imported once into Zarr Vectors, both solvers
reading their geometry back from it."""

import warnings

import numpy as np
import pytest

from zvcfd.mesh.fluent import match_rows, write_fluent_mesh
from zvcfd.mesh.generate import tube

building = pytest.importorskip("zarr_vectors.building")
pytestmark = pytest.mark.skipif(not hasattr(building, "write_link_attributes"),
                                reason="needs zarr-vectors building links")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def pipe_msh(tmp_path_factory):
    m = tube(1.0, 4.0, n_core=2, n_ring=2, n_axial=8, kind="mixed", layers=1)
    path = tmp_path_factory.mktemp("mesh") / "pipe.msh"
    write_fluent_mesh(m, path)
    return path


def _same_zones(a, b):
    assert {z.name: z.kind for z in a.zones.values()} == {z.name: z.kind for z in b.zones.values()}
    by_name = {z.name: z for z in b.zones.values()}
    for z in a.zones.values():                   # same faces and owners; order may differ
        w = by_name[z.name]
        assert z.n_faces == w.n_faces
        h = match_rows(z.faces, w.faces)
        assert (h >= 0).all()
        np.testing.assert_array_equal(z.cells[h], w.cells)


def test_import_and_read_back(pipe_msh, tmp_path):
    from zvcfd.io.mesh_collection import (
        collection_info,
        import_mesh,
        is_mesh_collection,
        read_mesh,
        read_source,
    )

    out = import_mesh(pipe_msh, tmp_path / "pipe.zvmesh", unit="mm", log=lambda s: None)
    assert is_mesh_collection(out) and not is_mesh_collection(pipe_msh)
    info = collection_info(out)
    assert info["unit"] == "meter" and info["volume"]
    src = read_source(pipe_msh, unit="mm")
    m = read_mesh(out)
    assert m.unit == "meter" and np.isclose(m.nodes[:, 2].max(), 4e-3)
    np.testing.assert_array_equal(m.nodes, src.nodes)
    assert m.counts() == src.counts()
    for k in src.elements:
        np.testing.assert_array_equal(m.elements[k], src.elements[k])
    _same_zones(m, src)


def test_surface_only(pipe_msh, tmp_path):
    from zvcfd.io.mesh_collection import import_mesh, read_mesh, read_source, read_surface

    out = import_mesh(pipe_msh, tmp_path / "s.zvmesh", unit="mm", surface_only=True,
                      log=lambda s: None)
    tri, zone, zones = read_surface(out)
    src = read_source(pipe_msh, unit="mm")
    n = sum(len(z.triangles()) for z in src.zones.values())
    assert tri.shape == (n, 3, 3) and len(zone) == n
    assert {z["name"] for z in zones} == {"inlet", "outlet", "wall"}
    for z in src.zones.values():                 # every source triangle, by coordinates
        want = np.sort(src.nodes[z.triangles()].reshape(-1, 9), 1)
        got = np.sort(tri[zone == z.zone].reshape(-1, 9), 1)
        np.testing.assert_array_equal(np.unique(want, axis=0), np.unique(got, axis=0))
    with pytest.raises(ValueError, match="surface-only"):
        read_mesh(out)


def test_cached_import_is_reused(pipe_msh, tmp_path):
    from zvcfd.io.mesh_collection import cached_import, resolve

    lines = []
    a = cached_import(pipe_msh, tmp_path, unit="mm", log=lines.append)
    b = cached_import(pipe_msh, tmp_path, unit="mm", log=lines.append)
    c = cached_import(pipe_msh, tmp_path, unit="mm", surface_only=True, log=lines.append)
    assert a == b == c                           # a full import serves surface requests
    assert sum("importing" in s for s in lines) == 1
    assert cached_import(pipe_msh, tmp_path, unit="cm", log=lines.append) != a
    assert resolve(a, tmp_path / "unused") == a
    s = cached_import(pipe_msh, tmp_path / "s", unit="mm", surface_only=True, log=lines.append)
    with pytest.raises(ValueError, match="surface-only"):
        resolve(s, tmp_path)


def test_voxelised_store_matches_direct(tmp_path):
    """The voxel solver's geometry from the store equals the direct Fluent voxelisation."""
    from test_fluent_msh import CUBE

    from zvcfd.geometry import voxelize_fluent
    from zvcfd.io.mesh_collection import import_mesh, voxelize_collection

    msh = tmp_path / "cube.msh"
    msh.write_text(CUBE)
    out = import_mesh(msh, unit="mm", surface_only=True, log=lambda s: None)
    assert out == tmp_path / "cube.zvmesh"
    d0, b0, g0, r0 = voxelize_fluent(str(msh), 0.05, unit_scale=1e-3)
    d1, b1, g1, r1 = voxelize_collection(out, 50e-6)
    assert r0["fluid_voxels"] == r1["fluid_voxels"] == 8000
    # same voxels in space: the grids share an origin; the mm grid rounds its box up a brick
    np.testing.assert_allclose(g1.origin, g0.origin * 1e-3, rtol=1e-12)

    def dense(d, b):
        n = d.n_bricks * d.cells_per_brick
        patch, links = np.zeros(n, np.int64), np.zeros(n, np.int64)
        patch[b.cells] = b.patch + 1
        links[b.link_cells] = b.link_mask.astype(np.int64) * 100 + b.link_patch
        return [d.to_dense(d.flags.astype(np.int64), fill=-1),
                d.to_dense(patch.reshape(d.n_bricks, -1)),
                d.to_dense(links.reshape(d.n_bricks, -1))]

    s = tuple(slice(0, k) for k in d1.shape)
    for a, b in zip(dense(d0, b0), dense(d1, b1)):
        np.testing.assert_array_equal(a[s], b)
        assert (np.delete(a.reshape(-1), np.ravel_multi_index(np.indices(d1.shape).reshape(3, -1),
                                                              d0.shape)) <= 0).all()
    assert [p.name for p in b0.patches] == [p.name for p in b1.patches]
    for p0, p1 in zip(b0.patches, b1.patches):
        assert p0.area == pytest.approx(p1.area, rel=1e-12)


def test_cli_import_and_info(pipe_msh, tmp_path, capsys):
    from zvcfd.cli import main

    out = tmp_path / "c.zvmesh"
    assert main(["import-mesh", str(pipe_msh), "--out", str(out), "--unit", "mm"]) == 0
    assert main(["info", str(out)]) == 0
    text = capsys.readouterr().out
    assert "mesh collection c.zvmesh" in text and "zvcfd:fv-mesh" in text


def test_meshio_source(tmp_path):
    meshio = pytest.importorskip("meshio")
    from zvcfd.io.mesh_collection import import_mesh, read_mesh

    m = tube(1.0, 3.0, n_core=2, n_ring=2, n_axial=4, kind="tet")
    cells = [("tetra", m.elements["tet"])]
    tags = [np.zeros(len(m.elements["tet"]), int)]
    for zid, z in m.zones.items():
        if z.name in ("inlet", "outlet"):
            cells.append(("triangle", z.triangles()))
            tags.append(np.full(len(z.triangles()), zid))
    path = tmp_path / "pipe.msh"
    names = {z.name: [zid, 2] for zid, z in m.zones.items() if z.name != "wall"}
    meshio.write(str(path), meshio.Mesh(m.nodes, cells, cell_data={"gmsh:physical": tags},
                                        field_data=names), file_format="gmsh22", binary=False)
    r = read_mesh(import_mesh(path, unit="mm", log=lambda s: None))
    assert {z.name for z in r.zones.values()} == {"inlet", "outlet", "wall"}
    assert sum(z.n_faces for z in r.zones.values()) == sum(z.n_faces for z in m.zones.values())
