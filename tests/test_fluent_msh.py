import pytest

from zvcfd.io.fluent_msh import read_fluent_boundary, voxel_estimate

CUBE = """(0 "unit cube")
(2 3)
(10 (0 1 8 0 3))
(10 (1 1 8 1 3)(
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
))
(12 (0 1 1 0))
(12 (2 1 1 1 4))
(13 (0 1 6 0))
(13 (3 1 4 3 0)(
4 1 4 3 2 1 0
4 5 6 7 8 1 0
4 1 2 6 5 1 0
4 4 8 7 3 1 0
))
(13 (a 5 5 a 0)(
4 1 5 8 4 1 0
))
(13 (b 6 6 5 0)(
4 2 3 7 6 1 0
))
(45 (3 wall walls)())
(45 (10 velocity-inlet inlet)())
(45 (11 pressure-outlet outlet)())
"""


@pytest.fixture
def cube(tmp_path):
    p = tmp_path / "cube.msh"
    p.write_bytes(CUBE.replace("\n", "\r\n").encode())
    return read_fluent_boundary(str(p))


def test_counts_and_zones(cube):
    assert cube.nodes.shape == (8, 3)
    assert cube.n_cells == 1 and cube.n_faces == 6
    assert {z.kind for z in cube.zones.values()} == {"wall", "velocity-inlet", "pressure-outlet"}
    assert cube.zones[10].name == "inlet"          # (45) ids are decimal, (13) ids hex


def test_volume_and_patches(cube):
    s = cube.summary()
    assert s["volume"] == pytest.approx(1.0)
    areas = {p["kind"]: p["area"] for p in s["patches"]}
    assert areas == {"velocity-inlet": pytest.approx(1.0), "pressure-outlet": pytest.approx(1.0)}
    e = voxel_estimate(s, 0.1)
    assert e["fluid_voxels"] == pytest.approx(1000.0)
