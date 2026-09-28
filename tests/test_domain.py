import numpy as np

from zvcfd.domain import NEIGHBOUR_OFFSETS, SOLID, BrickDomain, morton3


def test_active_bricks_and_fill(porous64):
    dom = BrickDomain.from_flags(porous64)
    assert dom.shape == (64, 64, 64)
    assert dom.flags.shape == (dom.n_bricks, 512)
    assert dom.fluid_cells == int((porous64 != SOLID).sum())
    assert 0 < dom.fill <= 1


def test_solid_bricks_are_dropped():
    flags = np.ones((32, 32, 32), np.uint8)
    flags[:8, :8, :8] = 0
    dom = BrickDomain.from_flags(flags)
    assert dom.n_bricks == 1 and dom.fluid_cells == 512


def test_neighbour_table_is_symmetric(porous64):
    dom = BrickDomain.from_flags(porous64, periodic=True)
    for k, off in enumerate(NEIGHBOUR_OFFSETS):
        back = int(np.flatnonzero((NEIGHBOUR_OFFSETS == -off).all(1))[0])
        nb = dom.neighbours[:, k]
        ok = nb >= 0
        assert np.array_equal(dom.neighbours[nb[ok], back], np.flatnonzero(ok))


def test_non_periodic_edges_have_no_neighbour():
    dom = BrickDomain.from_flags(np.zeros((16, 16, 16), np.uint8))
    corner = dom.brick_index(np.array([[0, 0, 0]]))[0]
    assert dom.neighbours[corner, 0] == -1          # (-1, -1, -1)
    assert dom.neighbours[corner, 13] == corner      # self


def test_partition_owns_whole_chunks_and_balances():
    rng = np.random.default_rng(0)
    active = rng.random((16, 16, 16)) < 0.3
    dom = BrickDomain.from_brick_mask(active)
    parts = dom.partition(8, chunk_bricks=4)
    uniq, inv = dom.chunks(4)
    for c in range(len(uniq)):
        assert len(np.unique(parts[inv == c])) == 1          # a chunk has one owner
    work = np.bincount(parts, minlength=8)
    assert work.min() > 0 and work.max() / work.mean() < 1.6


def test_halo_is_foreign_and_adjacent():
    dom = BrickDomain.from_brick_mask(np.ones((8, 8, 8), bool))
    parts = dom.partition(2, chunk_bricks=4)
    h = dom.halo(parts, 0)
    assert len(h) and (parts[h] == 1).all()


def test_morton_orders_octants():
    c = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    assert list(np.argsort(morton3(c))) == [0, 1, 2, 3, 4]


def test_from_bricks_and_to_dense(porous64):
    dom = BrickDomain.from_flags(porous64)
    rng = np.random.default_rng(1)
    perm = rng.permutation(dom.n_bricks)
    again = BrickDomain.from_bricks(dom.shape, dom.coords[perm], dom.flags[perm])
    assert np.array_equal(again.coords, dom.coords)
    assert np.array_equal(again.neighbours, dom.neighbours)
    vol = dom.to_dense(np.ones((dom.n_bricks, 512), np.float32))
    assert np.array_equal(vol == 1, porous64 == 0)
