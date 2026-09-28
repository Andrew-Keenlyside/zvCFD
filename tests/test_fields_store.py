import numpy as np
import pytest

from zvcfd.domain import BrickDomain

pytestmark = pytest.mark.store


def _rows(dom, seed=0):
    rng = np.random.default_rng(seed)
    return {f: rng.random((dom.n_bricks, 512), dtype=np.float32) for f in ("rho", "ux")}


@pytest.mark.parametrize("shard", [None, 2])
def test_brick_store_round_trip(tmp_path, porous64, shard):
    from zvcfd.io import fields as zf

    dom = BrickDomain.from_flags(porous64)
    rows = _rows(dom)
    path = tmp_path / "s.zarrvectors"
    lv = zf.create_brick_store(path, dom, voxel_size=2.5, fields=["rho", "ux"], chunk_bricks=2,
                               compressor="zstd", shard_shape=shard)
    parts = dom.partition(3, 2 * (shard or 1))
    for p in range(3):                       # three "workers", disjoint chunks (or shards)
        ids = np.flatnonzero(parts == p)
        zf.write_brick_chunks(lv, dom, {k: v[ids] for k, v in rows.items()}, voxel_size=2.5,
                              chunk_bricks=2, bricks=ids)
    zf.finalize_brick_store(lv)
    lv = zf.open_brick_level(path)
    uniq, _ = dom.chunks(2)
    batch = zf.read_brick_chunks(lv, uniq, ["rho", "ux"])
    centres = batch["vertices"].data
    ids = dom.brick_index(np.round(centres / 2.5 / 8 - 0.5).astype(int))
    assert sorted(ids.tolist()) == list(range(dom.n_bricks))
    for f in ("rho", "ux"):
        assert np.array_equal(batch[f"vertex_attributes/{f}"].data, rows[f][ids])


@pytest.mark.gpu
def test_device_read(tmp_path, porous64):
    import cupy as cp

    from zvcfd.io import fields as zf

    dom = BrickDomain.from_flags(porous64)
    rows = _rows(dom)
    path = tmp_path / "s.zarrvectors"
    lv = zf.create_brick_store(path, dom, voxel_size=1.0, fields=["rho", "ux"], chunk_bricks=4)
    zf.write_brick_chunks(lv, dom, rows, voxel_size=1.0, chunk_bricks=4)
    zf.finalize_brick_store(lv)
    uniq, _ = dom.chunks(4)
    b = zf.read_brick_chunks(zf.open_brick_level(path), uniq, ["ux"], device="cuda")
    assert isinstance(b["vertex_attributes/ux"].data, cp.ndarray)
    ids = dom.brick_index(np.round(cp.asnumpy(b["vertices"].data) / 8 - 0.5).astype(int))
    assert np.array_equal(cp.asnumpy(b["vertex_attributes/ux"].data), rows["ux"][ids])


def test_read_domain_and_snapshot(tmp_path, porous64):
    from zvcfd.io import fields as zf

    dom = BrickDomain.from_flags(porous64)
    d = zf.create_brick_store(tmp_path / "d.zarrvectors", dom, voxel_size=2.0, fields={},
                              chunk_bricks=4, flags=True)
    zf.write_brick_chunks(d, dom, {"flags": dom.flags}, voxel_size=2.0, chunk_bricks=4)
    zf.finalize_brick_store(d)
    back = zf.read_domain(tmp_path / "d.zarrvectors")
    assert np.array_equal(back.coords, dom.coords) and np.array_equal(back.flags, dom.flags)
    assert np.array_equal(back.to_dense(np.zeros((dom.n_bricks, 512))), np.zeros(dom.shape))

    rows = _rows(dom)
    s = zf.create_brick_store(tmp_path / "s.zarrvectors", dom, voxel_size=2.0,
                              fields=["rho", "ux"], chunk_bricks=4)
    zf.write_brick_chunks(s, dom, rows, voxel_size=2.0, chunk_bricks=4)
    zf.finalize_brick_store(s)
    snap = zf.read_snapshot(tmp_path / "s.zarrvectors", back)
    assert all(np.array_equal(snap[k], rows[k]) for k in rows)
