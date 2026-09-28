"""Parallel field-snapshot I/O: Zarr Vectors brick stores vs plain Zarr vs Icechunk.

Simulates N GPU workers (processes) each writing the chunks it owns of
one snapshot (rho, ux, uy, uz as float32) of a sparse brick domain, then N
readers reading it back after the page cache is flushed and dropped for
the store's files. Variants:

  zv-flat-{none,zstd}   brick store, one file per chunk cell per field
  zv-shard2-zstd        brick store, 2x2x2 cells per shard; workers own whole shards
  zarr-dense-{none,zstd}  OME-Zarr-style dense float32 arrays, 64^3 chunks in
                          shards the size of a store chunk; empty chunks skipped
  icechunk-dense-zstd   the dense layout inside an Icechunk repo on local disk:
                        coordinator commits arrays, forks; workers write; merge; commit

    python benchmarks/bench_io.py --root /path/on/target/fs [--workers 8]

Run on the target filesystem: page cache, NVMe vs Lustre and CPU count
all change the answer. Needs zarr-vectors on the gpu-backend branch on
PYTHONPATH, and icechunk for the last variant.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FIELDS = ("rho", "ux", "uy", "uz")
VOXEL = 2.5                 # micrometre; arbitrary
CHUNK_BRICKS = 16           # 128^3 voxels per store chunk


def brick_mask(grid=192, tubes=110, seed=1) -> np.ndarray:
    """Random tubes at brick resolution (a few % of bricks active)."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[:grid, :grid, :grid].astype(np.float32)
    p = np.stack([z, y, x], -1)
    act = np.zeros((grid,) * 3, bool)
    for _ in range(tubes):
        a, b = rng.uniform(0, grid, 3), rng.uniform(0, grid, 3)
        ax = rng.integers(3)
        a[ax], b[ax] = 0, grid - 1
        d = b - a
        t = np.clip(((p - a) @ d) / (d @ d), 0, 1)
        act |= np.linalg.norm(p - (a + t[..., None] * d), axis=-1) <= rng.uniform(1.0, 3.0)
    return act


def domain(cache: Path):
    from zvcfd.domain import BrickDomain

    f = cache / "brick_mask.npy"
    if not f.exists():
        np.save(f, brick_mask())
    return BrickDomain.from_brick_mask(np.load(f))


def field_rows(dom, bricks) -> dict[str, np.ndarray]:
    """Smooth synthetic fields (roughly as compressible as real flow fields)."""
    b = dom.brick
    cell = np.arange(b ** 3)
    lz, ly, lx = cell // (b * b), (cell // b) % b, cell % b
    c = dom.coords[bricks].astype(np.float32)
    gz = c[:, :1] * b + lz
    gy = c[:, 1:2] * b + ly
    gx = c[:, 2:3] * b + lx
    base = np.sin(gx / 37.0) * np.cos(gy / 53.0) + 0.3 * np.sin(gz / 29.0)
    return {"rho": (1 + 1e-3 * base).astype(np.float32),
            "ux": (1e-2 * base).astype(np.float32),
            "uy": (5e-3 * np.cos(gx / 41.0) * base).astype(np.float32),
            "uz": (5e-3 * np.sin(gz / 17.0) * base).astype(np.float32)}


def owners(dom, workers: int, unit_bricks: int) -> np.ndarray:
    return dom.partition(workers, unit_bricks)


# ------------------------------------------------------------------ writers

def w_zv(rank, cfg, bricks, rows):
    from zvcfd.io import fields as zf

    dom = domain(Path(cfg["cache"]))
    level = zf.open_brick_level(cfg["path"], "r+")
    return zf.write_brick_chunks(level, dom, rows, voxel_size=VOXEL, chunk_bricks=CHUNK_BRICKS,
                                 bricks=bricks)


def _dense_blocks(dom, bricks, rows):
    """Yield (slices, {field: block}) for each store chunk among ``bricks``."""
    b = dom.brick
    cb = CHUNK_BRICKS
    cc = dom.coords[bricks] // cb
    keys, inv = np.unique(cc, axis=0, return_inverse=True)
    for k, key in enumerate(keys):
        sel = np.flatnonzero(inv.reshape(-1) == k)
        loc = dom.coords[bricks[sel]] - key * cb
        blocks = {}
        for f, v in rows.items():
            blk = np.zeros((cb, cb, cb, b, b, b), np.float32)
            blk[loc[:, 0], loc[:, 1], loc[:, 2]] = v[sel].reshape(-1, b, b, b)
            blocks[f] = blk.transpose(0, 3, 1, 4, 2, 5).reshape(cb * b, cb * b, cb * b)
        lo = key * cb * b
        yield tuple(slice(int(v), int(v) + cb * b) for v in lo), blocks


def w_zarr(rank, cfg, bricks, blocks, store=None):
    """``blocks``: precomputed [(slices, {field: dense block})]; densifying bricks is
    GPU work in a real run, so it is kept out of the timed region."""
    import zarr

    g = zarr.open_group(store if store is not None else cfg["path"], mode="r+")
    for sl, blk in blocks:
        for f, b in blk.items():
            g[f][sl] = b
    return len(blocks)


def w_icechunk(rank, cfg, bricks, rows, fork):
    w_zarr(rank, cfg, bricks, rows, store=fork.store)
    return fork


def worker(variant, rank, cfg, bricks, barrier, queue, fork=None):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    dom = domain(Path(cfg["cache"]))
    rows = field_rows(dom, bricks)
    payload = sum(v.nbytes for v in rows.values())
    if not variant.startswith("zv"):
        rows = list(_dense_blocks(dom, bricks, rows))
    barrier.wait()
    t0 = time.time()
    if variant.startswith("zv"):
        out = w_zv(rank, cfg, bricks, rows)
    elif variant.startswith("zarr"):
        out = w_zarr(rank, cfg, bricks, rows)
    else:
        out = w_icechunk(rank, cfg, bricks, rows, fork)
    queue.put((rank, t0, time.time(), payload, out if variant.startswith("icechunk") else None))


# ------------------------------------------------------------------ setup / finalize

def setup(variant, cfg, dom):
    path = cfg["path"]
    if variant.startswith("zv"):
        from zvcfd.io import fields as zf

        comp = "zstd" if variant.endswith("zstd") else None
        shard = 2 if "shard2" in variant else None
        zf.create_brick_store(path, dom, voxel_size=VOXEL, chunk_bricks=CHUNK_BRICKS,
                              compressor=comp, shard_shape=shard)
        return None
    import zarr
    from zarr.codecs import ZstdCodec

    comp = [ZstdCodec(level=1)] if variant.endswith("zstd") else None
    side = CHUNK_BRICKS * dom.brick

    def alloc(store):
        g = zarr.open_group(store, mode="w")
        for f in FIELDS:
            g.create_array(f, shape=dom.shape, dtype="float32", chunks=(64, 64, 64),
                           shards=(side, side, side), compressors=comp, fill_value=0.0,
                           config={"write_empty_chunks": False}, dimension_names=("z", "y", "x"))
        return g

    if variant.startswith("zarr"):
        alloc(path)
        return None
    import icechunk

    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(path))
    s = repo.writable_session("main")
    alloc(s.store)
    s.commit("allocate arrays")
    session = repo.writable_session("main")
    return repo, session


def finalize(variant, cfg, state, forks):
    if variant.startswith("zv"):
        from zvcfd.io import fields as zf

        zf.finalize_brick_store(zf.open_brick_level(cfg["path"], "r+"))
    elif variant.startswith("icechunk"):
        repo, session = state
        session.merge(*forks)
        session.commit("snapshot")


# ------------------------------------------------------------------ cache control / reads

def evict(path: str) -> tuple[float, int, int]:
    """Flush dirty pages to the device (one ``sync``; timed), then drop the store's
    files from the page cache so the read phase is cold."""
    t = time.time()
    os.sync()
    t_sync = time.time() - t
    n = size = 0
    for dp, _, files in os.walk(path):
        for f in files:
            fd = os.open(os.path.join(dp, f), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                size += os.fstat(fd).st_size
            finally:
                os.close(fd)
            n += 1
    return t_sync, n, size


def reader(variant, rank, cfg, bricks, barrier, queue):
    dom = domain(Path(cfg["cache"]))
    barrier.wait()
    t0 = time.time()
    nbytes = 0
    if variant.startswith("zv"):
        from zvcfd.io import fields as zf

        level = zf.open_brick_level(cfg["path"])
        cc = np.unique(dom.coords[bricks] // CHUNK_BRICKS, axis=0)
        batch = zf.read_brick_chunks(level, cc, FIELDS)
        nbytes = sum(batch[f"vertex_attributes/{f}"].data.nbytes for f in FIELDS)
    else:
        import zarr

        if variant.startswith("icechunk"):
            import icechunk

            repo = icechunk.Repository.open(icechunk.local_filesystem_storage(cfg["path"]))
            g = zarr.open_group(repo.readonly_session("main").store, mode="r")
        else:
            g = zarr.open_group(cfg["path"], mode="r")
        side = CHUNK_BRICKS * dom.brick
        cc = np.unique(dom.coords[bricks] // CHUNK_BRICKS, axis=0)
        for key in cc:
            sl = tuple(slice(int(k) * side, (int(k) + 1) * side) for k in key)
            for f in FIELDS:
                nbytes += g[f][sl].nbytes
    queue.put((rank, t0, time.time(), nbytes, None))


def run_phase(target, variant, cfg, parts, workers, extra=None):
    ctx = mp.get_context("forkserver")
    barrier, queue = ctx.Barrier(workers), ctx.Queue()
    procs = []
    for r in range(workers):
        args = (variant, r, cfg, np.flatnonzero(parts == r), barrier, queue)
        if extra is not None:
            args = args + (extra[r],)
        procs.append(ctx.Process(target=target, args=args))
    for p in procs:
        p.start()
    res = [queue.get() for _ in procs]
    for p in procs:
        p.join()
        if p.exitcode:
            raise RuntimeError(f"{variant}: a worker failed ({p.exitcode})")
    t0 = min(r[1] for r in res)
    t1 = max(r[2] for r in res)
    return t1 - t0, sum(r[3] for r in res), [r[4] for r in sorted(res)]


def bench(variant, root: Path, workers: int, cache: Path, keep: bool = False) -> dict:
    dom = domain(cache)
    path = str(root / f"{variant}-w{workers}")
    shutil.rmtree(path, ignore_errors=True)
    cfg = {"path": path, "cache": str(cache)}
    unit = CHUNK_BRICKS * (2 if "shard2" in variant else 1)
    parts = owners(dom, workers, unit)
    t = time.time()
    state = setup(variant, cfg, dom)
    t_setup = time.time() - t
    forks = [state[1].fork() for _ in range(workers)] if variant.startswith("icechunk") else None
    t_write, payload, outs = run_phase(worker, variant, cfg, parts, workers, extra=forks)
    t = time.time()
    finalize(variant, cfg, state, outs if forks else None)
    t_final = time.time() - t
    t_sync, n_files, disk = evict(path)
    t_read, read_bytes, _ = run_phase(reader, variant, cfg, parts, workers)
    row = {"variant": variant, "workers": workers, "payload_gb": payload / 1e9,
           "setup_s": t_setup, "write_s": t_write, "finalize_s": t_final, "sync_s": t_sync,
           "write_gbs": payload / 1e9 / (t_write + t_final),
           "write_incl_sync_gbs": payload / 1e9 / (t_write + t_final + t_sync),
           "read_s": t_read, "read_gbs": read_bytes / 1e9 / t_read, "files": n_files,
           "disk_gb": disk / 1e9, "ratio": payload / max(disk, 1)}
    print(f"{variant:<22} w={workers}  payload {row['payload_gb']:.2f} GB  "
          f"write {row['write_gbs']:5.2f} GB/s ({row['write_incl_sync_gbs']:5.2f} to disk)  "
          f"read {row['read_gbs']:5.2f} GB/s  files {n_files:6d}  disk {row['disk_gb']:.2f} GB "
          f"(x{row['ratio']:.2f})  finalize {t_final:.2f} s", flush=True)
    if not keep:
        shutil.rmtree(path, ignore_errors=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variants", default="zv-flat-none,zv-flat-zstd,zv-shard2-zstd,"
                    "zarr-dense-none,zarr-dense-zstd,icechunk-dense-zstd")
    ap.add_argument("--serial-baseline", action="store_true")
    ap.add_argument("--keep", default="", help="comma list of variants whose store to keep")
    ap.add_argument("--json")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "_cache"
    cache.mkdir(exist_ok=True)
    dom = domain(cache)
    print(f"domain {dom.shape}, {dom.n_bricks} bricks ({100 * dom.active_fraction:.1f}% active), "
          f"{dom.stored_cells / 1e6:.0f} M cells, {len(dom.chunks(CHUNK_BRICKS)[0])} store chunks")
    rows = []
    for v in args.variants.split(","):
        rows.append(bench(v, root, args.workers, cache, keep=v in args.keep.split(",")))
        if args.serial_baseline and v in ("zv-flat-zstd", "zarr-dense-zstd"):
            rows.append(bench(v, root, 1, cache))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"host": os.uname().nodename, "cpus": os.cpu_count(), "rows": rows}, fh,
                      indent=1)


if __name__ == "__main__":
    main()
