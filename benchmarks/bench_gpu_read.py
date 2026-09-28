"""Read a brick-store snapshot back onto the GPU: host decode + upload vs device decode.

Uses stores kept by ``bench_io.py --keep``. Each case reads every chunk's
four field arrays in one ``read_cells`` call, cold (page cache dropped
for the store's files) and warm.

    python benchmarks/bench_gpu_read.py --root <bench_io root>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FIELDS = ("rho", "ux", "uy", "uz")


def drop_cache(path):
    for dp, _, files in os.walk(path):
        for f in files:
            fd = os.open(os.path.join(dp, f), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


def main():
    import cupy as cp

    from zvcfd.io import fields as zf

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = []
    for store, modes in (("zv-flat-none-w8", [("host", "host"), ("device", "auto")]),
                         ("zv-shard2-zstd-w8", [("host", "host"), ("device+nvcomp", "device")])):
        path = Path(args.root) / store
        level = zf.open_brick_level(str(path))
        from zarr_vectors import building as zb

        keys = np.array([tuple(int(v) for v in k) for k in zb.list_chunk_keys(level, "vertices")])
        for label, decode in modes:
            for cache in ("cold", "warm"):
                if cache == "cold":
                    drop_cache(path)
                cp.cuda.Device().synchronize()
                t = time.time()
                try:
                    b = zf.read_brick_chunks(level, keys, FIELDS, device="cuda", decode=decode)
                    cp.cuda.Device().synchronize()
                except Exception as exc:   # e.g. nvCOMP decoding the whole read in one batch
                    rows.append({"store": store, "decode": label, "cache": cache,
                                 "error": f"{type(exc).__name__}: {exc}"})
                    print(f"{store:<20} {label:<15} {cache:<5} FAILED: {exc}")
                    cp.get_default_memory_pool().free_all_blocks()
                    continue
                dt = time.time() - t
                nbytes = sum(b[f"vertex_attributes/{f}"].data.nbytes for f in FIELDS)
                rows.append({"store": store, "decode": label, "cache": cache, "chunks": len(keys),
                             "gb": nbytes / 1e9, "seconds": dt, "gbs": nbytes / 1e9 / dt})
                print(f"{store:<20} {label:<15} {cache:<5} {len(keys)} chunks  "
                      f"{nbytes / 1e9:.2f} GB in {dt:.2f} s = {nbytes / 1e9 / dt:.2f} GB/s")
                del b
                cp.get_default_memory_pool().free_all_blocks()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
