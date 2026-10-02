"""Linear solvers on captured systems of a real case (from ``speed_simvascular.py --capture``).

    ZVCFD_AMGX_LIB=... PYTHONPATH=. python benchmarks/fv/linear_study.py DIR [--rtol 0.1] \
        [--only NAME,..]

Each system is the scaled coupled system of one outer iteration, with its
starting guess. Every configuration solves it to ``rtol`` (relative to the
initial residual, as the outer loop asks) from that guess; recorded: set-up
and solve time, iterations, the true residual reduction, and whether it
converged. The configurations are AmgX (presets and variants), AmgX in
mixed precision, the SIMPLE block preconditioner and zvCFD's ACM.

Results: ``benchmarks/results/fv/linear_study.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def configs():
    from zvcfd.fv.linear import ACM, AMGX, SIMPLE

    def amgx(**kw):
        return lambda rtol: AMGX(rtol=rtol, maxiter=400, **kw)

    return {
        "amgx-dilu": amgx(preset="dilu"),
        "amgx-dilu-p": amgx(preset="dilu-p"),
        "amgx-robust": amgx(preset="robust"),
        "amgx-dilu-mixed": amgx(preset="dilu", precision="mixed"),
        "amgx-robust-mixed": amgx(preset="robust", precision="mixed"),
        "amgx-ilu0-size4": amgx(smoother="ilu0", selector="SIZE_4", strength=3),
        "amgx-ilu0-size8": amgx(smoother="ilu0", selector="SIZE_8", strength=3),
        "amgx-dilu-size8-p": amgx(smoother="dilu", selector="SIZE_8", strength=3),
        "amgx-dilu-size2": amgx(smoother="dilu", selector="SIZE_2", strength=None),
        "simple": lambda rtol: SIMPLE(rtol=rtol, maxiter=400),
        "acm": lambda rtol: ACM(rtol=rtol, maxiter=400),
    }


def main():
    import cupy as cp

    from zvcfd.fv.linear import BlockMatrix

    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rtol", type=float, default=0.1)
    ap.add_argument("--only", default=None)
    ap.add_argument("--systems", default=None)
    ap.add_argument("--pattern", default="system_*.npz", help="which files of DIR to solve")
    a = ap.parse_args()
    files = sorted(Path(a.dir).glob(a.pattern))
    if a.systems:
        files = [f for f in files if f.stem.split("_")[1] in a.systems.split(",")]
    cfg = configs()
    if a.only:
        cfg = {k: v for k, v in cfg.items() if k in a.only.split(",")}
    dest = ROOT / "benchmarks" / "results" / "fv" / "linear_study.json"
    results = json.loads(dest.read_text()) if dest.exists() else {}
    for f in files:
        z = np.load(f)
        A = BlockMatrix(cp.asarray(z["indptr"]), cp.asarray(z["indices"]),
                        cp.asarray(z["data"]))
        b, x0 = cp.asarray(z["b"]).reshape(-1), cp.asarray(z["x0"]).reshape(-1)
        r0 = float(cp.linalg.norm(b - A.matvec(x0)))
        for name, make in cfg.items():
            s = None
            try:
                s = make(a.rtol)
                cp.cuda.Device().synchronize()
                t = time.time()
                x, info = s.solve(A, b, x0, rtol=a.rtol)
                cp.cuda.Device().synchronize()
                total = time.time() - t
                red = float(cp.linalg.norm(b - A.matvec(x))) / r0
                row = {"iterations": info.iterations, "reduction": red,
                       "converged": bool(red <= 1.01 * a.rtol), "setup_s": info.setup_seconds,
                       "solve_s": info.seconds, "total_s": total}
            except Exception as exc:
                row = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            finally:
                if s is not None and hasattr(s, "close"):
                    s.close()
                s = None
            cp.get_default_memory_pool().free_all_blocks()
            results.setdefault(f"{f.stem}-rtol{a.rtol:g}", {})[name] = row
            print(f.stem, name, json.dumps(row), flush=True)
            dest.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
