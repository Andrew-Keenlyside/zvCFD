"""AmgX smoother and hierarchy variants on captured coupled systems, one process per run.

    ZVCFD_AMGX_LIB=... PYTHONPATH=. python benchmarks/fv/amgx_smoothers.py SYSTEM.npz [...] \\
        [--rtol 0.1] [--only NAME,..]

The systems are those ``speed_simvascular.py --capture`` (the coronary
case) and ``linear_study.py``'s synthetic hard cases save: the scaled
coupled system of one outer iteration and its starting guess. Each
configuration is AmgX's FGMRES with the default hierarchy
(:data:`zvcfd.fv.linear.DEFAULT_AMGX_CONFIG`, DILU base smoother,
pressure-weighted SIZE_4 aggregation) with the overrides below. Every run is
its own process, so an AmgX failure cannot leak into the next one.
Recorded: iterations to ``rtol``, the true residual reduction, set-up and
solve time.

Results: ``benchmarks/results/fv/amgx_smoothers.json`` (merged per system).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOOTH_COARSE = {"coarse_solver": "MULTICOLOR_DILU", "coarsest_sweeps": 4}
GS = {"smoother": "MULTICOLOR_GS", "symmetric_GS": 1, "relaxation_factor": 0.9}

# name -> (description, preconditioner overrides, selector, strength[, base smoother])
CONFIGS = {
    "dilu 1/2 (old default)": (
        "MULTICOLOR_DILU 1 pre / 2 post, relax 0.75, SIZE_4, AmgX strength",
        SMOOTH_COARSE, "SIZE_4", None),
    "dilu-p 0/1": ("MULTICOLOR_DILU 0 pre / 1 post, SIZE_4, pressure-weighted strength",
                   SMOOTH_COARSE | {"presweeps": 0, "postsweeps": 1}, "SIZE_4", 3),
    "gs-sym p SIZE_4": (
        "MULTICOLOR_GS symmetric, 1 pre / 2 post, relax 0.9, SIZE_4, pressure-weighted strength"
        " (the 'gs' preset)", SMOOTH_COARSE | GS, "SIZE_4", 3),
    "gs-sym SIZE_4": ("as gs-sym p SIZE_4 with AmgX's strength", SMOOTH_COARSE | GS,
                      "SIZE_4", None),
    "gs-sym p SIZE_2": ("as gs-sym p SIZE_4, pairwise aggregation", SMOOTH_COARSE | GS,
                        "SIZE_2", 3),
    "gs-sym p 1/1": ("as gs-sym p SIZE_4, 1 pre / 1 post",
                     SMOOTH_COARSE | GS | {"presweeps": 1, "postsweeps": 1}, "SIZE_4", 3),
    "gs-sym p dense-coarse": ("as gs-sym p SIZE_4, dense LU on the coarsest level", GS,
                              "SIZE_4", 3),
    "gs p": ("MULTICOLOR_GS forward only", SMOOTH_COARSE | GS | {"symmetric_GS": 0},
             "SIZE_4", 3),
    "robust": ("multicolour ILU(0), SIZE_2, pressure-weighted strength, dense LU coarsest "
               "(the 'robust' preset)", {}, "SIZE_2", 3, "ilu0"),
    "robust smooth-coarse": ("as robust, smoothed coarsest level", SMOOTH_COARSE, "SIZE_2", 3,
                             "ilu0"),
}


def one(path: str, name: str, rtol: float, precision: str = "double") -> dict:
    """Solve one system with one configuration (in this process)."""
    import cupy as cp
    import numpy as np

    from zvcfd.fv.linear import (
        AMGX,
        DEFAULT_AMGX_CONFIG,
        BlockMatrix,
        amgx_aggregation,
        amgx_smoother,
    )

    _, over, selector, strength, *base = CONFIGS[name]
    z = np.load(path)
    A = BlockMatrix(cp.asarray(z["indptr"]), cp.asarray(z["indices"]), cp.asarray(z["data"]))
    b, x0 = cp.asarray(z["b"]).reshape(-1), cp.asarray(z["x0"]).reshape(-1)
    cfg = json.loads(DEFAULT_AMGX_CONFIG % {
        "rtol": rtol, "maxiter": 300, "reuse": 0,
        "smoother": amgx_smoother(base[0] if base else "dilu"),
        "aggregation": amgx_aggregation(selector, strength)})
    cfg["solver"]["preconditioner"].update(over)
    s = AMGX(config=json.dumps(cfg), precision=precision)
    r0 = float(cp.linalg.norm(b - A.matvec(x0)))
    try:
        cp.cuda.Device().synchronize()
        t = time.time()
        x, info = s.solve(A, b, x0, rtol=rtol)
        cp.cuda.Device().synchronize()
        red = float(cp.linalg.norm(b - A.matvec(x))) / r0
        return {"iterations": info.iterations, "reduction": red,
                "setup_s": info.setup_seconds, "solve_s": info.seconds,
                "total_s": time.time() - t, "converged": red <= 1.01 * rtol}
    except RuntimeError as exc:
        return {"error": str(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="*")
    ap.add_argument("--rtol", type=float, default=0.1)
    ap.add_argument("--only", default=None)
    ap.add_argument("--precision", default="double", choices=["double", "mixed"])
    ap.add_argument("--tag", default="", help="prefix of the result keys (e.g. v2-)")
    ap.add_argument("--one", nargs=2, metavar=("SYSTEM", "NAME"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print("RESULT " + json.dumps(one(a.one[0], a.one[1], a.rtol, a.precision)), flush=True)
        os._exit(0)
    names = a.only.split(",") if a.only else list(CONFIGS)
    dest = ROOT / "benchmarks" / "results" / "fv" / "amgx_smoothers.json"
    res = json.loads(dest.read_text()) if dest.exists() else {}
    res.update({"rtol": a.rtol, "maxiter": 300,
                "configs": {k: v[0] for k, v in CONFIGS.items()},
                "note": "the gs and dilu variants smooth the coarsest level (MULTICOLOR_DILU, "
                        "4 sweeps) unless dense-coarse; system_0/2 are 399k-node SimVascular "
                        "systems captured from a cropped mesh later found defective (isolated "
                        "pieces, singular systems): not representative of the coronary case; "
                        "v2-system_* are captured from the corrected crop (398,105 nodes, "
                        "2,089,669 tets); keys ending -mixed ran in mixed precision"})
    res.setdefault("systems", {})
    for path in a.systems:
        key = a.tag + Path(path).stem + ("" if a.precision == "double" else "-mixed")
        rows = res["systems"].setdefault(key, {})
        for name in names:
            try:
                out = subprocess.run(
                    [sys.executable, __file__, "--one", path, name, "--rtol", str(a.rtol),
                     "--precision", a.precision],
                    capture_output=True, text=True, timeout=900)
                line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
                rows[name] = json.loads(line[-1][7:]) if line else \
                    {"error": "crash: " + out.stderr.strip().splitlines()[-1][:200]
                     if out.stderr.strip() else "crash"}
            except subprocess.TimeoutExpired:
                rows[name] = {"error": "timeout"}
            print(key, name, json.dumps(rows[name]), flush=True)
            dest.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
