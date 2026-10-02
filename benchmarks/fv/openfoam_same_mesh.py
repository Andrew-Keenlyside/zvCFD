"""Cross-code: the finite-volume reference solver and OpenFOAM on the *same* unstructured mesh.

    FOAM_SIF=/path/to/esi2506.sif PYTHONPATH=. python benchmarks/fv/openfoam_same_mesh.py \
        [--levels 3,6,12,24] [--only foam|zvcfd] [--procs 16]

A generated pipe (wall wedges around a tetrahedral core, like the coronary
mesh) is written as a Fluent ``.msh`` by :func:`zvcfd.mesh.write_fluent_mesh`,
imported into OpenFOAM with ``fluent3DMeshToFoam``, and checked with
``checkMesh``, an independent test of the writer. Both codes then solve
the same pressure-driven laminar flow: fixed pressure at both ends, zero
normal velocity gradient there, and no-slip walls. There are no velocity
profiles to impose differently. OpenFOAM is cell-centred (``simpleFoam``,
second-order ``linearUpwind``); zvCFD is vertex-centred. Compared: the flow
rate (the developed Poiseuille value is only a guide: at Re = 50 the flow
is still developing over part of the pipe).

Levels up to 6 use the CPU reference solver; finer ones the GPU solver
(``linear: auto``), which reaches the same discrete solution. ``--only``
runs one code and merges into the results file, so the CPU-bound
OpenFOAM runs and the GPU runs can be done at different times.

Results: ``benchmarks/results/fv/openfoam_same_mesh.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "openfoam"))
import foamcase as fc  # noqa: E402

from zvcfd.fv.reference import Fluid, ReferenceSolver  # noqa: E402
from zvcfd.mesh import write_fluent_mesh  # noqa: E402
from zvcfd.mesh.generate import tube  # noqa: E402

R, L = 0.5, 4.0
MU, RHO = 0.02, 1.0
DP = 2.56          # U_mean (developed) = DP R^2 / (8 mu L) = 1, Re = rho U_mean 2R / mu = 50


def exact_q():
    return np.pi * R ** 4 * DP / (8 * MU * L)


def run_zvcfd(m, gpu: bool = False):
    zid = {z.name: k for k, z in m.zones.items()}
    bcs = {zid["inlet"]: {"kind": "pressure", "value": DP},
           zid["outlet"]: {"kind": "pressure", "value": 0.0}, zid["wall"]: {"kind": "wall"}}
    t = time.time()
    if gpu:
        from zvcfd.fv.solver import GPUSolver

        s = GPUSolver(m, Fluid(RHO, MU), bcs, advection="high-resolution", linear="auto")
        rep = s.solve(max_iterations=600, tol=1e-9)
        U = s.fields()["U"]
    else:
        s = ReferenceSolver(m, Fluid(RHO, MU), bcs, advection="high-resolution")
        rep = s.solve(max_iterations=100, tol=1e-11)
        U = s.U
    q = s.zone_flows()
    mid = np.isclose(m.nodes[:, 2], L / 2)
    return {"q": -q[zid["inlet"]] / RHO, "q_imbalance": (q[zid["inlet"]] + q[zid["outlet"]])
            / abs(q[zid["inlet"]]), "iterations": rep.iterations, "converged": rep.converged,
            "solver": "gpu" if gpu else "reference", "seconds": time.time() - t,
            "u_mid": U[mid, 2].tolist(), "r_mid": np.hypot(m.nodes[mid, 0],
                                                           m.nodes[mid, 1]).tolist()}


def run_openfoam(m, case: Path, procs: int = 1):
    msh = case.parent / f"{case.name}.msh"
    if case.exists():                 # controlDict starts from the latest time: start clean
        import shutil

        shutil.rmtree(case)
    case.mkdir(parents=True, exist_ok=True)
    write_fluent_mesh(m, msh)
    t = {"import": fc.import_fluent(case, str(msh), scale=1.0)}
    fc.foam(case, "checkMesh", "log.checkMesh", check=False)
    check = (case / "log.checkMesh").read_text()
    names = fc.patch_names(case)
    fc.write_system(case, n_procs=procs, end=8000, flow_patches=["inlet", "outlet"])
    fc.write_physics(case, MU / RHO)
    kin = DP / RHO                                       # OpenFOAM's p is kinematic
    fc.write_fields(case, {"inlet": fc.pressure(kin), "outlet": fc.pressure(0.0),
                           "wall": fc.WALL})
    if procs > 1:
        t["solve"] = fc.run_parallel(case, procs)
        fc.foam(case, "reconstructPar -latestTime", "log.reconstructPar", check=False)
    else:
        t["solve"] = fc.foam(case, "simpleFoam", "log.simpleFoam")
    fc.foam(case, "postProcess -func 'singleGraph' -latestTime", "log.graph", check=False)
    flows = fc.read_flows(case, 2)
    log = fc.read_log(case)
    return {"q": float(-flows[-1, 0]), "q_imbalance": float((flows[-1, 0] + flows[-1, 1])
                                                             / abs(flows[-1, 0])),
            "iterations": log["iterations"], "converged": log["converged"],
            "patches": names, "checkMesh_ok": "Mesh OK." in check,
            "checkMesh_failed": "Failed" in check and "Mesh OK." not in check, "seconds": t}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default=str(ROOT / "benchmarks" / "results" / "fv"))
    ap.add_argument("--levels", default="3,6")
    ap.add_argument("--only", choices=["foam", "zvcfd"])
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    dest = Path(a.out_dir) / "openfoam_same_mesh.json"
    work = Path("/tmp") / "zvcfd_foam_same_mesh"
    rows = {r["nc"]: r for r in json.loads(dest.read_text())} if dest.exists() else {}
    for nc in (int(v) for v in a.levels.split(",")):
        m = tube(R, L, n_core=nc, n_ring=nc, n_axial=8 * nc, kind="mixed",
                 layers=max(1, nc // 2), growth=0.85)
        poly = (4 * nc / (2 * np.pi)) * np.sin(2 * np.pi / (4 * nc))   # wall polygon / circle
        row = rows.get(nc, {}) | {"nc": nc, "nodes": m.n_nodes, "cells": m.n_elements,
                                  "elements": m.counts(), "exact_q_circle": exact_q(),
                                  "wall_polygon_area_ratio": poly}
        if a.only != "foam":
            z = run_zvcfd(m, gpu=nc > 6)
            row["zvcfd"] = {k: v for k, v in z.items() if k not in ("u_mid", "r_mid")}
        if a.only != "zvcfd":
            procs = a.procs if nc > 6 else 1
            row["openfoam"] = run_openfoam(m, work / f"pipe_nc{nc}", procs) | {"procs": procs}
        if "zvcfd" in row and "openfoam" in row:
            row["q_difference"] = (row["zvcfd"]["q"] - row["openfoam"]["q"]) / row["openfoam"]["q"]
        rows[nc] = row
        print(json.dumps(row, default=float), flush=True)
        dest.write_text(json.dumps([rows[k] for k in sorted(rows)], indent=1, default=float))


if __name__ == "__main__":
    main()
