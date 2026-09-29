"""zvCFD vs OpenFOAM on identical voxel geometries: steady pressure-driven Stokes flow.

Both solvers see the same staircase geometry: zvCFD's half-way bounce-back
walls lie on the voxel faces, and the OpenFOAM mesh is the fluid voxels
themselves (blockMesh + subsetMesh), so walls are the same faces. Flow is
driven by a fixed pressure difference between the x faces; every other
exposed face is a no-slip wall. Only the fluid component touching both x
faces is kept, for both solvers, and the porous and vessel samples get
four open voxel layers at each end so neither solver's pressure boundary
cuts through the sample.

Physical setting: voxel 10 um, water (nu = 1e-6 m^2/s), kinematic pressure
gradient 0.0125 m/s^2, so the flow is Stokes (Re << 1). zvCFD: TRT (magic
3/16: walls exact and steady solutions independent of tau, so tau is a
convergence knob, ``--tau``), fp32, one RTX A2000. OpenFOAM v2506 (official
ESI image, run through Apptainer with ``FOAM_SIF``): laminar
simpleFoam (SIMPLEC, GAMG, linear-upwind), 16 MPI ranks on a 16-core
Threadripper PRO 5955WX. Timing covers the solve only (no meshing, no I/O).

Metric: wall time until the outlet flux is within 1 % and 0.1 % of that
solver's converged value, the converged flux itself, and for pipes the
analytic Poiseuille flux.

    FOAM_SIF=esi2506.sif python benchmarks/openfoam/voxel_compare.py --cases pipe6,pipe12 \
        --solvers zvcfd,openfoam
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

DX = 10e-6          # m
NU = 1e-6           # m^2/s
G_KIN = 0.0125      # kinematic pressure gradient (m/s^2): Stokes; lattice density drop < 0.5 %
BUFFER = 4          # open voxel layers added at each end of the porous and vessel samples

RESULTS = ROOT / "benchmarks" / "results" / "openfoam"


def connected_through(fluid: np.ndarray) -> np.ndarray:
    lab, _ = ndimage.label(fluid)
    keep = np.intersect1d(np.unique(lab[:, :, 0]), np.unique(lab[:, :, -1]))
    return np.isin(lab, keep[keep > 0])


def case_geometry(name: str) -> tuple[np.ndarray, dict]:
    """Fluid mask (z, y, x) and metadata; walls enclose everything but the x faces."""
    if name.startswith("pipe"):
        r = int(name[4:])
        side = int(np.ceil((2 * r + 2) / 8) * 8)
        length = int(np.ceil(16 * r / 8) * 8)
        z, y = np.mgrid[:side, :side] + 0.5
        c = side / 2
        disk = (z - c) ** 2 + (y - c) ** 2 <= r * r
        fluid = np.repeat(disk[:, :, None], length, axis=2)
        return fluid, {"radius_vox": r, "area_vox": int(disk.sum())}
    if name.startswith("duct"):
        h = int(name[4:])                      # square duct, side h voxels, 1-voxel walls
        side = int(np.ceil((h + 2) / 8) * 8)
        length = int(np.ceil(8 * h / 8) * 8)
        f = np.zeros((side, side, length), bool)
        f[1:h + 1, 1:h + 1, :] = True
        return f, {"duct_side_vox": h}
    if name == "vessels":
        from zvcfd.phantoms import vessel_network

        f = vessel_network((128, 128, 512))
    elif name == "porous":
        from zvcfd.phantoms import porous_spheres

        f = porous_spheres(128, porosity_target=0.45, radius=6) == 0
    else:
        raise ValueError(name)
    f = f.copy()
    f[0], f[-1], f[:, 0], f[:, -1] = False, False, False, False     # wall the y, z faces
    f = connected_through(f)
    # open buffer layers at both ends, as in permeability workflows: the pressure
    # boundaries then see nearly uniform flow instead of cutting through the sample
    pad = np.zeros(f.shape[:2] + (BUFFER,), bool)
    pad[1:-1, 1:-1] = True
    return np.concatenate([pad, f, pad], axis=2), {"buffer_vox": BUFFER}


def kinematic_dp(nx: int, *, faces_apart: bool) -> float:
    """Pressure drop (m^2/s^2) for the common gradient over the solver's pressure-plane distance.

    LBM pressure nodes are voxel centres x = 0 and nx-1; OpenFOAM's are the
    box faces x = 0 and nx. Same gradient on both.
    """
    return G_KIN * DX * (nx if faces_apart else nx - 1)


def duct_flux(h_vox: int, terms: int = 200) -> float:
    """Exact Stokes flux through a square duct of side h (series solution)."""
    a = h_vox * DX / 2
    n = np.arange(1, 2 * terms, 2)
    s = (np.tanh(n * np.pi / 2) / n ** 5).sum()
    return 4 * a ** 4 * G_KIN / (3 * NU) * (1 - 192 / np.pi ** 5 * s)


def poiseuille(meta: dict, nx: int) -> float:
    """Analytic flux (m^3/s) of a circular pipe of the voxel disk's area-equivalent radius."""
    r = np.sqrt(meta["area_vox"] / np.pi) * DX
    return np.pi * r ** 4 * G_KIN / (8 * NU)


# ---------------------------------------------------------------- zvCFD

def run_zvcfd(name: str, fluid: np.ndarray, *, tau: float = 0.8, check: int = 200,
              max_steps: int = 400_000, tol: float = 1e-7) -> dict:
    import cupy as cp

    from zvcfd.boundary import Patch, face_patches
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM

    flags = np.where(fluid, 0, 1).astype(np.uint8)
    dom = BrickDomain.from_flags(flags)
    b = face_patches(dom, {"xmin": Patch("in", "pressure"), "xmax": Patch("out", "pressure")})
    sim = SparseLBM(dom, tau=tau, collision="trt", boundary=b)
    nx = fluid.shape[2]
    dt = (tau - 0.5) / 3 * DX ** 2 / NU
    drho = 3 * kinematic_dp(nx, faces_apart=False) * (dt / DX) ** 2
    sim.set_patch(0, rho=1 + drho / 2)
    sim.set_patch(1, rho=1 - drho / 2)
    sim.step(10)
    cp.cuda.Device().synchronize()
    hist, prev = [], None
    t0 = time.time()
    while sim.steps < max_steps:
        sim.step(check)
        q = -sim.patch_flux()[1] * DX ** 3 / dt
        hist.append((sim.steps, time.time() - t0, q))
        if prev is not None and abs(q - prev) <= tol * abs(q):
            break
        prev = q
    return _summary("zvcfd", hist, cells=dom.fluid_cells,
                    extra={"tau": tau, "steps": sim.steps,
                           "mlups": dom.fluid_cells * sim.steps /
                           hist[-1][1] / 1e6, "gpu": cp.cuda.runtime.getDeviceProperties(0)
                           ["name"].decode()})


# ---------------------------------------------------------------- OpenFOAM

def run_openfoam(name: str, fluid: np.ndarray, *, work: Path, procs: int = 16,
                 end: int = 5000, solver: str = "simpleFoam") -> dict:
    import foamcase as fc

    case = work / (name + ("" if solver == "simpleFoam" else "_" + solver))
    nx = fluid.shape[2]
    dp = kinematic_dp(nx, faces_apart=True)
    cells = fc.write_voxel_case(case, fluid, DX, NU, n_procs=procs, end=end,
                                faces={"xmin": fc.pressure(dp), "xmax": fc.pressure(0.0)})
    mesh_t = fc.mesh_voxel_case(case)
    if solver == "icoFoam":
        # steady state by time marching: dt of a few cell diffusion times
        scale = 2.0 * (fluid.sum() / max(fluid.shape[2], 1)) ** 0.5     # ~ cross-section width
        fc.make_transient_ico(case, dt=0.05 * (scale * DX) ** 2 / NU, steps=400, n_procs=procs)
    t = fc.run_parallel(case, procs, solver=solver)
    log = fc.read_log(case)
    flows = fc.read_flows(case, 2)
    exe = np.asarray(log["exec_s"])
    n = min(len(exe), len(flows))
    hist = [(i + 1, float(exe[i]), float(flows[i, 1])) for i in range(n)]
    return _summary("openfoam", hist, cells=cells,
                    extra={"iterations": log["iterations"], "converged": log["converged"],
                           "procs": procs, **mesh_t, **t})


# ---------------------------------------------------------------- common

def _summary(solver, hist, *, cells, extra) -> dict:
    steps = np.array([h[0] for h in hist])
    wall = np.array([h[1] for h in hist])
    q = np.array([h[2] for h in hist])
    final = q[-1]
    out = {"solver": solver, "cells": int(cells), "flux_final": float(final),
           "wall_total_s": float(wall[-1]), **extra}
    for tol in (1e-2, 1e-3):
        # first time after which the flux stays within tol of the final value
        ok = np.abs(q - final) <= tol * abs(final)
        bad = np.flatnonzero(~ok)
        i = 0 if len(bad) == 0 else bad[-1] + 1
        i = min(i, len(q) - 1)
        out[f"iters_to_{tol:g}"] = int(steps[i])
        out[f"wall_to_{tol:g}"] = float(wall[i])
    out["history"] = [[int(s), float(w), float(v)] for s, w, v in hist[::max(1, len(hist) // 400)]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="pipe6,pipe12,pipe24,vessels,porous")
    ap.add_argument("--solvers", default="zvcfd,openfoam")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--work", default="/tmp/zvcfd-foam")
    ap.add_argument("--tau", type=float, default=0.8,
                    help="zvCFD relaxation time (TRT); see tau_scan.py")
    ap.add_argument("--tag", default="", help="suffix for the zvcfd result key, e.g. _tau2")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    for name in args.cases.split(","):
        fluid, meta = case_geometry(name)
        rec_path = RESULTS / f"voxel_{name}.json"
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
        rec.update({"case": name, "shape": list(fluid.shape), "fluid_voxels": int(fluid.sum()),
                    "dx": DX, "nu": NU, **meta})
        if meta.get("area_vox"):
            rec["flux_poiseuille"] = poiseuille(meta, fluid.shape[2])
        if meta.get("duct_side_vox"):
            rec["flux_poiseuille"] = duct_flux(meta["duct_side_vox"])      # exact here
        for solver in args.solvers.split(","):
            if solver == "zvcfd":
                r = run_zvcfd(name, fluid, tau=args.tau)
            else:
                r = run_openfoam(name, fluid, work=work, procs=args.procs,
                                 solver="icoFoam" if solver == "openfoam_ico" else "simpleFoam")
            key = solver + (args.tag if solver == "zvcfd" else "")
            rec[key] = r
            print(f"{name:<8} {key:<14} cells {r['cells']:>9,}  flux {r['flux_final']:.5e}  "
                  f"to 1%: {r['wall_to_0.01']:8.2f} s  to 0.1%: {r['wall_to_0.001']:8.2f} s  "
                  f"total {r['wall_total_s']:8.2f} s", flush=True)
        if "zvcfd" in rec and "openfoam_ico" in rec:
            print(f"{name:<8} zvCFD / OpenFOAM(icoFoam) flux: "
                  f"{rec['zvcfd']['flux_final'] / rec['openfoam_ico']['flux_final']:.4f}")
        if "zvcfd" in rec and "openfoam" in rec:
            a, b = rec["zvcfd"]["flux_final"], rec["openfoam"]["flux_final"]
            rec["flux_difference"] = a / b - 1
            print(f"{name:<8} zvCFD / OpenFOAM flux: {a / b:.4f}"
                  + (f"; vs Poiseuille: zvCFD {a / rec['flux_poiseuille']:.4f}, OpenFOAM "
                     f"{b / rec['flux_poiseuille']:.4f}" if "flux_poiseuille" in rec else ""))
        rec_path.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
