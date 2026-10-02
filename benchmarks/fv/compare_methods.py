"""A small local benchmark: one steady pressure-driven pipe, solved by several methods.

    FOAM_SIF=esi2506.sif ZVCFD_AMGX_LIB=... PYTHONPATH=. python benchmarks/fv/compare_methods.py \\
        [--levels 6,12] [--methods gpu-fv,gpu-fv-before,openfoam,cpu-fv,lbm] [--procs 16] \\
        [--lbm-radii 8,16,32] [--clean-tries 3] [--out FILE]

The case is that of ``openfoam_same_mesh.py``: a pipe of radius 0.5 and
length 4 (wall wedge layers around a tetrahedral core), fixed pressure at
both ends (Δp = 2.56), no-slip walls, Re = 50 on the developed mean
velocity. The methods:

- ``gpu-fv``: zvCFD's GPU finite-volume solver as it is now, with the
  defaults of ``zvcfd run`` (fused CUDA kernels, ``linear: auto``: AmgX
  ``gs``, which probes ``robust`` on the third system and keeps the faster);
- ``gpu-fv-before``: the same solver before the speed review (the classic
  kernels, the staged boundary path, AmgX ``robust``, which the ``auto``
  ladder then tried first);
- ``openfoam``: OpenFOAM v2506 ``simpleFoam`` (``linearUpwind``, GAMG) on
  the same mesh, ``--procs`` MPI ranks (1 on levels up to 6);
- ``cpu-fv``: zvCFD's CPU reference solver (SciPy; levels up to 3 only);
- ``lbm``: zvCFD's lattice Boltzmann solver on a voxelised pipe (radius
  ``R`` voxels, pressure patches at the ends, lattice velocity 0.04):
  another discretisation of the same problem, with staircase walls.

Recorded for each: set-up time, time to converge by the method's own
criterion, and the time after which the flow rate stays within 0.1 % of
its converged value (``t_q_0.1pct_s``; ``t_q_1pct_s`` for 1 %, the metric
of docs/benchmarks/openfoam.md): the time an engineer waits for the number
they want. The flow rate is also given against the developed Poiseuille
value, as a check that every method solved the same problem.

The GPU of the reference workstation is shared. Each GPU run records a
probe of the GPU before and after it (copy bandwidth, launch-plus-sync
latency) and the fraction of one-second ``nvidia-smi pmon`` samples in which
another process used the SMs; ``--clean-tries`` repeats a run until one is
uncontended.

Results: ``benchmarks/results/fv/compare_methods.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "fv"))
sys.path.insert(0, str(ROOT / "benchmarks" / "openfoam"))
import openfoam_same_mesh as osm  # noqa: E402

R, L, MU, RHO, DP = osm.R, osm.L, osm.MU, osm.RHO, osm.DP
Q_POISEUILLE = osm.exact_q()


def gpu_probe() -> dict:
    """How much of the GPU is available now: device-to-device copy bandwidth (the idle A2000:
    253 GB/s) and the median time of one tiny kernel launch plus synchronisation (tens of µs
    idle; milliseconds when another process's kernels time-slice the GPU). The GPU is shared,
    so every GPU timing is recorded with both."""
    import cupy as cp

    a = cp.ones(1 << 25)
    b = cp.empty_like(a)
    for _ in range(3):
        b[...] = a
    cp.cuda.Device().synchronize()
    t = time.perf_counter()
    for _ in range(20):
        b[...] = a
    cp.cuda.Device().synchronize()
    gbs = 2 * a.nbytes * 20 / (time.perf_counter() - t) / 1e9
    del a, b
    cp.get_default_memory_pool().free_all_blocks()
    x = cp.zeros(1024)
    ts = []
    for _ in range(200):
        t = time.perf_counter()
        x += 1
        cp.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t)
    return {"copy_gbs": gbs, "sync_us": 1e6 * float(np.median(ts[20:]))}


class Contention:
    """Samples ``nvidia-smi pmon`` once a second while a GPU run lasts: ``fraction`` is the
    share of samples in which another compute process used the SMs (the probes at the two
    ends miss a job that starts or stops in between)."""

    def __enter__(self):
        import subprocess

        self.proc = subprocess.Popen(["nvidia-smi", "pmon", "-d", "1", "-s", "u", "-o", "T"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        out, _ = self.proc.communicate(timeout=10)
        me = os.getpid()
        samples, busy = set(), set()
        for line in out.splitlines():
            f = line.split()
            if line.startswith("#") or len(f) < 5 or not f[2].isdigit():
                continue
            samples.add(f[0])
            if int(f[2]) != me and f[3] == "C" and f[4] not in ("-", "0"):
                busy.add(f[0])
        self.fraction = len(busy) / max(len(samples), 1)
        return False


def cleanest(run, tries: int, wait_s: float = 60.0) -> dict:
    """Run a GPU measurement up to ``tries`` times, until one sees another process on the
    SMs in under 5 % of its samples; the cleanest run is the result, every attempt is
    listed."""
    runs = []
    for k in range(tries):
        runs.append(run())
        if runs[-1]["gpu_contended_fraction"] < 0.05:
            break
        if k + 1 < tries:
            time.sleep(wait_s)
    best = dict(min(runs, key=lambda r: r["gpu_contended_fraction"]))
    best["attempts"] = [{k: r.get(k) for k in ("gpu_contended_fraction", "solve_s",
                                              "t_q_0.1pct_s", "per_iteration_s")}
                        for r in runs]
    return best


def settle_time(t, q, rel=1e-3) -> float | None:
    """The first time after which ``q`` stays within ``rel`` of its last value."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    if not len(q):
        return None
    off = np.abs(q - q[-1]) > rel * abs(q[-1])
    if not off.any():
        return float(t[0])
    last = int(np.flatnonzero(off)[-1])
    return float(t[min(last + 1, len(t) - 1)])


def mesh(nc):
    from zvcfd.mesh.generate import tube

    return tube(R, L, n_core=nc, n_ring=nc, n_axial=8 * nc, kind="mixed",
                layers=max(1, nc // 2), growth=0.85)


def bcs_of(m):
    zid = {z.name: k for k, z in m.zones.items()}
    return zid, {zid["inlet"]: {"kind": "pressure", "value": DP},
                 zid["outlet"]: {"kind": "pressure", "value": 0.0}, zid["wall"]: {"kind": "wall"}}


def run_gpu_fv(m, before=False):
    import cupy as cp

    from zvcfd.fv.reference import Fluid
    from zvcfd.fv.solver import GPUSolver

    zid, bcs = bcs_of(m)
    probe = gpu_probe()
    watch = Contention().__enter__()
    cp.cuda.Device().synchronize()
    t0 = time.time()
    kw = {"kernels": "classic", "linear": "amgx", "linear_options": {"preset": "robust"}} \
        if before else {"linear": "auto"}
    s = GPUSolver(m, Fluid(RHO, MU), bcs, advection="high-resolution", **kw)
    cp.cuda.Device().synchronize()
    setup = time.time() - t0
    rep = s.solve(max_iterations=600, tol=1e-9)
    cp.cuda.Device().synchronize()
    solve = time.time() - t0 - setup
    h = rep.history
    t = np.cumsum([r["seconds"] for r in h])
    q = [-r["flows"][zid["inlet"]] / RHO for r in h]
    out = {"q": q[-1], "q_ratio_poiseuille": q[-1] / Q_POISEUILLE,
           "iterations": rep.iterations, "converged": bool(rep.converged), "setup_s": setup,
           "solve_s": solve, "per_iteration_s": solve / max(rep.iterations, 1),
           "t_q_0.1pct_s": settle_time(t, q), "t_q_1pct_s": settle_time(t, q, 1e-2),
           "linear_iterations_mean": float(np.mean([r["linear_iterations"] for r in h])),
           "device_pool_gb": cp.get_default_memory_pool().total_bytes() / 1e9,
           "linear_active": getattr(s.linear, "active", s.linear.name),
           "linear_switches": getattr(s.linear, "switches", []),
           "linear_probe": getattr(s.linear, "probe", None), "gpu_before": probe}
    watch.__exit__(None, None, None)
    out["gpu_after"], out["gpu_contended_fraction"] = gpu_probe(), watch.fraction
    if hasattr(s.linear, "close"):
        s.linear.close()
    return out


def run_cpu_fv(m):
    from zvcfd.fv.reference import Fluid, ReferenceSolver

    zid, bcs = bcs_of(m)
    t0 = time.time()
    s = ReferenceSolver(m, Fluid(RHO, MU), bcs, advection="high-resolution")
    setup = time.time() - t0
    rep = s.solve(max_iterations=100, tol=1e-11)
    solve = time.time() - t0 - setup
    q = -s.zone_flows()[zid["inlet"]] / RHO
    return {"q": q, "q_ratio_poiseuille": q / Q_POISEUILLE, "iterations": rep.iterations,
            "converged": bool(rep.converged), "setup_s": setup, "solve_s": solve,
            "per_iteration_s": solve / max(rep.iterations, 1)}


def run_openfoam(m, nc, procs):
    import foamcase as fc

    case = Path("/tmp") / "zvcfd_compare_methods" / f"pipe_nc{nc}_np{procs}"
    if case.exists():
        import shutil

        shutil.rmtree(case)
    t0 = time.time()
    r = osm.run_openfoam(m, case, procs)
    wall = time.time() - t0
    log = fc.read_log(case)
    flows = fc.read_flows(case, 2)
    q = -flows[:, 0]
    exe = np.asarray(log["exec_s"][:len(q)])
    solve = r["seconds"]["solve"]
    solve = solve["solve_s"] if isinstance(solve, dict) else solve
    return {"q": r["q"], "q_ratio_poiseuille": r["q"] / Q_POISEUILLE,
            "iterations": r["iterations"], "converged": bool(r["converged"]), "procs": procs,
            "setup_s": wall - solve, "solve_s": solve,
            "per_iteration_s": solve / max(r["iterations"], 1),
            "t_q_0.1pct_s": settle_time(exe, q[:len(exe)]),
            "t_q_1pct_s": settle_time(exe, q[:len(exe)], 1e-2), "checkMesh_ok": r["checkMesh_ok"]}


def run_lbm(Rv, u_lat=0.04, check=None, tau_min=0.55):
    """The same pipe on the lattice: radius ``Rv`` voxels, length 8 Rv (L/R = 8).

    Re = 50 fixes ``ν = u D / 50``; the lattice velocity ``u_lat`` is raised
    where that would put τ below ``tau_min`` (TRT with pressure patches is
    unstable near 0.5): 0.052 at 8 voxels, 0.04 from 10 on.
    """
    import cupy as cp

    from zvcfd.boundary import Patch, face_patches
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM

    n = int(np.ceil((2 * Rv + 2) / 8) * 8)
    nx = 8 * Rv
    c = n / 2
    j = np.arange(n) + 0.5
    Zc, Yc = np.meshgrid(j - c, j - c, indexing="ij")
    disk = np.hypot(Zc, Yc) <= Rv
    flags = np.ones((n, n, nx), np.uint8)
    flags[disk] = 0
    dx = R / Rv
    re = RHO * 2 * R * 1.0 / MU                              # 50, on U = 1
    nu = max(u_lat * 2 * Rv / re, (tau_min - 0.5) / 3)
    u_lat = nu * re / (2 * Rv)
    tau = 3 * nu + 0.5
    # patch centres are nx - 1 voxels apart: the same pressure gradient as Δp over L
    dp = DP * u_lat ** 2 * (nx - 1) / nx
    probe = gpu_probe()
    watch = Contention().__enter__()
    t0 = time.time()
    dom = BrickDomain.from_flags(flags, periodic=(False, False, False))
    b = face_patches(dom, {"xmin": Patch("in", "pressure"), "xmax": Patch("out", "pressure")})
    sim = SparseLBM(dom, tau=tau, collision="trt", boundary=b)
    sim.set_patch(0, rho=1 + 1.5 * dp)
    sim.set_patch(1, rho=1 - 1.5 * dp)
    cp.cuda.Device().synchronize()
    setup = time.time() - t0
    check = check or max(100, int(Rv * Rv / nu / 50))
    t0 = time.time()
    ts, qs = [], []
    prev = None
    min_steps = int(3 * (Rv * Rv / nu + nx / u_lat))    # viscous plus convective times
    while sim.steps < 5_000_000:
        sim.step(check)
        qs.append(float(sim.patch_flux()[0]) / RHO * dx * dx / u_lat)
        ts.append(time.time() - t0)
        if not np.isfinite(qs[-1]):
            break
        if prev is not None and sim.steps >= min_steps and abs(qs[-1] - prev) <= 1e-6 * abs(prev):
            break
        prev = qs[-1]
    watch.__exit__(None, None, None)
    cells = int(dom.n_fluid) if hasattr(dom, "n_fluid") else int((flags == 0).sum())
    return {"R_voxels": Rv, "tau": tau, "u_lattice": u_lat, "fluid_cells": cells, "q": qs[-1],
            "q_ratio_poiseuille": qs[-1] / Q_POISEUILLE, "steps": int(sim.steps),
            "setup_s": setup, "solve_s": ts[-1], "t_q_0.1pct_s": settle_time(ts, qs),
            "t_q_1pct_s": settle_time(ts, qs, 1e-2),
            "mlups": cells * sim.steps / ts[-1] / 1e6, "gpu_before": probe,
            "gpu_after": gpu_probe(), "gpu_contended_fraction": watch.fraction}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="6,12")
    ap.add_argument("--methods", default="gpu-fv,gpu-fv-before,openfoam,cpu-fv,lbm")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--lbm-radii", default="8,16,32")
    ap.add_argument("--clean-tries", type=int, default=1,
                    help="repeat each GPU run until one is uncontended (at most this often)")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "fv"
                                         / "compare_methods.json"))
    a = ap.parse_args()
    methods = a.methods.split(",")
    dest = Path(a.out)
    res = {"fv_levels": {}, "lbm": {}}
    load = os.getloadavg()[0]

    def save():
        """Merge this run's rows into the file as it is now (runs of other methods may be
        writing it too)."""
        cur = json.loads(dest.read_text()) if dest.exists() else {"fv_levels": {}, "lbm": {}}
        cur["machine"] = {"gpu": "RTX A2000 12GB (idle copy bandwidth 253 GB/s)",
                          "cpu": "Threadripper PRO 5955WX (16 cores)"}
        cur["q_poiseuille"] = Q_POISEUILLE
        for nc, row in res["fv_levels"].items():
            cur.setdefault("fv_levels", {}).setdefault(nc, {}).update(row)
        cur.setdefault("lbm", {}).update(res["lbm"])
        dest.write_text(json.dumps(cur, indent=1, default=float))

    for nc in (int(v) for v in a.levels.split(",") if v):
        m = mesh(nc)
        row = res["fv_levels"].setdefault(str(nc), {})
        row.update({"nodes": m.n_nodes, "cells": m.n_elements, "elements": m.counts()})
        for meth in methods:
            if meth == "cpu-fv" and nc > 3:
                continue
            if meth == "lbm":
                continue
            print(f"nc {nc}: {meth}", flush=True)
            if meth == "gpu-fv":
                row[meth] = cleanest(lambda: run_gpu_fv(m), a.clean_tries)
            elif meth == "gpu-fv-before":
                row[meth] = cleanest(lambda: run_gpu_fv(m, before=True), a.clean_tries)
            elif meth == "openfoam":
                row[meth] = run_openfoam(m, nc, a.procs if nc > 6 else 1)
            elif meth == "cpu-fv":
                row[meth] = run_cpu_fv(m)
            row[meth]["load_average_at_start"] = load
            print(json.dumps(row[meth], default=float), flush=True)
            save()
    if "lbm" in methods:
        for Rv in (int(v) for v in a.lbm_radii.split(",")):
            print(f"lbm R {Rv}", flush=True)
            res["lbm"][str(Rv)] = cleanest(lambda: run_lbm(Rv), a.clean_tries)
            print(json.dumps(res["lbm"][str(Rv)], default=float), flush=True)
            save()


if __name__ == "__main__":
    main()
    os._exit(0)             # AmgX handles at interpreter exit
