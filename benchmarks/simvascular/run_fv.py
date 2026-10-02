"""The finite-volume solver on the SimVascular case, on SimVascular's own mesh.

    python benchmarks/simvascular/run_fv.py [--solver gpu|reference] [--cycles 1] [--dt 1e-3]
    python benchmarks/simvascular/run_fv.py --segment --solver reference --steps 20   # a test

The mesh is SimVascular's tetrahedral mesh cropped to the cut coronary
trees (:mod:`fvcase`), and the boundary data are SimVascular's own
solution node by node: velocity on the two inlet cuts, pressure on the 24
outlet caps. The run starts from SimVascular's solution at the start of
the cycle. Newtonian blood as in SimVascular, BDF2.

Writes, to ``<out>/fv-<solver>[-segment]/``, the same files as
``run_zvcfd.py`` so ``compare.py`` reads both codes alike:
``monitors.npz`` (flow into the domain through every zone, m³/s, and
mean zone pressure relative to ``p_ref``, every 5 ms), ``probes.npz``
(velocity at the shared probe points, every 5 ms), ``fields_<ms>.npz``
(nodal velocity and pressure at the snapshot phases) and ``run.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fvcase  # noqa: E402
import vmrcase as vc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RES = ROOT / "benchmarks" / "results" / "simvascular"


def segment_region(path="RCA", a0=2.5, a1=3.3):
    """A short segment of one vessel between two arc lengths: for tests."""
    from scipy.spatial import cKDTree

    p = vc.read_path(path)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    arcs = np.arange(0.0, s[-1], 0.01)
    tr = cKDTree(np.stack([np.interp(arcs, s, p[:, k]) for k in range(3)], 1))

    def region(c):
        d, i = tr.query(c)
        return (d < 0.3) & (arcs[i] > a0) & (arcs[i] < a1)

    def cut_kind(fc):
        _, i = tr.query(fc)
        return [("segment_inlet", "velocity") if a < 0.5 * (a0 + a1) else
                ("segment_outlet", "pressure") for a in arcs[i]]

    return region, cut_kind


def p_ref_of():
    s = np.load(RES / "svsurface.npz")
    caps = [str(c) for c in s["caps"]]
    cor = [i for i, c in enumerate(caps) if c not in ("inflow", "aorta")]
    t, pr = s["t"], s["p"][:, cor].mean(1)
    return lambda tt: float(np.interp(tt % vc.PERIOD, t, pr))


class Probes:
    """Nodal velocity at the shared probe points, through SimVascular's tetrahedra
    (the mesh is SimVascular's, so its linear interpolation is the solver's own)."""

    def __init__(self, case):
        from svref import SVVolume

        pr = np.load(RES / "probes.npz", allow_pickle=True)
        sv = SVVolume()
        cells, w = sv.locate(pr["points"])
        fv_of_sv = -np.ones(len(sv.points), np.int64)
        fv_of_sv[case.node_sv] = np.arange(len(case.node_sv))
        nodes = fv_of_sv[sv.tets[np.maximum(cells, 0)]]
        self.inside = (cells >= 0) & (nodes >= 0).all(1)
        self.nodes = np.where(self.inside[:, None], nodes, 0)
        self.w = np.where(self.inside[:, None], w, 0.0)

    def sample(self, U):
        u = (U[self.nodes] * self.w[..., None]).sum(1)
        u[~self.inside] = np.nan
        return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["gpu", "reference"], default="gpu")
    ap.add_argument("--segment", action="store_true", help="an 8 mm RCA segment (a test)")
    ap.add_argument("--cycles", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=None, help="override the number of steps")
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--loops", type=int, default=5)
    ap.add_argument("--backflow", type=float, default=0.2,
                    help="backflow stabilisation coefficient at the outlets (svSolver's: 0.2; "
                         "0 for none)")
    ap.add_argument("--precision", choices=["mixed", "double"], default="mixed",
                    help="AmgX hierarchy precision (GPU solver)")
    ap.add_argument("--linear", choices=["amgx", "simple"], default="amgx",
                    help="GPU linear solver: AmgX coupled AMG, or the SIMPLE-type block "
                         "preconditioner (robust on thin prism layers; double only)")
    ap.add_argument("--smoother", default=None, help="AmgX smoother (e.g. ilu0; default dilu)")
    ap.add_argument("--linear-rtol", type=float, default=0.1,
                    help="residual reduction of each linear solve (GPU solver; CFX: 0.1)")
    ap.add_argument("--residual-target", type=float, default=None,
                    help="stop each step's coefficient loops at this max RMS normalised "
                         "residual (CFX-style, e.g. 1e-5), if the solver supports it")
    ap.add_argument("--snapshots", default="160,265,530,800")
    ap.add_argument("--out", default=str(vc.VMR_DIR / "runs"))
    ap.add_argument("--kernels", choices=["fast", "classic"], default=None,
                    help="GPU assembly kernels (default: the solver's)")
    ap.add_argument("--store-geometry", choices=["auto", "true", "tets", "false"], default=None,
                    help="per-element geometry kept on the device (default: the solver's)")
    ap.add_argument("--mesh-store", default=None,
                    help="Zarr Vectors mesh collection the solver reads its mesh from "
                         "(default: <VMR>/meshes/<case>.zvmesh; written on first use)")
    args = ap.parse_args()

    tag = f"fv-{args.solver}" + ("-segment" if args.segment else "") + (
        f"-{args.linear}" if args.linear != "amgx" else "") + (
        f"-{args.smoother}" if args.smoother else "") + (
        "-nobackflow" if not args.backflow else "") + (
        f"-rtol{args.linear_rtol:g}" if args.linear_rtol != 0.1 else "")
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    region, cut_kind = segment_region() if args.segment else (None, None)
    store = Path(args.mesh_store) if args.mesh_store else \
        vc.VMR_DIR / "meshes" / ("coronary-segment.zvmesh" if args.segment else "coronary.zvmesh")
    case = fvcase.build(region=region, cut_kind=cut_kind, log=log, store=store)
    data = fvcase.SVBoundaryData(case)
    bcs = fvcase.boundary_conditions(case, data, backflow=args.backflow)
    if args.solver == "gpu":
        from zvcfd.fv.solver import GPUSolver as Solver
    else:
        from zvcfd.fv.reference import ReferenceSolver as Solver
    from zvcfd.fv.reference import Fluid

    fluid = Fluid(rho=vc.RHO, mu=vc.MU)
    if args.solver == "gpu":
        opts = {"precision": args.precision if args.linear == "amgx" else "double"}
        if args.smoother:
            opts["smoother"] = args.smoother
        kw = {"kernels": args.kernels} if args.kernels else {}
        if args.store_geometry:
            kw["store_geometry"] = {"true": True, "false": False}.get(args.store_geometry,
                                                                       args.store_geometry)
        sim = Solver(case.mesh, fluid, bcs, linear=args.linear, linear_rtol=args.linear_rtol,
                     linear_options=opts, **kw)
    else:
        sim = Solver(case.mesh, fluid, bcs)
    x0 = data.at_nodes(0.0)
    sim.initialise(U=x0[:, :3].astype(float), P=x0[:, 3].astype(float))
    log(f"set up in {time.time() - t0:.0f} s")

    p_ref = p_ref_of()
    probes = Probes(case)
    zones = sorted(case.mesh.zones)
    names = [case.mesh.zones[z].name for z in zones]
    kinds = ["velocity" if bcs[z]["kind"] == "velocity" else
             ("wall" if bcs[z]["kind"] == "wall" else "pressure") for z in zones]
    znodes = {z: np.unique(case.mesh.zones[z].faces[case.mesh.zones[z].faces >= 0])
              for z in zones}
    steps = args.steps or int(round(args.cycles * vc.PERIOD / args.dt))
    every = int(round(0.005 / args.dt))
    last0 = steps - int(round(vc.PERIOD / args.dt)) if steps * args.dt >= vc.PERIOD else 0
    snaps = {int(s) for s in args.snapshots.split(",") if s}
    rec = {"t": [], "q": [], "p": []}
    wss = None
    if hasattr(sim, "wall_shear"):
        from zvcfd.fv.wss import WallShearStats

        wall_nodes = sim.wall_shear()[0]
        wss = WallShearStats(len(wall_nodes))
    probe_t, probe_u = [], []
    n = {"step": 0}
    t_solve = time.time()

    def callback(s):
        n["step"] += 1
        k = n["step"]
        if k % every:
            return
        t = s.t
        flows = s.zone_flows()
        rec["t"].append(t)
        rec["q"].append([-flows[z] / vc.RHO for z in zones])       # into the domain, m^3/s
        P = s.P if isinstance(s.P, np.ndarray) else s.cp.asnumpy(s.P)
        rec["p"].append([float(P[znodes[z]].mean()) - p_ref(t) for z in zones])
        if k >= last0:
            U = s.U if isinstance(s.U, np.ndarray) else s.cp.asnumpy(s.U)
            probe_t.append(t)
            probe_u.append(probes.sample(U))
            if wss is not None:
                wss.add(t, s.wall_shear()[1])
            ms = int(round((k - last0) * args.dt * 1e3))
            if ms in snaps:
                x = case.mesh.nodes / vc.UNIT_SCALE
                P = s.P if isinstance(s.P, np.ndarray) else s.cp.asnumpy(s.P)
                np.savez_compressed(out / f"fields_{ms:04d}.npz", t=t, xyz=x.astype(np.float32),
                                    u=U.astype(np.float32),
                                    p=(P - p_ref(t)).astype(np.float32),
                                    flag=np.zeros(len(x), np.uint8), voxel=np.nan)
        if k % (20 * every) == 0 or k == steps:
            el = time.time() - t_solve
            qin = sum(q for q, kd in zip(rec["q"][-1], kinds) if kd == "velocity")
            log(f"  t = {t:.3f} s  inflow {qin * 1e6:.3f} mL/s  balance "
                f"{sum(rec['q'][-1]) / max(abs(qin), 1e-30):+.1e}  {el / k:.2f} s/step")

    import inspect

    kw = {}
    if args.residual_target is not None:
        if "residual_target" not in inspect.signature(sim.solve_transient).parameters:
            raise SystemExit("this solver has no residual_target yet")
        kw["residual_target"] = args.residual_target
    rep = sim.solve_transient(args.dt, steps, scheme="bdf2", loops=args.loops, callback=callback,
                              **kw)
    wall = time.time() - t_solve
    np.savez_compressed(out / "monitors.npz", t=np.array(rec["t"]), q=np.array(rec["q"]),
                        tp=np.array(rec["t"]), p=np.array(rec["p"]), names=np.array(names),
                        kinds=np.array(kinds), area=np.full(len(zones), np.nan),
                        counts=np.array([len(znodes[z]) for z in zones]))
    if wss is not None and wss.T > 0:
        r = wss.results()
        np.savez_compressed(out / "wss.npz", nodes=wall_nodes,
                            xyz=case.mesh.nodes[wall_nodes] / vc.UNIT_SCALE,
                            node_sv=case.node_sv[wall_nodes], tawss=r["tawss"], osi=r["osi"])
    if probe_u:
        np.savez_compressed(out / "probes.npz", t=np.array(probe_t),
                            u=np.array(probe_u, np.float32), inside=probes.inside)
    run = {"code": "fv", "solver": args.solver, "segment": args.segment, "dt_s": args.dt,
           "steps": steps, "cycles": steps * args.dt / vc.PERIOD, "loops": args.loops,
           "residual_target": args.residual_target,
           "backflow": args.backflow, "linear_rtol": args.linear_rtol,
           "precision": args.precision if args.solver == "gpu"
           else "double", "nodes": case.mesh.n_nodes,
           "elements": int(len(case.mesh.elements["tet"])), "solve_s": wall,
           "iterations": rep.iterations, "voxel_um": None, "inlet": "sv", "u_lat": None,
           "mesh_store": str(store), "kernels": getattr(sim, "kernels", None)}
    hist = [h for h in rep.history if isinstance(h, dict)]
    if hist and "linear_iterations" in hist[0]:
        li = np.array([h["linear_iterations"] for h in hist])
        lc = np.array([h["linear_converged"] for h in hist])
        run["linear"] = {"solver": args.linear, "smoother": args.smoother or "default",
                         "outer_iterations": len(hist),
                         "linear_iterations_mean": float(li.mean()),
                         "linear_iterations_max": int(li.max()),
                         "not_converged": int((~lc).sum()),
                         "assemble_s_mean": float(np.mean([h["assemble_s"] for h in hist])),
                         "linear_s_mean": float(np.mean([h["linear_setup_s"] + h["linear_solve_s"]
                                                         for h in hist]))}
        (out / "history.json").write_text(json.dumps(
            [{k: v for k, v in h.items() if isinstance(v, (int, float, bool))} for h in hist],
            default=float))
        log(f"linear solves: {run['linear']}")
    (out / "run.json").write_text(json.dumps(run, indent=1, default=float))
    log(f"done: {steps} steps in {wall:.0f} s -> {out}")


if __name__ == "__main__":
    main()
