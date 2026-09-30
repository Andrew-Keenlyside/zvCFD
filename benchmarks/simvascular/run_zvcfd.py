"""zvCFD on the cut coronary trees of VMR 0066_H_CORO_H, driven by SimVascular's boundary data.

    python benchmarks/simvascular/run_zvcfd.py --voxel 100 --cycles 3 [--inlet sv|parabolic]

Boundary conditions, all periodic with the 1 s cardiac cycle:

- each inlet (the cut across the left main and across the proximal RCA):
  SimVascular's flow through the cut, which for rigid walls is the sum of
  the tree's outlet flows at each instant; flow-rate controlled, with
  either SimVascular's own velocity profile on the cut (``--inlet sv``,
  from ``svvolume.npz``: the normal velocity at each inlet voxel, every
  5 ms) or a parabolic profile (``--inlet parabolic``);
- each of the 24 outlets: SimVascular's mean pressure over the cap, less
  a reference ``p_ref(t)`` (the mean over the 24 outlets). Adding a
  uniform ``p_ref(t)`` changes nothing in incompressible flow and keeps the
  lattice density near 1.

Newtonian blood as in SimVascular (rho = 1060 kg/m^3, mu = 4 mPa s).
Lattice velocity 0.05 at 0.5 m/s (about the peak coronary velocity), so
dt = dx / 10 m/s and a 1 kPa pressure difference is a 3 % density
difference.

Records, per run directory ``<out>/vmr0066-<voxel>um-<inlet>/``:

- ``monitors.npz``: time, flow through every patch (every half ms) and
  mean patch pressure (every ms), all cycles;
- ``probes.npz``: velocity at the near-wall probe points and the section
  points (``probes.npz`` from ``svref.py probes``) every 5 ms of the last
  cycle;
- ``fields_<ms>.npz``: the full velocity and pressure field at the phases
  in ``--snapshots`` (ms into the last cycle);
- ``run.json``: lattice, geometry, timings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmrcase as vc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RES = ROOT / "benchmarks" / "results" / "simvascular"
U_REF, U_LAT = 0.5, 0.05
RAMP = 0.1                   # s: boundary data ramped up from rest (the first cycle is discarded)


class RampedPatch:
    """Mixin: a patch whose waveform is multiplied by a smooth start-up ramp."""

    def factor(self, t: float) -> float:
        r = 0.5 - 0.5 * np.cos(np.pi * min(t / RAMP, 1.0))
        return super().factor(t) * r


def build(voxel_um: float, log=print):
    from zvcfd.geometry import voxelize_mesh

    t0 = time.time()
    surf, names, info = vc.coronary_surface()
    tri, fid = vc.triangles(surf)
    kinds = vc.zone_kinds(names, np.unique(fid))
    domain, boundary, grid, rep = voxelize_mesh(tri, fid, kinds, voxel_um * 1e-4,
                                                zone_names=names, unit_scale=vc.UNIT_SCALE)
    rep["voxelize_s"] = time.time() - t0
    rep["cuts"] = info
    log(f"{voxel_um:g} um: {domain.fluid_cells:,} fluid voxels in {domain.n_bricks:,} bricks "
        f"(fill {domain.fill:.2f}), {len(boundary.patches)} patches, "
        f"odd columns {rep['odd_columns_dropped']}, {rep['voxelize_s']:.0f} s")
    if rep["patches_without_cells"]:
        raise RuntimeError(f"patches without voxels: {rep['patches_without_cells']}")
    return domain, boundary, grid, rep


def sv_boundary_data():
    d = np.load(RES / "svsurface.npz")
    caps = [str(c) for c in d["caps"]]
    t, q, p = d["t"], d["q_out"], d["p"]
    cor = [i for i, c in enumerate(caps) if c not in ("inflow", "aorta")]
    p_ref = p[:, cor].mean(1)
    flows = {"LCA_inlet": q[:, [i for i in cor if vc.outlet_tree(caps[i]) == "LCA"]].sum(1),
             "RCA_inlet": q[:, [i for i in cor if vc.outlet_tree(caps[i]) == "RCA"]].sum(1)}
    pressures = {caps[i]: p[:, i] - p_ref for i in cor}
    return t, flows, pressures, p_ref


def main():
    from zvcfd.boundary import Patch, cell_coords, profile_scale
    from zvcfd.lbm import SparseLBM
    from zvcfd.run import PatchController
    from zvcfd.units import Lattice

    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=100.0, help="micrometre")
    ap.add_argument("--cycles", type=float, default=3.0)
    ap.add_argument("--inlet", choices=["sv", "parabolic"], default="sv")
    ap.add_argument("--snapshots", default="150,300,500,700",
                    help="ms into the last cycle at which to save full fields")
    ap.add_argument("--out", default="/hdd/data/zvcfd_vmr/runs")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--u-lat", type=float, default=U_LAT,
                    help="lattice velocity at 0.5 m/s (sets dt; tau rises with it)")
    args = ap.parse_args()

    tag = f"vmr0066-{args.voxel:g}um-{args.inlet}" + (
        f"-ulat{args.u_lat:g}" if args.u_lat != U_LAT else "")
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    domain, boundary, grid, rep = build(args.voxel, log)
    dx = args.voxel * 1e-6
    # dt from the lattice velocity, rounded so that 1 ms is a whole, even number of steps
    per_ms = int(np.ceil(1e-3 / (args.u_lat * dx / U_REF) / 2) * 2)
    dt = 1e-3 / per_ms
    tau = 0.5 + 3.0 * vc.NU * dt / dx ** 2
    lat = Lattice(dx=dx, nu=vc.NU, tau=tau, rho=vc.RHO)
    log(f"dt = {dt:.4g} s ({per_ms} steps/ms), tau = {tau:.4f}, "
        f"u_lat at 0.5 m/s = {lat.u_lattice(0.5):.4f}, drho per kPa = {lat.drho_lattice(1e3):.4f}")

    t_sv, flows, pressures, p_ref = sv_boundary_data()
    names = [p.name for p in boundary.patches]
    Ramped = type("RampedPatch", (RampedPatch, Patch), {})
    patches = []
    for p in boundary.patches:
        if p.name in flows:
            qm = float(flows[p.name].mean())
            patches.append(Ramped(p.name, "velocity", flow_rate=qm, normal=p.normal, area=p.area,
                                 centroid=p.centroid, profile="parabolic",
                                 waveform=list(zip(t_sv.tolist(), (flows[p.name] / qm).tolist()))))
        else:
            patches.append(Ramped(p.name, "pressure", pressure=1.0, normal=p.normal, area=p.area,
                                 centroid=p.centroid,
                                 waveform=list(zip(t_sv.tolist(), pressures[p.name].tolist()))))
    boundary.patches = patches
    boundary.apply_flags(domain)
    for k, p in enumerate(patches):
        sel = boundary.patch == k
        boundary.scale[sel] = profile_scale(domain, boundary.cells[sel], p)

    # SimVascular's inlet profiles: normal velocity at each inlet voxel, every 5 ms
    profiles = {}
    if args.inlet == "sv":
        from svref import inlet_profiles

        for k, p in enumerate(patches):
            if p.kind != "velocity":
                continue
            sel = np.flatnonzero(boundary.patch == k)
            xyz = grid.origin[::-1] + (cell_coords(domain, boundary.cells[sel])[:, ::-1] + 0.5) \
                * grid.voxel
            un = inlet_profiles(xyz, -np.asarray(p.normal)[::-1])          # (201, n) m/s, inward
            mean = un.mean(1, keepdims=True)
            profiles[k] = (sel, np.where(np.abs(mean) > 0, un / mean, 1.0).astype(np.float32))
            log(f"  {p.name}: SimVascular profile on {len(sel)} voxels, "
                f"peak/mean {profiles[k][1].max(1).mean():.2f}")

    sim = SparseLBM(domain, tau=lat.tau, collision="trt", boundary=boundary)
    ctl = PatchController(sim, boundary, lat, flow_control=True)
    import cupy as cp

    def set_profiles(t):
        ph = (t % vc.PERIOD) / 0.005
        i0 = int(np.floor(ph)) % (len(t_sv) - 1)
        w = ph - np.floor(ph)
        for sel, prof in profiles.values():
            sim.bscale[cp.asarray(sel)] = cp.asarray((1 - w) * prof[i0] + w * prof[i0 + 1])

    probes = None
    if not args.no_probes and (RES / "probes.npz").exists():
        probes = ProbeSampler(domain, grid, np.load(RES / "probes.npz")["points"])
        log(f"  {len(probes.points):,} probe points ({probes.outside} outside the voxels)")

    half = per_ms // 2
    total = int(round(args.cycles * vc.PERIOD * 1e3)) * per_ms
    last0 = total - int(round(vc.PERIOD * 1e3)) * per_ms
    snaps = {int(s) for s in args.snapshots.split(",") if s}
    rec_t, rec_q, rec_pt, rec_p, probe_t, probe_u = [], [], [], [], [], []
    set_profiles(0.0)
    ctl.update(0.0, half * dt, None)
    t_solve = time.time()
    while sim.steps < total:
        sim.step(half)
        t = sim.steps * dt
        q = ctl.flows()
        if not np.isfinite(q).all():
            log(f"diverged at step {sim.steps} (t = {t:.4f} s)")
            break
        set_profiles(t)
        ctl.update(t, half * dt, q)
        rec_t.append(t)
        rec_q.append(q)
        if sim.steps % per_ms == 0:
            rec_pt.append(t)
            rec_p.append(ctl.pressures())
        ms = (sim.steps - last0) // per_ms if sim.steps >= last0 else -1
        if sim.steps >= last0 and (sim.steps - last0) % per_ms == 0:
            if probes is not None and ms % 5 == 0:
                probe_t.append(t)
                probe_u.append(probes.sample(sim) * (dx / dt))
            if ms in snaps:
                save_fields(out / f"fields_{ms:04d}.npz", sim, domain, grid, lat, t)
        if sim.steps % (100 * per_ms) == 0:
            qi = q[[i for i, p in enumerate(patches) if p.kind == "velocity"]]
            el = time.time() - t_solve
            log(f"  t = {t:.3f} s  inlets {qi[0] * 1e6:.3f} {qi[1] * 1e6:.3f} mL/s  "
                f"imbalance {q.sum() / max(q[q > 0].sum(), 1e-30):+.1e}  "
                f"{domain.fluid_cells * sim.steps / el / 1e6:.0f} MLUPS  {el:.0f} s")
    wall = time.time() - t_solve
    np.savez_compressed(out / "monitors.npz", t=np.array(rec_t), q=np.array(rec_q),
                        tp=np.array(rec_pt), p=np.array(rec_p), names=np.array(names),
                        kinds=np.array([p.kind for p in patches]),
                        area=np.array([p.area for p in patches]),
                        counts=boundary.counts())
    if probe_u:
        np.savez_compressed(out / "probes.npz", t=np.array(probe_t),
                            u=np.array(probe_u, np.float32), inside=probes.inside)
    run = {"voxel_um": args.voxel, "inlet": args.inlet, "cycles": args.cycles,
           "u_lat": args.u_lat,
           "steps": sim.steps, "dt_s": dt, "tau": lat.tau, "fluid_voxels": domain.fluid_cells,
           "bricks": domain.n_bricks, "fill": domain.fill, "stored_cells": domain.stored_cells,
           "device_gb": sim.device_bytes / 1e9, "solve_s": wall,
           "mlups": domain.fluid_cells * sim.steps / wall / 1e6,
           "geometry": {k: v for k, v in rep.items() if k != "patch_cells"},
           "patch_cells": dict(zip(names, map(int, boundary.counts())))}
    (out / "run.json").write_text(json.dumps(run, indent=1, default=float))
    log(f"done: {sim.steps} steps in {wall:.0f} s ({run['mlups']:.0f} MLUPS) -> {out}")


def voxel_centres(domain, grid) -> tuple[np.ndarray, np.ndarray]:
    """Global cell ids and centres (x, y, z, model units) of every non-solid voxel."""
    from zvcfd.boundary import cell_coords
    from zvcfd.domain import SOLID

    b = domain.brick ** 3
    br, loc = np.nonzero(domain.flags != SOLID)
    cells = br.astype(np.int64) * b + loc
    xyz = grid.origin[::-1] + (cell_coords(domain, cells)[:, ::-1] + 0.5) * grid.voxel
    return cells, xyz


def save_fields(path: Path, sim, domain, grid, lat, t: float) -> None:
    """Velocity (m/s) and pressure (Pa, relative to p_ref) at every non-solid voxel."""
    f = sim.fields()
    cells, xyz = voxel_centres(domain, grid)
    b = domain.brick ** 3
    br, loc = cells // b, cells % b
    u = np.stack([f[k].get()[br, loc] for k in ("ux", "uy", "uz")], 1) * (lat.dx / lat.dt)
    p = lat.pressure(f["rho"].get()[br, loc] - 1.0)
    np.savez_compressed(path, t=t, xyz=xyz.astype(np.float32), u=u.astype(np.float32),
                        p=p.astype(np.float32), flag=domain.flags[br, loc], voxel=grid.voxel)


class ProbeSampler:
    """Trilinear interpolation of the voxel velocity at fixed points (x, y, z, model units).

    Corners that are solid are dropped and the weights renormalised; a
    point with no fluid corner is marked outside and reads as NaN.
    """

    def __init__(self, domain, grid, points_xyz: np.ndarray):
        import cupy as cp

        from zvcfd.boundary import cell_flags, cell_ids
        from zvcfd.domain import SOLID

        self.points = points_xyz
        v = grid.to_voxel(points_xyz[:, ::-1])                        # zyx, centres at integers
        base = np.floor(v).astype(np.int64)
        fr = v - base
        ids, ws = [], []
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    c = cell_ids(domain, base + np.array([dz, dy, dx]))
                    ok = (c >= 0) & (cell_flags(domain, np.maximum(c, 0)) != SOLID)
                    w = ((fr[:, 0] if dz else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1])
                         * (fr[:, 2] if dx else 1 - fr[:, 2]))
                    ids.append(np.where(ok, c, 0))
                    ws.append(np.where(ok, w, 0.0))
        ids, ws = np.stack(ids, 1), np.stack(ws, 1)
        tot = ws.sum(1)
        self.inside = tot > 1e-3
        ws = np.where(self.inside[:, None], ws / np.maximum(tot, 1e-12)[:, None], 0.0)
        self.outside = int((~self.inside).sum())
        self.ids = cp.asarray(ids)
        self.w = cp.asarray(ws.astype(np.float32))

    def sample(self, sim) -> np.ndarray:
        """``(N, 3)`` velocity (x, y, z) in lattice units; NaN outside."""
        import cupy as cp

        f = sim.fields()
        out = cp.stack([(f[k].reshape(-1)[self.ids] * self.w).sum(1) for k in ("ux", "uy", "uz")],
                       1)
        u = out.get()
        u[~self.inside] = np.nan
        return u


if __name__ == "__main__":
    main()
