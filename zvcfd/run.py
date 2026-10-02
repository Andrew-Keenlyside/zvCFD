"""Run a configuration end to end: geometry -> domain + patches -> solve -> stores -> collection.

One GPU, or several through :class:`zvcfd.lbm.MultiLBM` (``parallel.gpus``,
``parallel.partitions``). Every check interval the patch controller sets
patch values (pressures, flow-rate-controlled inlet velocities, Windkessel
outlet pressures), measures the exact lattice flux through every patch, and
tests convergence; ``monitors.json`` records the history and
``patches.csv`` the final flows and pressures.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from zvcfd import collection as col
from zvcfd.boundary import BoundarySet, Patch, face_patches, profile_scale
from zvcfd.config import RunConfig, resolve_method
from zvcfd.domain import FLUID, SOLID, BrickDomain
from zvcfd.units import Lattice

_UNIT_UM = {"m": 1e6, "cm": 1e4, "mm": 1e3, "um": 1.0, "micrometer": 1.0, "millimeter": 1e3}


# ---------------------------------------------------------------- geometry

def pad_to_bricks(flags: np.ndarray, brick: int) -> np.ndarray:
    pad = [(0, (-s) % brick) for s in flags.shape]
    return np.pad(flags, pad, constant_values=SOLID) if any(p[1] for p in pad) else flags


def _flags_from_source(cfg: RunConfig):
    src = cfg.source
    image = None
    if src.kind == "phantom":
        from zvcfd.phantoms import PHANTOMS

        flags = PHANTOMS[src.name]()
        voxel = src.voxel_size or 1.0
    elif src.kind == "npy":
        from zvcfd.io.omezarr import fluid_mask

        vol = np.load(src.path)
        flags = np.where(fluid_mask(vol, threshold=src.threshold, label=src.label), FLUID, SOLID)
        voxel = src.voxel_size or 1.0
    else:
        from zvcfd.io.omezarr import fluid_mask, read_level

        vol, ms = read_level(src.path, src.level, region=src.region)
        flags = np.where(fluid_mask(vol, threshold=src.threshold, label=src.label), FLUID, SOLID)
        voxel = src.voxel_size or max(ms.voxel_size(src.level))
        image = src.path
    return pad_to_bricks(flags.astype(np.uint8), cfg.domain.brick), float(voxel), image


def _patch_from_spec(name: str, spec: dict, base: Patch | None = None) -> Patch:
    kw = {k: v for k, v in spec.items() if k != "match"}
    for key in ("rcr", "coronary"):
        if kw.get(key) is not None:
            kw[key] = tuple(kw[key])
    if isinstance(kw.get("pim"), list):
        kw["pim"] = [tuple(p) for p in kw["pim"]]
    if kw.get("waveform") is not None:
        kw["waveform"] = [tuple(p) for p in kw["waveform"]]
    kind = kw.pop("kind")
    if base is not None:
        return Patch(name, kind, normal=base.normal, area=base.area, centroid=base.centroid, **kw)
    return Patch(name, kind, **kw)


def build_geometry(cfg: RunConfig, log=print, *, out: str | None = None):
    """``(domain, boundary, voxel_um, image, report)`` for a configuration."""
    report: dict = {}
    if cfg.source.kind == "mesh":
        # always through Zarr Vectors: the boundary store of a mesh collection (.zvmesh),
        # imported once from a raw mesh file if need be, is what gets voxelised
        from zvcfd.io.mesh_collection import resolve, voxelize_collection

        zvm = resolve(cfg.source.path, Path(out or cfg.output.path) / "meshes",
                      unit=cfg.source.unit, surface_only=True, log=log)
        voxel_um = float(cfg.source.voxel_size)
        t0 = time.time()
        domain, boundary, grid, report = voxelize_collection(zvm, voxel_um * 1e-6)
        log(f"voxelised {zvm.name} (Zarr Vectors boundary store) at {voxel_um:g} um: "
            f"{time.time() - t0:.1f} s")
        report |= {"mesh_collection": str(zvm.resolve()),
                   "origin_m": grid.origin[::-1].tolist(), "voxel": voxel_um * 1e-6}
        rules = cfg.boundaries.patches
        patches = []
        for p in boundary.patches:
            rule = next((r for r in rules if r["match"].lower() in p.name.lower()), None)
            if rule is None:
                raise ValueError(f"no boundaries.patches rule matches patch {p.name!r}")
            patches.append(_patch_from_spec(p.name, rule, p))
        boundary.patches = patches
        boundary.apply_flags(domain)
        for k, p in enumerate(patches):
            sel = boundary.patch == k
            boundary.scale[sel] = profile_scale(domain, boundary.cells[sel], p)
        return domain, boundary, voxel_um, None, report

    flags, voxel_um, image = _flags_from_source(cfg)
    domain = BrickDomain.from_flags(flags, periodic=cfg.domain.periodic)
    faces = dict(cfg.boundaries.faces)
    if cfg.physics.pressure_drop and not faces:
        faces = {"xmin": {"kind": "pressure", "pressure": cfg.physics.pressure_drop},
                 "xmax": {"kind": "pressure", "pressure": 0.0}}
    boundary = None
    if faces:
        area = (voxel_um * 1e-6) ** 2
        specs = {f: _patch_from_spec(f, s) for f, s in faces.items()}
        boundary = face_patches(domain, specs)
        for k, p in enumerate(boundary.patches):
            p.area = float((boundary.patch == k).sum()) * area
            if p.profile != "plug":
                sel = boundary.patch == k
                boundary.scale[sel] = profile_scale(domain, boundary.cells[sel], p)
    return domain, boundary, voxel_um, image, report


def lattice_for(cfg: RunConfig, voxel_um: float, log=print) -> Lattice:
    """The lattice: ``solver.tau`` if given, else dt from ``solver.mach`` at ``physics.u_ref``."""
    dx = voxel_um * 1e-6
    if cfg.solver.tau is not None:
        return Lattice(dx=dx, nu=cfg.physics.nu, tau=cfg.solver.tau, rho=cfg.physics.rho)
    if not cfg.physics.u_ref:
        return Lattice(dx=dx, nu=cfg.physics.nu, tau=1.0, rho=cfg.physics.rho)
    dt = cfg.solver.mach * dx / cfg.physics.u_ref
    tau = 0.5 + 3.0 * cfg.physics.nu * dt / dx ** 2
    if tau < 0.52:
        log(f"warning: tau = {tau:.4f} is close to 1/2; lower solver.mach or refine the voxels")
    return Lattice(dx=dx, nu=cfg.physics.nu, tau=tau, rho=cfg.physics.rho)


# ---------------------------------------------------------------- patch control

class PatchController:
    """Sets patch values each check interval and measures patch flows."""

    def __init__(self, sim, boundary: BoundarySet, lat: Lattice, *, flow_control: bool = True):
        self.sim, self.b, self.lat = sim, boundary, lat
        self.flow_control = flow_control
        self.gain = np.ones(len(boundary.patches))
        self.vol_per_flux = lat.dx ** 3 / lat.dt          # lattice mass flux -> m^3/s (rho ~ 1)

    def flows(self) -> np.ndarray:
        """Volume flow into the domain through each patch, m^3/s."""
        return self.sim.patch_flux() * self.vol_per_flux

    def pressures(self) -> np.ndarray:
        """Mean pressure over each patch's cells, Pa (relative to the reference density)."""
        st = self.sim.patch_velocity()
        rho = st[:, 3] / np.maximum(st[:, 4], 1)
        return np.array([self.lat.pressure(r - 1.0) for r in rho])

    def update(self, t: float, dt_check: float, q: np.ndarray | None) -> None:
        for i, p in enumerate(self.b.patches):
            if p.kind == "pressure":
                self.sim.set_patch(i, rho=1.0 + self.lat.drho_lattice(p.pressure * p.factor(t)))
            elif p.lumped:
                if q is None:
                    pr = p.outlet_model().pressure()
                else:
                    pr = p.outlet_pressure(-q[i], t, dt_check)
                self.sim.set_patch(i, rho=1.0 + self.lat.drho_lattice(pr))
            else:
                u = p.mean_velocity(t)
                if (self.flow_control and p.flow_rate is not None and q is not None
                        and q[i] > 0 and u > 0):
                    self.gain[i] *= 1.0 + 0.5 * (p.flow_rate * p.factor(t) / q[i] - 1.0)
                    self.gain[i] = float(np.clip(self.gain[i], 0.5, 2.0))
                ul = self.lat.u_lattice(u) * self.gain[i]
                self.sim.set_patch(i, u_zyx=tuple(-ul * np.asarray(p.normal)))


# ---------------------------------------------------------------- run

def make_solver(cfg: RunConfig, domain: BrickDomain, boundary, lat: Lattice):
    from zvcfd.lbm import CarreauYasuda, MultiLBM, SparseLBM

    rheo = None
    if cfg.physics.rheology:
        r = dict(cfg.physics.rheology)
        if r.pop("model", "carreau-yasuda") != "carreau-yasuda":
            raise ValueError("physics.rheology.model: only carreau-yasuda is implemented")
        rheo = CarreauYasuda.from_si(lat, **r)
    force = tuple(cfg.physics.body_force or (0.0, 0.0, 0.0))
    kw = dict(tau=lat.tau, force=(force[2], force[1], force[0]), collision=cfg.solver.collision,
              half=cfg.solver.precision == "fp16", rheology=rheo)
    parts = cfg.parallel.partitions or cfg.parallel.gpus
    if parts > 1:
        return MultiLBM(domain, n_parts=parts, chunk_bricks=cfg.domain.chunk_bricks,
                        devices=list(range(cfg.parallel.gpus)), boundary=boundary, **kw)
    return SparseLBM(domain, boundary=boundary, **kw)


def run(cfg: RunConfig, *, out: str | None = None, steps: int | None = None,
        log=print) -> Path:
    from zvcfd.io import fields as zf

    cfg.solver.method = resolve_method(cfg)        # a RunConfig built without from_dict
    if cfg.solver.method == "fv":
        from zvcfd.fv.run import run_fv

        return run_fv(cfg, out=out, steps=steps, log=log)
    if cfg.solver.method != "lbm":
        raise NotImplementedError(
            "`zvcfd run` drives the LBM and FV solvers; use zvcfd.solvers.lubrication directly")
    t0 = time.time()
    domain, boundary, voxel_um, image, report = build_geometry(cfg, log, out=out)
    lat = lattice_for(cfg, voxel_um, log)
    npatch = len(boundary.patches) if boundary is not None else 0
    log(f"domain {domain.shape}: {domain.n_bricks} bricks ({100 * domain.active_fraction:.2f}% "
        f"active), {domain.fluid_cells:,} fluid cells, fill {domain.fill:.2f}; {npatch} patches; "
        f"tau = {lat.tau:.4f}, dt = {lat.dt:.3g} s  ({time.time() - t0:.1f} s)")
    sim = make_solver(cfg, domain, boundary, lat)
    ctl = PatchController(sim, boundary, lat, flow_control=cfg.solver.flow_control) \
        if boundary is not None else None

    run_dir = Path(out or cfg.output.path) / f"{cfg.name}-{cfg.hash}.zvcfd"
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = col.run_document(run_dir, name=cfg.name,
                           attributes={"config_hash": cfg.hash, "voxel_size_um": voxel_um,
                                       "dt_s": lat.dt, "tau": lat.tau,
                                       "solver": cfg.solver.method})
    if image:
        col.add_image(doc, run_dir, image)
    if report.get("mesh_collection"):
        from zvcfd.io.mesh_collection import collection_info

        zvm = report["mesh_collection"]
        doc["attributes"][f"{col.PREFIX}:run"] |= {"mesh_collection": zvm,
                                                   "grid_origin_m": report["origin_m"]}
        col.add_node(doc, col.node(f"{col.PREFIX}:mesh", "mesh-collection",
                                   col.rel(zvm, run_dir),
                                   attributes={f"{col.PREFIX}:mesh": collection_info(zvm)}))
    domain_path = run_dir / "domain.zarrvectors"
    lv = zf.create_brick_store(domain_path, domain, voxel_size=voxel_um, fields={},
                               chunk_bricks=cfg.domain.chunk_bricks, flags=True,
                               compressor=cfg.output.compressor)
    zf.write_brick_chunks(lv, domain, {"flags": domain.flags}, voxel_size=voxel_um,
                          chunk_bricks=cfg.domain.chunk_bricks)
    zf.finalize_brick_store(lv)
    col.add_domain(doc, run_dir, domain_path,
                   summary=domain.summary() | {"shape": list(domain.shape)})
    config_path = run_dir / "config.json"
    col.atomic_write_json(config_path, cfg.as_dict())
    col.add_node(doc, col.node(f"{col.PREFIX}:config", "config", col.rel(config_path, run_dir),
                               path_type="json"))
    col.write(run_dir, doc)

    def host_fields():
        import cupy as cp

        f = sim.fields()
        return {k: (cp.asnumpy(v) if hasattr(v, "__cuda_array_interface__") else v)
                for k, v in f.items()}

    def snapshot():
        f = host_fields()
        path = run_dir / "fields" / f"step-{sim.steps:09d}.zarrvectors"
        path.parent.mkdir(exist_ok=True)
        names = [n for n in cfg.output.fields if n in f]
        lvl = zf.create_brick_store(path, domain, voxel_size=voxel_um,
                                    fields={n: cfg.output.dtype for n in names},
                                    chunk_bricks=cfg.domain.chunk_bricks,
                                    compressor=cfg.output.compressor,
                                    shard_shape=cfg.output.shard_shape)
        zf.write_brick_chunks(lvl, domain, {n: f[n].astype(cfg.output.dtype) for n in names},
                              voxel_size=voxel_um, chunk_bricks=cfg.domain.chunk_bricks)
        zf.finalize_brick_store(lvl)
        col.add_snapshot(doc, run_dir, path, step=sim.steps, time_s=sim.steps * lat.dt,
                         fields=names)
        col.write(run_dir, doc)            # the snapshot becomes visible here, atomically
        log(f"  wrote {path.name}")

    total = steps or cfg.solver.steps
    every = cfg.solver.check_every
    monitors = []
    prev = None
    if ctl is not None:
        ctl.update(0.0, every * lat.dt, None)
    t_solve = time.time()
    converged = False
    fluid = domain.flags.reshape(-1) == FLUID
    while sim.steps < total:
        n = min(every, total - sim.steps)
        sim.step(n)
        t = sim.steps * lat.dt
        rec = {"step": sim.steps, "t": t, "wall_s": time.time() - t_solve}
        if ctl is not None:
            q = ctl.flows()
            rec["q"] = q.tolist()
            ctl.update(t, n * lat.dt, q)
            q_in = q[q > 0].sum()
            change = (np.abs(q - prev).max() / max(q_in, 1e-30)) if prev is not None else np.inf
            imbalance = float(q.sum() / max(q_in, 1e-30))
            rec["change"], rec["imbalance"] = float(change), imbalance
            log(f"  step {sim.steps:>8d}  inflow {q_in:.4e} m3/s  imbalance "
                f"{imbalance:+.2e}  change {change:.2e}")
            prev = q.copy()
        else:
            ux = float(host_fields()["ux"].reshape(-1)[fluid].mean())
            change = abs(ux - prev) / max(abs(ux), 1e-30) if prev is not None else np.inf
            rec["mean_ux"] = ux
            log(f"  step {sim.steps:>8d}  mean u_x {ux:.4e} (lattice)")
            prev = ux
        converged = change <= cfg.solver.tolerance and (
            ctl is None or abs(rec["imbalance"]) <= cfg.solver.mass_tolerance)
        monitors.append(rec)
        if cfg.output.every and sim.steps % cfg.output.every == 0:
            snapshot()
        if converged:
            log(f"  converged: change <= {cfg.solver.tolerance:g}"
                + (f", |imbalance| <= {cfg.solver.mass_tolerance:g}" if ctl is not None else ""))
            break
    wall = time.time() - t_solve
    mlups = domain.fluid_cells * sim.steps / wall / 1e6
    if not any(n["id"] == f"step-{sim.steps:09d}" for n in col.snapshots(doc)):
        snapshot()
    summary = {"steps": sim.steps, "converged": bool(converged), "solve_s": wall,
               "mlups": mlups, "fluid_cells": domain.fluid_cells, "tau": lat.tau,
               "dt_s": lat.dt, "voxel_um": voxel_um,
               "geometry": {k: v for k, v in report.items() if k != "patch_cells"}}
    col.atomic_write_json(run_dir / "monitors.json", {"summary": summary, "history": monitors})
    if ctl is not None:
        _write_patch_table(run_dir / "patches.csv", boundary, ctl.flows(), ctl.pressures())
    log(f"done: {sim.steps} steps, {mlups:.0f} MLUPS incl. checks/output, "
        f"{time.time() - t0:.1f} s total -> {run_dir}")
    return run_dir


def _write_patch_table(path: Path, boundary: BoundarySet, q: np.ndarray, p: np.ndarray) -> None:
    out_total = -q[q < 0].sum()
    counts = boundary.counts()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patch", "kind", "cells", "area_m2", "flow_m3s", "split", "pressure_pa"])
        for i, pt in enumerate(boundary.patches):
            split = (-q[i] / out_total) if q[i] < 0 and out_total > 0 else ""
            w.writerow([pt.name, pt.kind, int(counts[i]), pt.area, float(q[i]), split,
                        float(p[i])])


def load_patch_table(path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


__all__ = ["PatchController", "build_geometry", "lattice_for", "load_patch_table",
           "make_solver", "run"]
