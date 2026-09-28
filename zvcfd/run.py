"""Run a configuration end to end on one GPU: source -> domain -> solve -> ZV stores -> collection.

Multi-GPU execution (one process per device, NCCL/peer halo exchange,
chunk ownership from :meth:`BrickDomain.partition`) follows the same
store and collection layout; it is on the roadmap, not built
(``docs/feasibility/roadmap.md``).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from zvcfd import collection as col
from zvcfd.config import RunConfig
from zvcfd.domain import FLUID, RESERVOIR, SOLID, BrickDomain
from zvcfd.units import Lattice


def load_flags(cfg: RunConfig) -> tuple[np.ndarray, float, str | None]:
    """Flag volume (z, y, x), voxel size (micrometre) and the source image path, if any."""
    src = cfg.source
    image = None
    if src.kind == "phantom":
        from zvcfd.phantoms import PHANTOMS

        flags = PHANTOMS[src.name]()
        voxel = src.voxel_size or 1.0
    elif src.kind == "npy":
        vol = np.load(src.path)
        from zvcfd.io.omezarr import fluid_mask

        flags = np.where(fluid_mask(vol, threshold=src.threshold, label=src.label), FLUID, SOLID)
        voxel = src.voxel_size or 1.0
    else:
        from zvcfd.io.omezarr import fluid_mask, read_level

        vol, ms = read_level(src.path, src.level, region=src.region)
        flags = np.where(fluid_mask(vol, threshold=src.threshold, label=src.label), FLUID, SOLID)
        voxel = src.voxel_size or max(ms.voxel_size(src.level))
        image = src.path
    return pad_to_bricks(flags.astype(np.uint8), cfg.domain.brick), float(voxel), image


def pad_to_bricks(flags: np.ndarray, brick: int) -> np.ndarray:
    pad = [(0, (-s) % brick) for s in flags.shape]
    return np.pad(flags, pad, constant_values=SOLID) if any(p[1] for p in pad) else flags


def add_x_reservoirs(flags: np.ndarray) -> np.ndarray:
    """Fixed-density reservoirs on the fluid voxels of both x faces."""
    out = flags.copy()
    for x in (0, -1):
        face = out[:, :, x]
        face[face == FLUID] = RESERVOIR
    return out


def run(cfg: RunConfig, *, out: str | None = None, steps: int | None = None,
        log=print) -> Path:
    from zvcfd.io import fields as zf
    from zvcfd.lbm import SparseLBM

    if cfg.parallel.gpus != 1:
        raise NotImplementedError("multi-GPU runs are not built yet; set parallel.gpus: 1")
    if cfg.solver.method != "lbm":
        raise NotImplementedError(
            "`zvcfd run` drives the LBM solver; use zvcfd.solvers.lubrication directly")
    t0 = time.time()
    flags, voxel_um, image = load_flags(cfg)
    lat = Lattice(dx=voxel_um * 1e-6, nu=cfg.physics.nu, tau=cfg.solver.tau, rho=cfg.physics.rho)
    drho = 0.0
    if cfg.physics.pressure_drop:
        flags = add_x_reservoirs(flags)
        drho = lat.drho_lattice(cfg.physics.pressure_drop)
    dom = BrickDomain.from_flags(flags, periodic=cfg.domain.periodic)
    log(f"domain {dom.shape}: {dom.n_bricks} bricks ({100 * dom.active_fraction:.1f}% active), "
        f"{dom.fluid_cells:,} fluid cells, fill {dom.fill:.2f}; dt = {lat.dt:.3g} s, "
        f"drho = {drho:.3g}")
    force = tuple(cfg.physics.body_force or (0.0, 0.0, 0.0))
    sim = SparseLBM(dom, tau=cfg.solver.tau, force=(force[2], force[1], force[0]),
                    rho_in=1 + drho / 2, rho_out=1 - drho / 2,
                    half=cfg.solver.precision == "fp16")

    run_dir = Path(out or cfg.output.path) / f"{cfg.name}-{cfg.hash}.zvcfd"
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = col.run_document(run_dir, name=cfg.name,
                           attributes={"config_hash": cfg.hash, "voxel_size_um": voxel_um,
                                       "dt_s": lat.dt, "solver": cfg.solver.method})
    if image:
        col.add_image(doc, run_dir, image)
    domain_path = run_dir / "domain.zarrvectors"
    lv = zf.create_brick_store(domain_path, dom, voxel_size=voxel_um, fields={},
                               chunk_bricks=cfg.domain.chunk_bricks, flags=True,
                               compressor=cfg.output.compressor)
    zf.write_brick_chunks(lv, dom, {"flags": dom.flags}, voxel_size=voxel_um,
                          chunk_bricks=cfg.domain.chunk_bricks)
    zf.finalize_brick_store(lv)
    col.add_domain(doc, run_dir, domain_path, summary=dom.summary() | {"shape": list(dom.shape)})
    config_path = run_dir / "config.json"
    col.atomic_write_json(config_path, cfg.as_dict())
    col.add_node(doc, col.node(f"{col.PREFIX}:config", "config", col.rel(config_path, run_dir),
                                path_type="json"))
    col.write(run_dir, doc)

    def snapshot():
        import cupy as cp

        f = sim.fields()
        path = run_dir / "fields" / f"step-{sim.steps:09d}.zarrvectors"
        path.parent.mkdir(exist_ok=True)
        names = [n for n in cfg.output.fields if n in f]
        lvl = zf.create_brick_store(path, dom, voxel_size=voxel_um,
                                    fields={n: cfg.output.dtype for n in names},
                                    chunk_bricks=cfg.domain.chunk_bricks,
                                    compressor=cfg.output.compressor,
                                    shard_shape=cfg.output.shard_shape)
        zf.write_brick_chunks(lvl, dom, {n: cp.asnumpy(f[n]).astype(cfg.output.dtype)
                                         for n in names},
                              voxel_size=voxel_um, chunk_bricks=cfg.domain.chunk_bricks)
        zf.finalize_brick_store(lvl)
        col.add_snapshot(doc, run_dir, path, step=sim.steps, time_s=sim.steps * lat.dt,
                         fields=names)
        col.write(run_dir, doc)            # the snapshot becomes visible here, atomically
        log(f"  wrote {path.name}")

    total = steps or cfg.solver.steps
    every = cfg.solver.check_every
    prev = None
    t_solve = time.time()
    while sim.steps < total:
        n = min(every, total - sim.steps)
        sim.step(n)
        flux = _mean_ux(sim)
        log(f"  step {sim.steps:>8d}  mean u_x {flux:.4e} (lattice)")
        if cfg.output.every and sim.steps % cfg.output.every == 0:
            snapshot()
        if prev is not None and flux != 0 and abs(flux - prev) <= cfg.solver.tolerance * abs(flux):
            log(f"  converged: relative change <= {cfg.solver.tolerance:g}")
            break
        prev = flux
    import cupy as cp

    cp.cuda.Device().synchronize()
    mlups = dom.fluid_cells * sim.steps / (time.time() - t_solve) / 1e6
    if not snapshots_has(doc, sim.steps):
        snapshot()
    log(f"done: {sim.steps} steps, {mlups:.0f} MLUPS incl. checks/output, "
        f"{time.time() - t0:.1f} s total -> {run_dir}")
    return run_dir


def snapshots_has(doc: dict, step: int) -> bool:
    return any(n["id"] == f"step-{step:09d}" for n in col.snapshots(doc))


def _mean_ux(sim) -> float:
    f = sim.fields()
    fluid = sim.flag == FLUID
    ux = f["ux"].reshape(-1)
    return float((ux * fluid).sum() / max(int(fluid.sum()), 1))
