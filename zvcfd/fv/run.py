"""``zvcfd run`` with ``solver.method: fv``: mesh → boundary conditions → GPU solve → stores.

The configuration is the one the voxel solvers read (:mod:`zvcfd.config`);
the finite-volume solver takes its settings from the ``fv`` section and
needs no voxel size. The steps:

1. **Mesh**, always from Zarr Vectors: ``source.path`` is a mesh collection
   (``.zvmesh``, written by ``zvcfd import-mesh``), or any mesh
   :func:`zvcfd.io.mesh_collection.read_source` reads (Fluent ``.msh``,
   ``.vtu``, a SimVascular ``mesh-complete`` folder, meshio formats), which
   is imported once, in metres, into ``<output.path>/meshes`` and reused.
   The solver's nodes, elements and zones are read back from the
   collection's volume and boundary stores.
2. **Zones**: each boundary zone takes the first ``boundaries.patches`` rule
   whose ``match`` is a substring of its name (case-insensitive); an
   unmatched zone the mesh calls a wall is a wall. Rules use the patch keys
   of :class:`zvcfd.boundary.Patch` (velocity with ``flow_rate`` or
   ``velocity``, ``profile`` and ``waveform``; pressure with ``waveform``;
   ``rcr`` and ``coronary`` lumped outlets), plus ``wall``/``symmetry``
   kinds and the pressure-zone options ``backflow_stabilisation``,
   ``pressure_profile: average`` and ``opening``.
3. **Solve**: :class:`zvcfd.fv.solver.GPUSolver`, steady to ``fv.tolerance``;
   with ``fv.dt`` a transient run follows (by default from that steady
   state), with lumped outlets advanced every step and wall shear statistics
   accumulated over the last period.
4. **Stores**, in a run collection (``<name>-<hash>.zvcfd``): a link to
   the mesh collection (its volume store is the snapshots' mesh), a
   boundary store (:mod:`zvcfd.io.mesh_store`), node-field snapshots
   (:mod:`zvcfd.io.node_fields`), TAWSS/OSI/RRT and the final WSS as
   boundary-store vertex attributes, ``config.json``, ``monitors.json``
   and ``patches.csv``.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from zvcfd import collection as col
from zvcfd.config import RunConfig
from zvcfd.io.mesh_collection import UNIT_M  # noqa: F401  (re-exported)

_FV_KEYS = ("backflow_stabilisation", "pressure_profile", "opening")


# ---------------------------------------------------------------- mesh

def mesh_collection(cfg: RunConfig, log=print, *, out: str | None = None) -> Path:
    """The Zarr Vectors mesh collection the run solves on: ``source.path`` if it is one
    (``.zvmesh``), else ``source.path`` imported once into ``<output.path>/meshes``."""
    from zvcfd.io.mesh_collection import resolve

    return resolve(cfg.source.path, Path(out or cfg.output.path) / "meshes",
                   unit=cfg.source.unit, chunk=cfg.fv.chunk, log=log)


def load_mesh(cfg: RunConfig, log=print):
    """The mesh in metres, read back from its Zarr Vectors stores (:func:`mesh_collection`)."""
    from zvcfd.io.mesh_collection import read_mesh

    return read_mesh(mesh_collection(cfg, log))


# ---------------------------------------------------------------- boundary conditions

def _zone_area(mesh, zone) -> float:
    t = zone.triangles()
    x = mesh.nodes
    return float(0.5 * np.linalg.norm(np.cross(x[t[:, 1]] - x[t[:, 0]],
                                               x[t[:, 2]] - x[t[:, 0]]), axis=1).sum())


def boundary_conditions(cfg: RunConfig, mesh):
    """``(bcs, table)``: the solver's ``{zone: spec}`` and one row per zone for reports."""
    from zvcfd.run import _patch_from_spec

    rules = cfg.boundaries.patches
    bcs, table = {}, []
    for zid, z in sorted(mesh.zones.items()):
        rule = next((r for r in rules if r["match"].lower() in z.name.lower()), None)
        if rule is None:
            if "wall" not in z.kind.lower() and "wall" not in z.name.lower():
                raise ValueError(f"no boundaries.patches rule matches zone {z.name!r} "
                                 f"({z.kind})")
            rule = {"match": z.name, "kind": "wall"}
        kind = rule["kind"]
        area = _zone_area(mesh, z)
        row = {"zone": zid, "name": z.name, "kind": kind, "faces": z.n_faces, "area_m2": area}
        table.append(row)
        if kind in ("wall", "symmetry"):
            bcs[zid] = {"kind": kind}
            continue
        extra = {k: rule[k] for k in _FV_KEYS if k in rule}
        patch = _patch_from_spec(z.name, {k: v for k, v in rule.items() if k not in _FV_KEYS})
        patch.area = area
        if kind == "velocity":
            q = patch.flow_rate if patch.flow_rate is not None else patch.velocity * area
            profile = {"parabolic": "poiseuille"}.get(patch.profile, patch.profile)
            bcs[zid] = {"kind": "velocity", "profile": profile,
                        "flow_rate": (lambda t, p=patch, q=q: q * p.factor(t))}
            continue
        spec = {"kind": "pressure",
                "backflow_stabilisation": extra.get("backflow_stabilisation",
                                                    cfg.fv.backflow_stabilisation)}
        if patch.lumped:
            spec["lumped"] = patch.outlet_model()
        else:
            spec["value"] = (lambda x, t, p=patch: np.full(len(x), p.pressure * p.factor(t)))
            if extra.get("pressure_profile") == "average":
                spec["profile"] = "average"
                spec["value"] = patch.pressure if not patch.waveform else \
                    [(t, patch.pressure * f) for t, f in patch.waveform]
            if extra.get("opening"):
                spec["opening"] = True
                spec["value"] = patch.pressure if not patch.waveform else \
                    [(t, patch.pressure * f) for t, f in patch.waveform]
        bcs[zid] = spec
    return bcs, table


def fluid_for(cfg: RunConfig):
    from zvcfd.fv.reference import Fluid
    from zvcfd.rheology import CarreauYasuda

    ph = cfg.physics
    visc = None
    if ph.rheology:
        r = dict(ph.rheology)
        if r.pop("model", "carreau-yasuda") != "carreau-yasuda":
            raise ValueError("physics.rheology.model: the GPU FV solver takes carreau-yasuda")
        visc = CarreauYasuda(**r)
    return Fluid(ph.rho, ph.rho * ph.nu, viscosity=visc)


def _period(cfg: RunConfig) -> float | None:
    if cfg.fv.period:
        return cfg.fv.period
    ends = [float(np.asarray(r["waveform"], float)[-1, 0]) for r in cfg.boundaries.patches
            if r.get("waveform")]
    ends += [float(np.asarray(r["pim"], float)[-1, 0]) for r in cfg.boundaries.patches
             if isinstance(r.get("pim"), list)]
    return max(ends) if ends else None


def make_gpu_solver(cfg: RunConfig, mesh, bcs, fluid):
    """One GPU (:class:`GPUSolver`), or partitions over ``parallel.gpus`` devices
    (:class:`MultiGPUSolver`) when ``parallel.partitions`` or ``parallel.gpus`` exceeds one."""
    from zvcfd.fv.linear import amgx_library
    from zvcfd.fv.solver import GPUSolver

    fv = cfg.fv
    linear = fv.linear
    parts = cfg.parallel.partitions or cfg.parallel.gpus
    if parts > 1:
        from zvcfd.fv.multi import MultiGPUSolver

        if linear == "host-direct":
            raise ValueError("fv.linear host-direct needs one partition")
        if linear == "auto":
            linear = "amgx" if amgx_library() else "acm"
        return MultiGPUSolver(mesh, fluid, bcs, parts=parts,
                              devices=list(range(cfg.parallel.gpus)), advection=fv.advection,
                              dt=fv.false_dt, linear=linear, linear_rtol=fv.linear_rtol)
    opts = {"precision": fv.precision} if linear in ("amgx", "auto") else None
    if fv.precision == "mixed" and linear not in ("amgx", "auto"):
        raise ValueError("fv.precision mixed needs fv.linear amgx or auto")
    return GPUSolver(mesh, fluid, bcs, advection=fv.advection, dt=fv.false_dt, linear=linear,
                     linear_rtol=0.0 if linear == "host-direct" else fv.linear_rtol,
                     linear_options=opts)


# ---------------------------------------------------------------- run

def run_fv(cfg: RunConfig, *, out: str | None = None, steps: int | None = None,
           log=print) -> Path:
    from zvcfd.fv.wss import WallShearStats
    from zvcfd.io.mesh_collection import VOLUME, collection_info, read_mesh
    from zvcfd.io.mesh_store import write_boundary_store
    from zvcfd.io.node_fields import write_node_fields

    t0 = time.time()
    fv = cfg.fv
    zvm = mesh_collection(cfg, log, out=out)
    zinfo = collection_info(zvm)
    t1 = time.time()
    mesh = read_mesh(zvm)
    log(f"read {zvm.name} (Zarr Vectors, {zinfo['chunk'] * 1e3:g} mm chunks): "
        f"{time.time() - t1:.1f} s")
    bcs, table = boundary_conditions(cfg, mesh)
    fluid = fluid_for(cfg)
    log(f"mesh {mesh.n_nodes:,} nodes, {mesh.counts()}; zones "
        + ", ".join(f"{r['name']}={r['kind']}" for r in table)
        + f"  ({time.time() - t0:.1f} s)")
    s = make_gpu_solver(cfg, mesh, bcs, fluid)
    log(f"solver set up: linear {s.linear.name}, {s.setup_seconds:.1f} s")

    run_dir = Path(out or cfg.output.path) / f"{cfg.name}-{cfg.hash}.zvcfd"
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = col.run_document(run_dir, name=cfg.name, unit="meter",
                           attributes={"config_hash": cfg.hash, "solver": "fv",
                                       "nodes": mesh.n_nodes, "elements": mesh.counts(),
                                       "mesh_collection": str(zvm.resolve())})
    chunk = float(zinfo["chunk"])
    # the solver's mesh is the collection's volume store: snapshots point at it, not a copy
    mesh_path, bnd_path = zvm / VOLUME, run_dir / "boundary.zarrvectors"
    col.add_node(doc, col.node(f"{col.PREFIX}:mesh", "mesh-collection", col.rel(zvm, run_dir),
                               attributes={f"{col.PREFIX}:mesh": zinfo}))
    col.add_node(doc, col.node(f"{col.PREFIX}:fv-mesh", "mesh", col.rel(mesh_path, run_dir)))
    write_boundary_store(bnd_path, mesh, chunk=chunk)
    col.add_node(doc, col.node(f"{col.PREFIX}:fv-boundary", "boundary",
                               col.rel(bnd_path, run_dir)))
    config_path = run_dir / "config.json"
    col.atomic_write_json(config_path, cfg.as_dict())
    col.add_node(doc, col.node(f"{col.PREFIX}:config", "config", col.rel(config_path, run_dir),
                               path_type="json"))
    col.write(run_dir, doc)

    names = {r["zone"]: r["name"] for r in table}
    history = []
    state = {"step": 0}

    def snapshot():
        f = s.fields()
        fields = {"velocity": f["U"], "pressure": f["P"]}
        if fv.wss:
            nodes, tau = s.wall_shear()
            fields |= {"wall_nodes": nodes, "wss": tau}
        path = run_dir / "fields" / f"step-{state['step']:09d}.zarr"
        path.parent.mkdir(exist_ok=True)
        write_node_fields(path, fields, step=state["step"], time_s=s.t,
                          mesh=col.rel(mesh_path, path), dtype=cfg.output.dtype)
        col.add_snapshot(doc, run_dir, path, step=state["step"], time_s=s.t,
                         fields=sorted(fields))
        col.write(run_dir, doc)
        log(f"  wrote {path.name}")

    def record(kind, extra=None):
        q, p = s.patch_flows(), s.patch_pressures()
        rec = {"kind": kind, "t": s.t, "step": state["step"],
               "flow_m3s": {names[z]: q[z] for z in q}, "pressure_pa": {names[z]: p[z] for z in p}}
        history.append(rec | (extra or {}))

    steady = None
    transient = fv.dt is not None
    if not transient or fv.steady_start:
        rep = s.solve(max_iterations=fv.iterations, tol=fv.tolerance, log=log,
                      residual_target=fv.residual_target)
        steady = {"iterations": rep.iterations, "converged": rep.converged,
                  "seconds": rep.seconds, "final": rep.history[-1] if rep.history else {}}
        record("steady", {"outer_iterations": rep.iterations})
        log(f"  steady: {rep.iterations} outer iterations, converged {rep.converged}, "
            f"{rep.seconds:.1f} s")
        if not transient or cfg.output.every:
            snapshot()

    trans = None
    if transient:
        period = _period(cfg)
        n = steps or fv.steps or (int(round(fv.periods * period / fv.dt))
                                  if fv.periods and period else None)
        if not n:
            raise ValueError("a transient fv run needs fv.steps, or fv.periods with a period")
        t_end = s.t + n * fv.dt
        avg_from = fv.tawss_from if fv.tawss_from is not None else \
            (t_end - period if period else s.t)
        stats = None

        def after_step(solver, dt):
            nonlocal stats
            state["step"] += 1
            if fv.wss and solver.t >= avg_from - 1e-12 * max(abs(avg_from), 1.0):
                nodes, tau = solver.wall_shear()
                if stats is None:
                    stats = (nodes, WallShearStats(len(nodes)))
                stats[1].add(solver.t, tau)
            record("step")
            if cfg.output.every and state["step"] % cfg.output.every == 0:
                snapshot()

        rep = s.solve_transient(fv.dt, n, scheme=fv.scheme, loops=fv.loops,
                                tol=fv.loop_tolerance, callback=after_step, log=None,
                                residual_target=fv.residual_target)
        trans = {"steps": n, "dt": fv.dt, "loops": rep.iterations, "all_converged":
                 rep.converged, "seconds": rep.seconds}
        log(f"  transient: {n} steps of {fv.dt:g} s, {rep.iterations} coefficient loops, "
            f"{rep.seconds:.1f} s")
        if not any(sn["id"] == f"step-{state['step']:09d}" for sn in col.snapshots(doc)):
            snapshot()
        if stats is not None and stats[1].T > 0:
            nodes, st = stats
            wall = {}
            for k, v in st.results().items():
                a = np.full(mesh.n_nodes, np.nan)
                a[nodes] = v
                wall[k] = a
            write_boundary_store(bnd_path, mesh, chunk=chunk, node_fields=wall)
            log(f"  TAWSS/OSI/RRT over {st.T:g} s -> {bnd_path.name}")
    if not transient and fv.wss:
        nodes, tau = s.wall_shear()
        w = np.full((mesh.n_nodes, 3), np.nan)
        w[nodes] = tau
        write_boundary_store(bnd_path, mesh, chunk=chunk,
                             node_fields={"wss": w, "wss_magnitude": np.linalg.norm(w, axis=1)})

    summary = {"nodes": mesh.n_nodes, "elements": mesh.counts(), "linear": s.linear.name,
               "linear_switches": getattr(s.linear, "switches", []),
               "setup_s": s.setup_seconds, "steady": steady, "transient": trans,
               "wall_s": time.time() - t0}
    col.atomic_write_json(run_dir / "monitors.json", {"summary": summary, "history": history})
    write_patch_table(run_dir / "patches.csv", table, s)
    log(f"done: {time.time() - t0:.1f} s -> {run_dir}")
    return run_dir


def write_patch_table(path: Path, table: list[dict], s) -> None:
    """One row per zone: kind, faces, area, flow into the domain, outflow split, mean pressure."""
    q, p = s.patch_flows(), s.patch_pressures()
    out_total = sum(v for v in q.values() if v > 0)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patch", "kind", "faces", "area_m2", "flow_m3s", "split", "pressure_pa"])
        for r in table:
            if r["kind"] == "wall":
                continue
            zid = r["zone"]
            split = q[zid] / out_total if q[zid] > 0 and out_total > 0 else ""
            w.writerow([r["name"], r["kind"], r["faces"], r["area_m2"], -q[zid], split, p[zid]])


__all__ = ["boundary_conditions", "fluid_for", "load_mesh", "make_gpu_solver", "run_fv",
           "write_patch_table"]
