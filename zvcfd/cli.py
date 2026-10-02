"""``zvcfd`` command line.

Subcommands follow the zarr-vectors-tools / bridge-sim conventions:
argparse, kebab-case flags with snake_case dests, ``main(argv) -> int``,
comma-separated shapes (``--shape 2048,2048,2048``).

    zvcfd probe [--require cupy,zv_device_decode] [--json]
    zvcfd plan --fluid-cells 6.3e8 [--fill 0.6] [--gpus 8] [--gpu H100-SXM] [--method lbm-fp32]
    zvcfd plan --shape 2048,2048,2048 --fluid-fraction 0.2 --geometry porous --layout dense
    zvcfd import-mesh coronary.msh [--out coronary.zvmesh] [--unit mm] [--surface-only]
    zvcfd mesh-info coronary.msh [--voxel-size 0.02,0.01,0.005] [--unit mm]
    zvcfd voxelize coronary.msh --voxel-size 50 [--unit mm] [--out domain.zarrvectors]
    zvcfd phantom list | zvcfd phantom build <name> --out mask.npy
    zvcfd run config.yaml [--out runs/] [--steps N]
    zvcfd info <run.zvcfd | store.zarrvectors>
"""

from __future__ import annotations

import argparse
import json
import sys


def _floats(s: str) -> list[float]:
    return [float(v) for v in s.split(",") if v]


def _ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v]


def _fmt_s(sec: float | None) -> str:
    if sec is None:
        return "-"
    for unit, n in (("d", 86400), ("h", 3600), ("min", 60)):
        if sec >= n:
            return f"{sec / n:.1f} {unit}"
    return f"{sec:.2f} s"


# ---------------------------------------------------------------- probe

def cmd_probe(args) -> int:
    from zvcfd.capabilities import device_count, runtime_capabilities

    caps = runtime_capabilities(probe_device=True)
    if args.json:
        print(json.dumps(caps | {"device_count": device_count()}, indent=1))
    else:
        width = max(map(len, caps))
        for k, v in caps.items():
            print(f"{k:<{width}}  {'yes' if v else 'no'}")
        print(f"{'device_count':<{width}}  {device_count()}")
    missing = [r for r in (args.require.split(",") if args.require else []) if not caps.get(r)]
    if missing:
        print(f"missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------- plan

def cmd_plan(args) -> int:
    from zvcfd.perfmodel import estimate

    box = None
    if args.shape:
        z, y, x = _ints(args.shape)
        box = float(z) * y * x
    fluid = args.fluid_cells
    if fluid is None:
        if box is None or args.fluid_fraction is None:
            print("give --fluid-cells, or --shape with --fluid-fraction", file=sys.stderr)
            return 2
        fluid = box * args.fluid_fraction
    rows = []
    for method in args.method.split(","):
        e = estimate(fluid, fill=args.fill, box_cells=box, gpus=args.gpus, gpu=args.gpu,
                     method=method, layout=args.layout, geometry=args.geometry,
                     steps=args.steps, efficiency=args.efficiency)
        rows.append(e.as_dict())
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    print(f"{fluid:.3g} fluid cells, {args.layout} layout, {args.geometry} geometry, "
          f"{args.gpus} x {args.gpu}, {args.efficiency} kernel efficiency")
    print(f"{'method':<16}{'stored':>10}{'GB/GPU':>9}{'fits':>6}{'Gupd/s':>9}"
          f"{'ms/step':>10}{'total':>12}")
    for r in rows:
        print(f"{r['method']:<16}{r['stored_cells']:>10.3g}{r['memory_per_gpu_gb']:>9.1f}"
              f"{'yes' if r['fits'] else 'NO':>6}{r['updates_per_s'] / 1e9:>9.1f}"
              f"{r['seconds_per_step'] * 1e3:>10.2f}{_fmt_s(r['seconds']):>12}")
    return 0


# ---------------------------------------------------------------- mesh-info

def cmd_mesh_info(args) -> int:
    from zvcfd.io.fluent_msh import read_fluent_boundary, voxel_estimate
    from zvcfd.perfmodel import estimate

    if args.volume:
        return _mesh_info_volume(args)
    s = read_fluent_boundary(args.mesh).summary()
    est = [voxel_estimate(s, h) for h in _floats(args.voxel_size)]
    if args.json:
        print(json.dumps({"summary": s, "voxel_estimates": est}, indent=1))
        return 0
    u = args.unit
    ext = [hi - lo for lo, hi in zip(s["bbox_min"], s["bbox_max"])]
    print(f"{s['source']}")
    print(f"  {s['nodes']:,} nodes, {s['cells']:,} cells, {s['faces']:,} faces "
          f"({s['boundary_faces']:,} on the boundary)")
    print(f"  bounding box {' x '.join(f'{e:.2f}' for e in ext)} {u}; "
          f"enclosed volume {s['volume']:.4g} {u}^3")
    for kind, k in sorted(s["kinds"].items()):
        print(f"  {kind:<16} {k['zones']:>4} zones {k['faces']:>10,} faces  "
              f"area {k['area']:.4g} {u}^2")
    d = sorted(p["equivalent_diameter"] for p in s["patches"] if "outlet" in p["kind"])
    if d:
        print(f"  outlet diameters {d[0]:.3g} .. {d[-1]:.3g} {u} (median {d[len(d) // 2]:.3g})")
    print(f"\n  {'voxel':>8} {'fluid voxels':>14} {'box voxels':>12} {'fluid %':>8} "
          f"{'min patch (vox)':>16} {'GB/GPU':>8} {'fits':>5} {'Gupd/s':>8}")
    for e in est:
        p = estimate(e["fluid_voxels"], fill=args.fill, gpus=args.gpus, gpu=args.gpu,
                     method=args.method, geometry="vessel")
        print(f"  {e['voxel_size']:>8g} {e['fluid_voxels']:>14.3g} {e['box_voxels']:>12.3g} "
              f"{100 * e['fluid_fraction']:>8.2f} {e['min_patch_diameter_voxels'] or 0:>16.1f} "
              f"{p.memory_per_gpu_gb:>8.1f} {'yes' if p.fits else 'NO':>5} "
              f"{p.updates_per_s / 1e9:>8.1f}")
    return 0


def _mesh_info_volume(args) -> int:
    """The finite-volume view: elements, control volumes, couplings, GPU memory."""
    from zvcfd.fv.geometry import dual_geometry, topology
    from zvcfd.fv.pattern import coupling_stats, memory_estimate, node_graph
    from zvcfd.mesh import read_fluent_mesh

    mesh = read_fluent_mesh(args.mesh, unit=args.unit)
    geo = dual_geometry(mesh, keep_areas=False).report
    cpl = coupling_stats(node_graph(mesh)[0])
    n_ip = sum(len(e) * topology(k).n_ip for k, e in mesh.elements.items())
    mem = memory_estimate(cpl["blocks"], mesh.n_nodes, n_ip)
    out = {"summary": mesh.summary(), "reader": mesh.meta, "dual": geo, "couplings": cpl,
           "integration_points": n_ip, "memory_gb": mem}
    if args.json:
        print(json.dumps(out, indent=1, default=float))
        return 0
    u = args.unit
    print(f"{mesh.source}  (read in {mesh.meta['total_s']:.0f} s)")
    print(f"  {mesh.n_nodes:,} nodes; " + ", ".join(f"{n:,} {k}" for k, n in mesh.counts().items()))
    print(f"  volume {geo['volume_elements']:.6g} {u}^3; control volumes sum to it within "
          f"{geo['volume_rel_diff']:.1e}, close within {geo['closure_max_rel']:.1e}; "
          f"min orthogonality {geo['orthogonality_min_deg']:.1f} deg")
    print(f"  coupled system: {mesh.n_nodes:,} block rows, {cpl['blocks']:,} 4x4 blocks "
          f"({cpl['mean']:.1f} per row, max {cpl['max']}); {n_ip:,} integration points")
    print("  GPU memory: " + ", ".join(f"{k} {v:.1f} GB" for k, v in mem.items()))
    return 0


# ---------------------------------------------------------------- import-mesh

def cmd_import_mesh(args) -> int:
    from zvcfd.io.mesh_collection import collection_info, import_mesh

    out = import_mesh(args.mesh, args.out, unit=args.unit, chunk=args.chunk,
                      surface_only=args.surface_only)
    info = collection_info(out)
    lo, hi = info["bounds_m"]
    print(f"  {info['nodes']:,} nodes, {info['elements'] or 'surface only'}; chunk "
          f"{info['chunk'] * 1e3:g} mm; box {[round((b - a) * 1e3, 3) for a, b in zip(lo, hi)]} mm")
    for z in info["zones"]:
        print(f"  zone {z['zone']:>4} {z['kind']:<16} {z['faces']:>9,} faces  {z['name']}")
    return 0


# ---------------------------------------------------------------- voxelize

def cmd_voxelize(args) -> int:
    from zvcfd.io.mesh_collection import is_mesh_collection, voxelize_collection

    if is_mesh_collection(args.mesh):
        dom, bnd, grid, rep = voxelize_collection(args.mesh, args.voxel_size * 1e-6)
    else:
        from zvcfd.geometry import voxelize_fluent
        from zvcfd.run import _UNIT_UM

        unit_um = _UNIT_UM[args.unit]
        dom, bnd, grid, rep = voxelize_fluent(args.mesh, args.voxel_size / unit_um,
                                              unit_scale=unit_um * 1e-6)
    counts = bnd.counts()
    print(f"{args.mesh}: {args.voxel_size:g} um voxels, box {grid.shape}, "
          f"{rep['fluid_voxels']:,} fluid voxels in {dom.n_bricks:,} bricks (fill {dom.fill:.2f}); "
          f"{rep['odd_columns_dropped']} leaky columns dropped")
    kinds = {}
    for p, c in zip(bnd.patches, counts):
        kinds.setdefault(p.kind, []).append(int(c))
    for k, c in kinds.items():
        print(f"  {k:<9} {len(c):>3} patches, {sum(c):>8,} cells (min {min(c)}, max {max(c)})")
    if rep["patches_without_cells"]:
        print(f"  patches with no cells: {rep['patches_without_cells']}")
    if args.out:
        from zvcfd.io import fields as zf

        lv = zf.create_brick_store(args.out, dom, voxel_size=args.voxel_size, fields={},
                                   chunk_bricks=args.chunk_bricks, flags=True, compressor="zstd")
        zf.write_brick_chunks(lv, dom, {"flags": dom.flags}, voxel_size=args.voxel_size,
                              chunk_bricks=args.chunk_bricks)
        zf.finalize_brick_store(lv)
        print(f"  wrote {args.out}")
    return 0


# ---------------------------------------------------------------- phantom

def cmd_phantom(args) -> int:
    from zvcfd.phantoms import PHANTOMS

    if args.action == "list":
        for name in PHANTOMS:
            print(name)
        return 0
    import numpy as np

    flags = PHANTOMS[args.name]()
    np.save(args.out, flags == 0)
    print(f"{args.name}: {flags.shape}, fluid {100 * (flags == 0).mean():.2f}% -> {args.out}")
    return 0


# ---------------------------------------------------------------- run / info

def cmd_run(args) -> int:
    from zvcfd.config import load
    from zvcfd.run import run

    run(load(args.config), out=args.out, steps=args.steps)
    return 0


def cmd_info(args) -> int:
    from pathlib import Path

    p = Path(args.target)
    if (p / "zarr.json").exists():
        attrs = json.loads((p / "zarr.json").read_text()).get("attributes", {})
        if "ome" in attrs and attrs["ome"].get("type") == "collection" and \
                "zvcfd:run" in attrs["ome"].get("attributes", {}):
            from zvcfd import collection as col

            doc = attrs["ome"]
            print(f"run {doc['name']} ({doc['id']})")
            for n in doc["nodes"]:
                print(f"  {n['type']:<16} {n['id']:<12} {n.get('path', {}).get('path', '')}")
            snaps = col.snapshots(doc)
            if snaps:
                print(f"  {len(snaps)} snapshots: {snaps[0]['id']} .. {snaps[-1]['id']}")
            return 0
        if "ome" in attrs and "zvcfd:mesh" in attrs["ome"].get("attributes", {}):
            info = attrs["ome"]["attributes"]["zvcfd:mesh"]
            print(f"mesh collection {p.name}: {info['nodes']:,} nodes, "
                  f"{info['elements'] or 'surface only'}, {len(info['zones'])} zones, "
                  f"chunk {info['chunk']:g} m; from {info['source']}")
            for n in attrs["ome"]["nodes"]:
                print(f"  {n['type']:<18} {n['id']:<10} {n['path']['path']}")
            return 0
        if "zarr_vectors" in attrs:
            import zarr_vectors as zv

            ds = zv.open(str(p))
            meta = dict(ds.metadata["zvcfd"]) if "zvcfd" in ds.metadata else {}
            print(f"zarr vectors store {p.name}: levels {len(ds.levels)}; zvcfd {meta}")
            return 0
    print(f"{p}: not a zvcfd run, mesh collection or store", file=sys.stderr)
    return 1


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="zvcfd", description="GPU CFD on Zarr Vectors stores: a "
                                 "CFX-style finite-volume solver on meshes, and a "
                                 "lattice-Boltzmann solver on voxels.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="report what this install and machine can do")
    p.add_argument("--require", default="", help="comma list of capabilities that must be present")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_probe)

    def plan_opts(p):
        p.add_argument("--fill", type=float, default=0.6,
                       help="brick fill: fluid / stored cells (sparse layout)")
        p.add_argument("--gpus", type=int, default=8)
        p.add_argument("--gpu", default="H100-SXM")
        p.add_argument("--method", default="lbm-fp32",
                       help="comma list: lbm-fp32, lbm-fp16, lubrication-amg")

    p = sub.add_parser("plan", help="memory and time estimate for a domain")
    p.add_argument("--fluid-cells", type=float, dest="fluid_cells")
    p.add_argument("--shape", help="bounding box Z,Y,X in voxels")
    p.add_argument("--fluid-fraction", type=float, dest="fluid_fraction")
    p.add_argument("--layout", default="sparse", choices=["sparse", "dense"])
    p.add_argument("--geometry", default="vessel", choices=["open", "porous", "vessel"])
    p.add_argument("--steps", type=int)
    p.add_argument("--efficiency", default="planning", choices=["planning", "measured", "target"],
                   help="kernel efficiency basis (planning = the lower of measured and target)")
    p.add_argument("--json", action="store_true")
    plan_opts(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("mesh-info", help="summarise an ASCII Fluent .msh and size a voxel run")
    p.add_argument("mesh")
    p.add_argument("--volume", action="store_true",
                   help="read the volume mesh: elements, control volumes, couplings, FV memory")
    p.add_argument("--voxel-size", default="0.02,0.01,0.005", dest="voxel_size",
                   help="comma list, in mesh units")
    p.add_argument("--unit", default="mm")
    p.add_argument("--json", action="store_true")
    plan_opts(p)
    p.set_defaults(func=cmd_mesh_info)

    p = sub.add_parser("import-mesh",
                       help="import a mesh into a Zarr Vectors mesh collection (.zvmesh)")
    p.add_argument("mesh", help="Fluent .msh, .vtu, SimVascular mesh-complete folder, or any "
                                "meshio format")
    p.add_argument("--out", help="collection path (default: <mesh>.zvmesh beside the mesh)")
    p.add_argument("--unit", default="mm", help="the source's coordinate unit; stored in metres")
    p.add_argument("--chunk", type=float, help="spatial chunk edge, metres "
                                               "(default: a quarter of the longest box edge)")
    p.add_argument("--surface-only", action="store_true", dest="surface_only",
                   help="boundary store only (enough for the voxel solver)")
    p.set_defaults(func=cmd_import_mesh)

    p = sub.add_parser("voxelize", help="voxelise a mesh collection or a Fluent mesh into a "
                                        "sparse domain with patches")
    p.add_argument("mesh", help=".zvmesh collection, or a Fluent .msh")
    p.add_argument("--voxel-size", type=float, required=True, dest="voxel_size",
                   help="micrometre")
    p.add_argument("--unit", default="mm", help="Fluent mesh coordinate unit")
    p.add_argument("--out", help="write the domain brick store here")
    p.add_argument("--chunk-bricks", type=int, default=32, dest="chunk_bricks")
    p.set_defaults(func=cmd_voxelize)

    p = sub.add_parser("phantom", help="synthetic geometries")
    p.add_argument("action", choices=["list", "build"])
    p.add_argument("name", nargs="?")
    p.add_argument("--out", default="phantom.npy")
    p.set_defaults(func=cmd_phantom)

    p = sub.add_parser("run", help="run a configuration: the finite-volume solver for a mesh, "
                                   "the lattice-Boltzmann solver for voxels")
    p.add_argument("config")
    p.add_argument("--out")
    p.add_argument("--steps", type=int)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("info", help="describe a run collection, mesh collection or store")
    p.add_argument("target")
    p.set_defaults(func=cmd_info)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
