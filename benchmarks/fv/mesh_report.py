"""Gate A of the finite-volume plan: read a Fluent mesh, build its dual, measure its couplings.

    python benchmarks/fv/mesh_report.py ["/home/andrew/Downloads/mesh 1.msh"] [--out NAME]

Writes ``benchmarks/results/fv/<NAME>.json`` with element counts, reader
timings and peak memory, the dual-geometry checks (node volumes against
element volumes, control-volume closure), mesh quality, blocks per row of
the coupled system and the GPU memory that implies (docs/feasibility/fv_plan.md,
"Estimates").
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from zvcfd.fv.geometry import dual_geometry
from zvcfd.fv.pattern import coupling_stats, memory_estimate, node_graph
from zvcfd.mesh import read_fluent_mesh

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", nargs="?", default="/home/andrew/Downloads/mesh 1.msh")
    ap.add_argument("--out", default="coronary_mesh")
    ap.add_argument("--unit", default="mm")
    args = ap.parse_args()
    out = {"mesh": args.mesh}
    t = time.time()
    mesh = read_fluent_mesh(args.mesh, unit=args.unit, log=None)
    out["read_s"] = time.time() - t
    out["reader"] = mesh.meta
    out["counts"] = mesh.counts()
    out["nodes"] = mesh.n_nodes
    out["zones"] = {k: sum(1 for z in mesh.zones.values() if z.kind == k)
                    for k in sorted({z.kind for z in mesh.zones.values()})}
    print("read", out["counts"], f"{out['read_s']:.1f} s", flush=True)

    t = time.time()
    fx = mesh.faces()
    out["conforming"] = {"faces": int(len(fx["c0"])), "boundary_faces": int((fx["c1"] < 0).sum()),
                         "zone_faces": int(sum(z.n_faces for z in mesh.zones.values())),
                         "seconds": time.time() - t}
    del fx
    print("faces", out["conforming"], flush=True)

    t = time.time()
    out["quality"] = mesh.quality()
    out["quality_s"] = time.time() - t
    g = dual_geometry(mesh, keep_areas=False)
    out["dual"] = g.report
    print("dual", g.report, flush=True)

    t = time.time()
    indptr, _ = node_graph(mesh)
    out["couplings"] = coupling_stats(indptr)
    out["couplings"]["seconds"] = time.time() - t
    print("couplings", out["couplings"], flush=True)

    from zvcfd.fv.geometry import topology

    n_ip = sum(len(e) * topology(k).n_ip for k, e in mesh.elements.items())
    blocks = out["couplings"]["blocks"]
    out["integration_points"] = int(n_ip)
    out["memory_gb"] = memory_estimate(blocks, mesh.n_nodes, n_ip)
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    dest = ROOT / "benchmarks" / "results" / "fv" / f"{args.out}.json"
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["memory_gb"], indent=1), "\n->", dest)


if __name__ == "__main__":
    main()
