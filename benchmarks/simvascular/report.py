"""Markdown tables for docs/validation/simvascular.md, from summary.json.

    python benchmarks/simvascular/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "simvascular"


def label(run: dict) -> str:
    if run.get("code") == "fv":
        s = "finite volume, SimVascular's mesh"
        if run.get("backflow"):
            s += f", backflow {run['backflow']:g}"
        return s
    s = f"zvCFD {run['voxel_um']:g} µm"
    if run["inlet"] != "sv":
        s += ", parabolic inlet"
    if abs(run.get("u_lat", 0.05) - 0.05) > 1e-9:
        s += f", u_lat {run['u_lat']:g}"
    return s


def order(runs: dict) -> list:
    return sorted(runs, key=lambda k: (runs[k]["run"].get("code") == "fv",
                                       runs[k]["run"]["inlet"] != "sv",
                                       -(runs[k]["run"]["voxel_um"] or 0),
                                       runs[k]["run"].get("u_lat") or 0.05))


def main():
    s = json.loads((RES / "summary.json").read_text())
    runs = s["runs"]
    keys = order(runs)
    print("### Runs\n")
    print("| Run | Fluid voxels | τ | dt (µs) | Steps per cycle | Solve time per cycle | MLUPS |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        r = runs[k]["run"]
        per = r["solve_s"] / r["cycles"]
        if r.get("code") == "fv":
            print(f"| {label(r)} | {r['nodes'] / 1e6:.2f} M nodes | — | {r['dt_s'] * 1e6:.0f} | "
                  f"{int(round(1 / r['dt_s'])):,} | {per / 60:.0f} min | — |")
            continue
        print(f"| {label(r)} | {r['fluid_voxels'] / 1e6:.2f} M | {r['tau']:.4f} | "
              f"{r['dt_s'] * 1e6:.2f} | {int(round(1 / r['dt_s'])):,} | {per / 60:.0f} min | "
              f"{r['mlups']:.0f} |")
    print("\n### Against SimVascular\n")
    print("| Run | Outlet mean flow, mean / max \\|diff\\| | Outlet waveform, median | "
          "Split, mean \\|diff\\| LCA / RCA | Pressure drop LCA / RCA | Velocity, sections | "
          "Velocity, volume | TAWSS (r) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        v = runs[k]
        o, p, w, f = v["outlets"], v.get("pressure", {}), v.get("wall", {}), v.get("fields", {})
        vol = np.mean([x["velocity_rel_l2_pct"] for x in f.values()]) if f else np.nan
        print(f"| {label(v['run'])} | {o['mean_flow_err_pct_mean_abs']:.1f} / "
              f"{o['mean_flow_err_pct_max_abs']:.1f} % | {o['waveform_err_pct_median']:.1f} % | "
              f"{o['trees']['LCA']['split_diff_pp_mean']:.2f} / "
              f"{o['trees']['RCA']['split_diff_pp_mean']:.2f} pp | "
              f"{p['LCA_inlet']['mean_err_pct']:+.1f} / {p['RCA_inlet']['mean_err_pct']:+.1f} % | "
              f"{w.get('section_velocity_rel_l2_pct', np.nan):.1f} % | {vol:.1f} % | "
              f"{w.get('tawss_rel_l2_pct', np.nan):.1f} % ({w.get('tawss_corr', np.nan):.3f}) |")
    sc = s.get("self_convergence", {})
    if sc:
        print(f"\n### zvCFD against its finest run ({sc['finest']})\n")
        print("| Run | Outlet mean flow, mean / max \\|diff\\| | Inlet pressure LCA / RCA "
              "| TAWSS |")
        print("|---|---:|---:|---:|")
        for k, v in sc.items():
            if k == "finest":
                continue
            ip = v.get("inlet_pressure_diff_pct", {})
            print(f"| {label(runs[k]['run'])} | {v['outlet_mean_flow_diff_pct_mean_abs']:.1f} / "
                  f"{v['outlet_mean_flow_diff_pct_max_abs']:.1f} % | "
                  f"{ip.get('LCA_inlet', np.nan):+.1f} / {ip.get('RCA_inlet', np.nan):+.1f} % | "
                  f"{v.get('tawss_rel_l2_pct', np.nan):.1f} % |")
    before_path = RES / "summary_neem.json"
    if before_path.exists():
        before = json.loads(before_path.read_text())["runs"]
        print("\n### Before and after the outlet fix\n")
        print("| Run | Outlet mean flow, mean / max \\|diff\\| | Split LCA / RCA | "
              "Pressure drop LCA / RCA | TAWSS |")
        print("|---|---|---|---|---|")
        for k in keys:
            if k not in before or runs[k]["run"]["inlet"] != "sv":
                continue
            for tag, v in (("before", before[k]), ("after", runs[k])):
                o, p, w = v["outlets"], v["pressure"], v.get("wall", {})
                print(f"| {label(v['run'])}, {tag} | {o['mean_flow_err_pct_mean_abs']:.1f} / "
                      f"{o['mean_flow_err_pct_max_abs']:.1f} % | "
                      f"{o['trees']['LCA']['split_diff_pp_mean']:.2f} / "
                      f"{o['trees']['RCA']['split_diff_pp_mean']:.2f} pp | "
                      f"{p['LCA_inlet']['mean_err_pct']:+.1f} / "
                      f"{p['RCA_inlet']['mean_err_pct']:+.1f} % | "
                      f"{w.get('tawss_rel_l2_pct', np.nan):.1f} % |")
    best = sc.get("finest") or keys[0]
    print(f"\n### Per outlet ({best})\n")
    per = runs[best].get("per_outlet")
    if per:
        print("| Outlet | SimVascular (mL/s) | zvCFD (mL/s) | Difference |")
        print("|---|---:|---:|---:|")
        for n, x in per.items():
            print(f"| {n} | {x['sv_mL_s']:.3f} | {x['zv_mL_s']:.3f} | {x['err_pct']:+.1f} % |")
    sv = s["sv"]
    if "cut_vs_outlets_rel_l2_pct" in sv:
        print(f"\nSimVascular flow through the cuts vs the sum of its outlet flows: "
              f"{sv['cut_vs_outlets_rel_l2_pct']}")


if __name__ == "__main__":
    main()
