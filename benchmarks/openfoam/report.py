"""Tables and figures for docs/benchmarks/openfoam.md from the raw results.

    python benchmarks/openfoam/report.py [--coronary-foam <case dir>] \
        [--coronary-zvcfd <run dir> ...]

Reads ``benchmarks/results/openfoam/voxel_*.json`` (voxel cases), an
OpenFOAM coronary case directory (``result.json`` + postProcessing) and
zvCFD coronary run directories (``monitors.json`` + ``patches.csv``).
Writes ``benchmarks/results/openfoam/summary.json``, ``summary.md`` and
figures under ``docs/_static/figures/``.

Convergence is judged the same way for both solvers: the first wall time
after which every outlet's share of the outflow stays within 0.1 (or 1)
percentage point of that solver's final shares.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results" / "openfoam"
FIG = ROOT / "docs" / "_static" / "figures"


def time_to(wall: np.ndarray, err: np.ndarray, tol: float) -> float:
    bad = np.flatnonzero(err > tol)
    i = 0 if not len(bad) else bad[-1] + 1
    return float(wall[min(i, len(wall) - 1)])


# ---------------------------------------------------------------- coronary

def foam_coronary(case: Path) -> dict:
    res = json.loads((case / "result.json").read_text())
    names = res["patches"]
    q = []
    for i in range(len(names)):
        f = sorted(glob.glob(str(case / "postProcessing" / f"flow_{i}" / "*" /
                                 "surfaceFieldValue*.dat")))[-1]
        q.append(np.loadtxt(f, comments="#")[:, 1])
    m = min(len(x) for x in q)
    Q = np.stack([x[:m] for x in q], 1)          # OpenFOAM phi: outward positive
    exe = np.asarray(res["exec_s"][:m])
    out = Q[:, 1:]
    split = out / out.sum(1, keepdims=True)
    err = np.abs(split - split[-1]).max(1)
    p_in = None
    pf = case / "inlet_pressure.txt"
    if pf.exists():
        p_in = float(pf.read_text().split()[0])
    return {"names": [n.split("-to-")[-1] for n in names[1:]], "split": split[-1],
            "inflow": float(-Q[-1, 0]), "outflow": float(out[-1].sum()),
            "wall": exe, "err": err, "iterations": m, "cells": res["cells"],
            "procs": res["procs"], "p_in": p_in,
            "t_1pp": time_to(exe, err, 1e-2), "t_01pp": time_to(exe, err, 1e-3)}


def zvcfd_coronary(run: Path) -> dict:
    mon = json.loads((run / "monitors.json").read_text())
    rows = list(csv.DictReader(open(run / "patches.csv")))
    names = [r["patch"].split("-to-")[-1] for r in rows]
    inlet = [i for i, r in enumerate(rows) if r["kind"] == "velocity"]
    outlets = [i for i in range(len(rows)) if i not in inlet]
    hist = mon["history"]
    wall = np.array([h["wall_s"] for h in hist])
    q = np.array([h["q"] for h in hist])               # into the domain positive
    out = -q[:, outlets]
    split = out / out.sum(1, keepdims=True)
    err = np.abs(split - split[-1]).max(1)
    final = {names[i]: float(rows[i]["split"]) for i in outlets}
    return {"names": [names[i] for i in outlets], "split": split[-1], "split_csv": final,
            "inflow": float(q[-1, inlet].sum()), "outflow": float(out[-1].sum()),
            "p_in": float(rows[inlet[0]]["pressure_pa"]), "wall": wall, "err": err,
            "steps": mon["summary"]["steps"], "solve_s": mon["summary"]["solve_s"],
            "cells": mon["summary"]["fluid_cells"], "voxel_um": mon["summary"]["voxel_um"],
            "mlups": mon["summary"]["mlups"], "converged": mon["summary"]["converged"],
            "imbalance": hist[-1].get("imbalance"),
            "t_1pp": time_to(wall, err, 1e-2), "t_01pp": time_to(wall, err, 1e-3)}


def compare_splits(foam: dict, z: dict) -> dict:
    fz = dict(zip(z["names"], z["split"]))
    a = np.array([fz[n] for n in foam["names"]])
    b = foam["split"]
    d = a - b
    big = b > 0.02
    return {"mean_abs_pp": float(100 * np.abs(d).mean()),
            "max_abs_pp": float(100 * np.abs(d).max()),
            "rel_err_big_median": float(np.median(np.abs(d[big]) / b[big])),
            "r2": float(1 - (d ** 2).sum() / ((b - b.mean()) ** 2).sum()),
            "pairs": list(zip(b.tolist(), a.tolist()))}


# ---------------------------------------------------------------- figures

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})


def fig_splits(foam: dict, zs: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lim = 100 * max(foam["split"].max(), max(z["split"].max() for z in zs)) * 1.08
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--")
    for z, col in zip(zs, (S1, S2, S3)):
        fz = dict(zip(z["names"], z["split"]))
        a = np.array([fz[n] for n in foam["names"]])
        ax.scatter(100 * foam["split"], 100 * a, s=22, color=col, edgecolor=SURFACE,
                   linewidth=0.8, label=f"zvCFD, {z['voxel_um']:g} µm voxels")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("OpenFOAM share of outflow (%), body-fitted 14.8 M-cell mesh")
    ax.set_ylabel("zvCFD share of outflow (%)")
    ax.legend(loc="upper left")
    ax.set_title("HiP-CT coronary tree: 77 outlet flow splits", loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "coronary_splits.png", dpi=160)
    plt.close(fig)


def fig_convergence(foam: dict, zs: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(foam["wall"] / 60, 100 * foam["err"], color=S2, lw=2,
            label=f"OpenFOAM simpleFoam, {foam['procs']} cores")
    for z, col in zip(zs, (S1, S3)):
        ax.plot(z["wall"] / 60, 100 * z["err"], color=col, lw=2,
                label=f"zvCFD {z['voxel_um']:g} µm, RTX A2000")
    ax.axhline(0.1, color=INK2, lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("wall time (min)")
    ax.set_ylabel("max |split − final split| (pp)")
    ax.legend(loc="lower left")
    ax.set_title("Convergence of the outlet flow splits", loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "coronary_convergence.png", dpi=160)
    plt.close(fig)


def fig_voxel(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    y = np.arange(len(rows))
    h = 0.36

    def fmt(v):
        return f"{v:.0f} s" if v >= 10 else f"{v:.1f} s" if v >= 1 else f"{v:.2g} s"

    top = 0.0
    for j, (key, col, lab) in enumerate((("zvcfd", S1, "zvCFD, 1 × RTX A2000"),
                                         ("openfoam", S2, "OpenFOAM v2506, 16 cores"))):
        v = [r[key]["wall_to_0.001"] for r in rows]
        top = max(top, max(v))
        ax.barh(y + (j - 0.5) * (h + 0.04), v, height=h, color=col, label=lab)
        for yy, vv in zip(y, v):
            ax.text(vv * 1.1, yy + (j - 0.5) * (h + 0.04), fmt(vv), va="center",
                    fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(right=top * 4)
    ax.set_yticks(y, [f"{r['case']} ({r['fluid_voxels'] / 1e6:.2g} M cells)" for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("wall time to 0.1 % of the converged flux (s), log scale")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "voxel_time_to_solution.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coronary-foam")
    ap.add_argument("--coronary-zvcfd", nargs="*", default=[])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    md = []
    rows = []
    for name in ("duct16", "duct32", "pipe6", "pipe12", "pipe24", "porous", "vessels"):
        f = RES / f"voxel_{name}.json"
        if f.exists():
            r = json.loads(f.read_text())
            if "zvcfd" in r and "openfoam" in r:
                rows.append(r)
    if rows:
        md += ["| Case | Fluid cells | zvCFD to 0.1 % | OpenFOAM to 0.1 % | Speed-up | "
               "Flux zvCFD / OpenFOAM | vs analytic (zvCFD, OpenFOAM) |",
               "|---|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            z, o = r["zvcfd"], r["openfoam"]
            pois = (f"{z['flux_final'] / r['flux_poiseuille']:.3f}, "
                    f"{o['flux_final'] / r['flux_poiseuille']:.3f}") if "flux_poiseuille" in r \
                else "—"
            md.append(f"| {r['case']} | {r['fluid_voxels']:,} | {z['wall_to_0.001']:.3g} s | "
                      f"{o['wall_to_0.001']:.3g} s | "
                      f"{o['wall_to_0.001'] / max(z['wall_to_0.001'], 1e-9):.0f}× | "
                      f"{z['flux_final'] / o['flux_final']:.4f} | {pois} |")
        summary["voxel"] = [{k: v for k, v in r.items() if k not in ("zvcfd", "openfoam")}
                            | {"zvcfd": {k: v for k, v in r["zvcfd"].items() if k != "history"},
                               "openfoam": {k: v for k, v in r["openfoam"].items()
                                            if k != "history"}} for r in rows]
        fig_voxel(rows)
    if args.coronary_foam and args.coronary_zvcfd:
        foam = foam_coronary(Path(args.coronary_foam))
        zs = [zvcfd_coronary(Path(p)) for p in args.coronary_zvcfd]
        md += ["", "| Solver | Cells | Hardware | Inlet pressure | Time to 1 pp | "
               "Time to 0.1 pp | Splits vs OpenFOAM (mean / max abs, pp) |",
               "|---|---:|---|---:|---:|---:|---|"]
        md.append(f"| OpenFOAM simpleFoam | {foam['cells']:,} | {foam['procs']} cores | "
                  f"{foam['p_in'] if foam['p_in'] is not None else float('nan'):.1f} Pa | "
                  f"{foam['t_1pp'] / 60:.1f} min | {foam['t_01pp'] / 60:.1f} min | — |")
        summary["coronary"] = {"openfoam": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                            for k, v in foam.items() if k not in ("wall", "err")}}
        for z in zs:
            c = compare_splits(foam, z)
            md.append(f"| zvCFD {z['voxel_um']:g} µm | {z['cells']:,} | RTX A2000 | "
                      f"{z['p_in']:.1f} Pa | {z['t_1pp'] / 60:.1f} min | "
                      f"{z['t_01pp'] / 60:.1f} min | {c['mean_abs_pp']:.2f} / "
                      f"{c['max_abs_pp']:.2f} |")
            summary["coronary"][f"zvcfd_{z['voxel_um']:g}um"] = {
                **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in z.items()
                   if k not in ("wall", "err", "split_csv")}, "vs_openfoam": c}
        fig_splits(foam, zs)
        fig_convergence(foam, zs)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (RES / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
