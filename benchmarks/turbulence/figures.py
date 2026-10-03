"""Figures and the summary table for docs/validation/turbulence.md.

    PYTHONPATH=. python benchmarks/turbulence/figures.py

Reads benchmarks/results/turbulence/ (flat_plate_<ni>.npz, naca0012_<ni>_a<alpha>.npz)
and the experimental data in $TMR (Ladson 1988; Gregory & O'Reilly 1970). The
boundary-layer reference for the flat plate (plate_bl.py) is computed once and
cached. Grid levels use the single-hue blue ramp, light to dark from coarse to
fine; references are ink, experiments open markers (dataviz reference palette).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import meshes as M  # noqa: E402
import plate_bl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "_static" / "figures"
RES = ROOT / "benchmarks" / "results" / "turbulence"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LEVELS = {35: "#9ec5f4", 69: "#5598e7", 137: "#1c5cab", 273: "#0d366b",
          113: "#9ec5f4", 225: "#5598e7", 449: "#1c5cab"}
DIMS = {35: "35 × 25", 69: "69 × 49", 137: "137 × 97", 273: "273 × 193", 113: "113 × 33",
        225: "225 × 65", 449: "449 × 129"}
NU = 1.0 / 5.0e6


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "font.size": 9.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})


def tecplot_zones(path) -> dict:
    """``{title: (n, k) array}`` from a Tecplot-style ASCII file (the Resource's data)."""
    zones, cur, rows = {}, None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("variables"):
            continue
        if line.lower().startswith("zone"):
            if cur is not None:
                zones[cur] = np.array(rows)
            m = re.search(r't="([^"]*)"', line, re.IGNORECASE)
            cur, rows = (m.group(1) if m else f"zone{len(zones)}"), []
            continue
        rows.append([float(v) for v in line.split()])
    if cur is not None:
        zones[cur] = np.array(rows)
    return zones


def boundary_layer_reference():
    path = RES / "plate_bl.npz"
    if not path.exists():
        res, y, u, k, phi = plate_bl.march()
        np.savez(path, res=res, y=y, u=u, k=k, phi=phi)
    return np.load(path)


def nasa_flat_plate():
    """CFL3D's and FUN3D's k-kL-MEAH2015 flat plate (the Resource's K-kL-MEAH2015m page):
    c_f(0.97) on each grid, c_f along the plate, and u, mu_t/mu at x = 0.97 (finest grid)."""
    d = M.TMR / "kkl_reference"
    if not d.exists():
        return None
    return {"conv": tecplot_zones(d / "cf_convergence_kklmeah2015.dat"),
            "cf": tecplot_zones(d / "cf_plate_kklmeah2015.dat"),
            "u": tecplot_zones(d / "u-kklmeah2015.dat"),
            "turb": tecplot_zones(d / "k-kl-mut-kklmeah2015.dat")}


def flat_plate(grids=(35, 69, 137, 273)):
    runs = {g: np.load(RES / f"flat_plate_{g}.npz") for g in grids
            if (RES / f"flat_plate_{g}.npz").exists()}
    if not runs:
        return {}
    ref = boundary_layer_reference()
    nasa = nasa_flat_plate()
    xr, cfr, _ = ref["res"].T
    codes = (("CFL3D", INK, (0, (4, 3))), ("FUN3D", MUTED, (0, (1, 1.5))))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for g, r in runs.items():
        a.plot(r["x"], r["cf"] * 1e3, color=LEVELS[g], lw=1.8, label=f"zvCFD, {DIMS[g]}")
    if nasa:
        for name, col, ls in codes:
            z = nasa["cf"][name]
            a.plot(z[:, 0], z[:, 1] * 1e3, color=col, lw=1.1, ls=ls, zorder=5,
                   label=f"{name}, 545 × 385")
    a.plot(xr, cfr * 1e3, color=SERIES[1], lw=1.1, ls=(0, (6, 2)), zorder=4,
           label="boundary-layer marching")
    a.set_xlim(0, 2)
    a.set_ylim(1.5, 6)
    a.set_xlabel("x (Re = 5 million per unit length)")
    a.set_ylabel("skin friction  c_f × 10³")
    a.set_title("Skin friction along the plate", loc="left", fontsize=10)
    a.legend(fontsize=7.5)
    g = max(runs)
    r = runs[g]
    ut = np.sqrt(float(r["cf097"]) / 2)
    yp = np.logspace(-1, 4, 200)
    b.plot(yp, yp, color=GRID, lw=1, ls=":")
    b.plot(yp, np.log(yp) / 0.41 + 5.0, color=GRID, lw=1, ls=":")
    b.annotate("u⁺ = y⁺", (3, 3), xytext=(4, -10), textcoords="offset points", color=INK2,
               fontsize=8)
    b.annotate("u⁺ = ln(y⁺)/0.41 + 5", (100, np.log(100) / 0.41 + 5), xytext=(6, -26),
               textcoords="offset points", color=INK2, fontsize=8)
    b.plot(r["y"][1:] * ut / NU, r["u"][1:] / ut, "o-", color=LEVELS[g], ms=3.5, lw=1.4,
           label=f"zvCFD, {DIMS[g]}")
    if nasa:
        cfn = {n: float(z[0, 3]) for n, z in nasa["conv"].items()}      # finest grid
        for name, col, ls in codes:
            z = nasa["u"][name][1:]
            un = np.sqrt(cfn[name] / 2)
            b.plot(z[:, 0] * un / NU, z[:, 1] / un, color=col, lw=1.1, ls=ls, zorder=5,
                   label=f"{name}, 545 × 385")
    b.set_xscale("log")
    b.set_xlim(0.1, 2e4)
    b.set_ylim(0, 32)
    b.set_xlabel("y⁺")
    b.set_ylabel("u⁺")
    b.set_title("Velocity at x = 0.97, wall units", loc="left", fontsize=10)
    b.legend(fontsize=7.5, loc="upper left")
    for gg, rr in runs.items():
        c.plot(rr["mut"], rr["y"] * 1e3, color=LEVELS[gg], lw=1.8, label=f"zvCFD, {DIMS[gg]}")
    if nasa:
        for name, col, ls in codes:
            z = nasa["turb"][name]
            c.plot(z[:, 3], z[:, 0] * 1e3, color=col, lw=1.1, ls=ls, zorder=5,
                   label=f"{name}, 545 × 385")
    c.set_ylim(0, 40)
    c.set_xlim(0, 230)
    c.set_xlabel("eddy viscosity  μ_t/μ")
    c.set_ylabel("y × 10³")
    c.set_title("Eddy viscosity at x = 0.97", loc="left", fontsize=10)
    c.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "turbulence_flat_plate.png", dpi=160)
    plt.close(fig)
    out = {g: {k: (float(r[k]) if r[k].ndim == 0 else None) for k in
               ("nodes", "iterations", "cf097", "re_theta", "cf_ks", "wall_s")}
           for g, r in runs.items()} | {"bl_cf097": float(np.interp(0.97, xr, cfr))}
    if nasa:
        out["nasa_cf097"] = {n: {int(row[0]): float(row[3]) for row in z}
                             for n, z in nasa["conv"].items()}
        for name in ("CFL3D", "FUN3D"):
            z = nasa["turb"][name]
            out[f"nasa_mut_max_{name}"] = float(z[:, 3].max())
        for gg, rr in runs.items():
            out[gg]["mut_max_097"] = float(rr["mut"].max())
    return out


# k-kL-MEAH2015m on the Resource's 897 x 257 grid, M = 0.15 (its NACA 0012 K-kL-MEAH2015m page)
NASA_NACA = {"CFL3D": {0: (0.0, 0.00831), 10: (1.0700, 0.01341), 15: (1.4980, 0.02428)},
             "FUN3D": {0: (0.0, 0.00827), 10: (1.0767, 0.01358), 15: (1.5048, 0.02473)},
             "TAU": {0: (0.0, 0.00836), 10: (1.0772, 0.01358), 15: (1.5040, 0.02477)}}


def naca0012(grids=(113, 225, 449), alphas=(0, 10, 15)):
    runs = {(g, a): np.load(RES / f"naca0012_{g}_a{a:g}.npz") for g in grids for a in alphas
            if (RES / f"naca0012_{g}_a{a:g}.npz").exists()}
    if not runs:
        return {}
    greg = tecplot_zones(M.TMR / "CP_Gregory_expdata.dat")
    ref = M.TMR / "kkl_reference"
    cfl_cp = tecplot_zones(ref / "n0012cp_cfl3d_kkl.dat") if ref.exists() else {}
    cfl_cf = tecplot_zones(ref / "n0012cf_cfl3d_kkl.dat") if ref.exists() else {}
    lad = tecplot_zones(M.TMR / "CLCD_Ladson_expdata.dat")
    gcl = tecplot_zones(M.TMR / "CL_Gregory_expdata.dat")
    finest = max(g for g, _ in runs)
    fig, axes = plt.subplots(1, len(alphas), figsize=(11, 3.7), sharey=True)
    for ax, al in zip(np.atleast_1d(axes), alphas):
        e = greg.get(f"alpha={al:g}")
        if e is not None:
            ax.plot(e[:, 0], e[:, 1], "o", mfc="none", mec=INK2, ms=4,
                    label="Gregory & O'Reilly (upper surface)")
        c = cfl_cp.get(f"alpha={al:g}")
        if c is not None:
            ax.plot(c[:, 0], c[:, 1], color=INK, lw=1.0, ls=(0, (4, 3)), zorder=5,
                    label="CFL3D, same model, 897 × 257")
        for g in grids:
            if (g, al) not in runs:
                continue
            r = runs[(g, al)]
            o = np.argsort(r["x"])
            up = r["y"][o] >= 0
            for sel, ls in ((up, "-"), (~up, "--")):
                ax.plot(r["x"][o][sel], r["cp"][o][sel], ls, color=LEVELS[g], lw=1.5,
                        label=f"zvCFD, {DIMS[g]}" if ls == "-" else None)
        ax.invert_yaxis() if al == alphas[0] else None
        ax.set_title(f"α = {al}°", loc="left", fontsize=10)
        ax.set_xlabel("x/c")
    np.atleast_1d(axes)[0].set_ylabel("pressure coefficient  Cp")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "turbulence_naca0012_cp.png", dpi=160)
    plt.close(fig)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, mk in (("80 grit", "o"), ("120 grit", "s"), ("180 grit", "^")):
        z = lad[name]
        a.plot(z[:, 0], z[:, 1], mk, mfc="none", mec=INK2, ms=4, label=f"Ladson, {name}")
        b.plot(z[:, 0], z[:, 2] * 1e4, mk, mfc="none", mec=INK2, ms=4, label=f"Ladson, {name}")
    z = next(iter(gcl.values()))
    a.plot(z[:, 0], z[:, 1], "x", color=MUTED, ms=4, label="Gregory & O'Reilly")
    for name, mk in (("CFL3D", "D"), ("FUN3D", "s"), ("TAU", "v")):
        al_ = sorted(NASA_NACA[name])
        a.plot(al_, [NASA_NACA[name][x][0] for x in al_], mk, color=INK, ms=4.5, zorder=5,
               label=f"{name}, same model, 897 × 257")
        b.plot(al_, [NASA_NACA[name][x][1] * 1e4 for x in al_], mk, color=INK, ms=4.5,
               zorder=5, label=f"{name}, same model")
    for g in grids:
        pts = sorted((al, float(runs[(g, al)]["CL"]), float(runs[(g, al)]["CD"]))
                     for al in alphas if (g, al) in runs)
        if not pts:
            continue
        al_, cl_, cd_ = np.array(pts).T
        conv = np.array([bool(runs[(g, al)]["converged"]) for al in al_])
        a.plot(al_, cl_, "-", color=LEVELS[g], lw=1.5, label=f"zvCFD, {DIMS[g]}")
        b.plot(al_, cd_ * 1e4, "-", color=LEVELS[g], lw=1.5, label=f"zvCFD, {DIMS[g]}")
        for ax_, v in ((a, cl_), (b, cd_ * 1e4)):
            ax_.plot(al_[conv], v[conv], "o", color=LEVELS[g], ms=5)
            ax_.plot(al_[~conv], v[~conv], "o", mfc=SURFACE, mec=LEVELS[g], ms=5)
    a.set_xlim(-1, 20)
    a.set_xlabel("angle of attack α (°)")
    a.set_ylabel("lift coefficient  CL")
    a.set_title("Lift", loc="left", fontsize=10)
    a.legend(fontsize=7.5, loc="upper left")
    a.annotate("open marker: not converged (the band of the last 100 iterations is in the JSON)",
               (0.02, 0.02), xycoords="axes fraction", fontsize=7.5, color=INK2)
    b.set_xlim(-1, 17)
    b.set_ylim(0, 560)
    b.set_xlabel("angle of attack α (°)")
    b.set_ylabel("drag coefficient  CD × 10⁴")
    b.set_title("Drag", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "turbulence_naca0012_forces.png", dpi=160)
    plt.close(fig)
    if cfl_cf:
        fig, axes = plt.subplots(1, len(alphas), figsize=(11, 3.5), sharey=True)
        for ax, al in zip(np.atleast_1d(axes), alphas):
            c = cfl_cf.get(f"alpha={al:g}, upper surface")
            if c is not None:
                ax.plot(c[:, 0], c[:, 1] * 1e3, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=5,
                        label="CFL3D, same model, 897 × 257")
            for g in grids:
                if (g, al) not in runs:
                    continue
                r = runs[(g, al)]
                up = r["y"] >= 0
                o = np.argsort(r["x"][up])
                ax.plot(r["x"][up][o], r["cf"][up][o] * 1e3, color=LEVELS[g], lw=1.5,
                        label=f"zvCFD, {DIMS[g]}")
            ax.set_title(f"α = {al}°, upper surface", loc="left", fontsize=10)
            ax.set_xlabel("x/c")
            ax.set_ylim(-2, 12)
        np.atleast_1d(axes)[0].set_ylabel("skin friction  Cf × 10³")
        np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG / "turbulence_naca0012_cf.png", dpi=160)
        plt.close(fig)
    out = {"nasa": NASA_NACA}
    for (g, al), r in runs.items():
        out[f"{g}_a{al}"] = {k: float(r[k]) for k in ("CL", "CD", "CL_surface", "CD_surface",
                                                       "CD_pressure", "CD_viscous", "nodes",
                                                       "iterations", "wall_s")}
    out["finest"] = finest
    return out


if __name__ == "__main__":
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    summary = {"flat_plate": flat_plate(), "naca0012": naca0012()}
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
