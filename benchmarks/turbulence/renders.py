"""Renderings of the turbulent solutions for docs/validation/turbulence.md.

    PYTHONPATH=. python benchmarks/turbulence/renders.py

Reads the fields that flat_plate.py and naca0012.py save on the z = 0 plane
(benchmarks/results/turbulence/fields_*.npz) and maps them back onto the NASA
Resource's structured grids, so the contours follow the grid lines; streamlines
are traced on a uniform grid interpolated from the nodes. Uses the finest grid with
saved fields. Magnitudes take the single-hue blue ramp, pressure the blue-red
diverging pair with a grey midpoint at Cp = 0 (dataviz reference palette).

- ``turbulence_render_flat_plate.png``: velocity and eddy viscosity in the
  boundary layer along the whole plate;
- ``turbulence_render_naca0012.png``: speed with streamlines, pressure, and eddy
  viscosity around the NACA 0012 at 10 degrees;
- ``turbulence_render_naca0012_te.png``: the trailing edge at 10 and 15 degrees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import figures as fg  # noqa: E402
import meshes as M  # noqa: E402

FIG, RES = fg.FIG, fg.RES
BLUE_RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab",
             "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("blue_red", ["#1c5cab", "#6da7ec", "#f0efec",
                                                     "#ec8f8e", "#c23535"])
SOLID = "#c3c2b7"                                       # the airfoil and the plate


def structured(fields: Path, grid: Path, alpha: float = 0.0):
    """The saved node fields on the Resource's structured grid ``(jdim, idim)``, turned by
    ``-alpha`` degrees into wind axes (the freestream along x)."""
    f = dict(np.load(fields))
    x2, y2 = M.read_plot3d_2d(grid)
    tree = cKDTree(np.c_[f["x"], f["y"]])
    d, idx = tree.query(np.c_[x2.ravel(), y2.ravel()])
    if d.max() > 1e-9:
        raise ValueError(f"{fields.name}: grid points {d.max():.1e} from the nearest node")
    c, s_ = np.cos(np.radians(alpha)), np.sin(np.radians(alpha))
    for a, b in (("x", "y"), ("u", "v")):
        f[a], f[b] = c * f[a] + s_ * f[b], -s_ * f[a] + c * f[b]
    out = {k: f[k][idx].reshape(x2.shape) for k in ("u", "v", "p", "k", "kl", "mut")}
    out["x"], out["y"] = c * x2 + s_ * y2, -s_ * x2 + c * y2
    out["x_body"] = x2
    out["nodes"] = f
    return out


def finest(pattern: str, levels) -> tuple[int, Path] | None:
    for g in sorted(levels, reverse=True):
        p = RES / pattern.format(g=g)
        if p.exists():
            return g, p
    return None


def _bar(fig, im, ax, label, fmt=None, ticks=None):
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.035, format=fmt, ticks=ticks)
    cb.set_label(label, color=fg.INK2)
    cb.outline.set_visible(False)
    return cb


def flat_plate():
    pick = finest("fields_flat_plate_{g}.npz", (35, 69, 137, 273, 545))
    if pick is None:
        return
    g, path = pick
    F = structured(path, M.TMR / "FlatPlate" / "Grids" / M.FLAT[g])
    X, Y = F["x"], F["y"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True)
    for ax in (a, b):
        ax.grid(False)
        ax.set_facecolor(fg.SURFACE)
    im = a.contourf(X, Y * 1e3, 1.0 - F["u"], levels=np.linspace(0, 1.0, 21), cmap=SEQ,
                    extend="min")
    a.contour(X, Y * 1e3, F["u"], levels=[0.99], colors=[fg.INK], linewidths=0.9)
    xl = 1.2
    j = np.argmin(np.abs(X[0] - xl))
    yd = np.interp(0.99, F["u"][:, j], Y[:, j]) * 1e3
    a.annotate("edge of the layer, u = 0.99 U∞", (xl, yd), xytext=(-10, 12),
               textcoords="offset points", color=fg.INK, fontsize=8.5, ha="right",
               arrowprops={"arrowstyle": "-", "color": fg.INK2, "lw": 0.6})
    _bar(fig, im, a, "1 − u / U∞", ticks=np.linspace(0, 1, 6))
    a.set_title(f"Velocity deficit ({fg.DIMS[g]} grid)", loc="left", fontsize=10)
    top = float(np.ceil(np.nanmax(F["mut"]) / 50) * 50)
    im = b.contourf(X, Y * 1e3, F["mut"], levels=np.linspace(0, top, 21), cmap=SEQ)
    _bar(fig, im, b, "μ_t / μ", fmt="%.0f", ticks=np.arange(0, top + 1, 50))
    b.set_title("Eddy viscosity", loc="left", fontsize=10)
    for ax in (a, b):
        ax.set_ylim(0, 40)
        ax.set_xlim(-0.1, 2.0)
        ax.axhspan(-1, 0, color=SOLID)
        ax.plot([0, 2], [0, 0], color=fg.INK2, lw=2.5, solid_capstyle="butt")
        ax.set_ylabel("y × 10³")
    b.set_xlabel("x (placeholder for the layout)")
    fig.tight_layout()
    bb = b.get_window_extent()
    stretch = (bb.height / 0.040) / (bb.width / 2.1)       # display units per unit y over x
    b.set_xlabel("x (the plate from x = 0; Re = 5 million per unit length; vertical scale "
                 f"exaggerated {stretch:.0f} times)")
    fig.savefig(FIG / "turbulence_render_flat_plate.png", dpi=170)
    plt.close(fig)


def _airfoil(F):
    """The airfoil's outline (the j = 0 grid line between the trailing-edge points)."""
    on = F["x_body"][0] <= 1.0 + 1e-9                    # body axes: the wake cut is x > 1
    return np.c_[F["x"][0][on], F["y"][0][on]]


def _streamlines(ax, F, window, density=0.9, n=320):
    (x0, x1), (y0, y1) = window
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, int(n * (y1 - y0) / (x1 - x0)))
    gx, gy = np.meshgrid(xs, ys)
    f = F["nodes"]
    near = (f["x"] > x0 - 0.2) & (f["x"] < x1 + 0.2) & (f["y"] > y0 - 0.2) & (f["y"] < y1 + 0.2)
    pts = np.c_[f["x"][near], f["y"][near]]
    gu = griddata(pts, f["u"][near], (gx, gy), method="linear")
    gv = griddata(pts, f["v"][near], (gx, gy), method="linear")
    inside = MPath(_airfoil(F)).contains_points(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    gu[inside] = np.nan
    gv[inside] = np.nan
    sp = ax.streamplot(xs, ys, gu, gv, color=fg.INK, linewidth=0.45, density=density,
                       arrowstyle="-")
    sp.lines.set_alpha(0.55)


def _frame(ax, F, window, equal=True):
    ax.grid(False)
    ax.fill(*_airfoil(F).T, color=SOLID, zorder=3)
    ax.plot(*_airfoil(F).T, color=fg.INK2, lw=0.8, zorder=4)
    (x0, x1), (y0, y1) = window
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if equal:
        ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")


def naca0012(alpha=10):
    pick = finest(f"fields_naca0012_{{g}}_a{alpha}.npz", (113, 225, 449))
    if pick is None:
        return
    g, path = pick
    F = structured(path, M.TMR / "NACA0012_grids" / M.NACA[g], alpha)
    X, Y = F["x"], F["y"]
    speed = np.hypot(F["u"], F["v"])
    p_inf = float(np.median(F["p"][-1]))                     # the C boundary, 500 chords out
    cp = (F["p"] - p_inf) / 0.5
    fig = plt.figure(figsize=(11.5, 6.9), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1])
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    near = ((-0.35, 1.45), (-0.6, 0.6))
    im = a.contourf(X, Y, speed, levels=np.linspace(0, 1.8, 19), cmap=SEQ, extend="max")
    _streamlines(a, F, near)
    _frame(a, F, near)
    _bar(fig, im, a, "|u| / U∞")
    a.annotate("", xy=(-0.12, -0.45), xytext=(-0.32, -0.45),
               arrowprops={"arrowstyle": "-|>", "color": fg.INK, "lw": 1.2})
    a.annotate("U∞", (-0.22, -0.45), xytext=(0, 5), textcoords="offset points",
               ha="center", fontsize=9, color=fg.INK)
    a.set_title(f"Speed and streamlines, α = {alpha}°", loc="left", fontsize=10)
    lo = float(np.floor(np.nanmin(cp)))
    im = b.contourf(X, Y, cp, levels=np.linspace(lo, 1.0, 25), cmap=DIV.reversed(),
                    norm=TwoSlopeNorm(0.0, lo, 1.0))
    _frame(b, F, near)
    _bar(fig, im, b, "Cp", ticks=list(range(int(lo), 0)) + [0, 0.5, 1])
    b.set_title("Pressure coefficient", loc="left", fontsize=10)
    wide = ((-0.15, 3.0), (-0.35, 0.3))
    top = float(np.ceil(np.nanmax(F["mut"]) / 100) * 100)
    im = c.contourf(X, Y, F["mut"], levels=np.linspace(0, top, 21), cmap=SEQ)
    _frame(c, F, wide)
    _bar(fig, im, c, "μ_t / μ", fmt="%.0f",
         ticks=MaxNLocator(6).tick_values(0, top))
    c.set_title("Eddy viscosity: the boundary layers and the wake", loc="left", fontsize=10)
    fig.suptitle(f"NACA 0012 at α = {alpha}°, Re = 6 million, {fg.DIMS[g]} grid, k-kL "
                 "(wind axes: the freestream along x)", x=0.01, ha="left", fontsize=11,
                 color=fg.INK)
    fig.savefig(FIG / "turbulence_render_naca0012.png", dpi=170)
    plt.close(fig)


def trailing_edge(alphas=(10, 15)):
    picks = {al: finest(f"fields_naca0012_{{g}}_a{al}.npz", (113, 225, 449)) for al in alphas}
    picks = {al: p for al, p in picks.items() if p is not None}
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(5.8 * len(picks), 4.4), squeeze=False,
                             layout="constrained")
    for ax, (al, (g, path)) in zip(axes[0], picks.items()):
        F = structured(path, M.TMR / "NACA0012_grids" / M.NACA[g], al)
        a = np.radians(al)
        te = np.array([np.cos(a), -np.sin(a)])              # the trailing edge, wind axes
        window = ((te[0] - 0.28, te[0] + 0.12), (te[1] - 0.12, te[1] + 0.16))
        speed = np.hypot(F["u"], F["v"])
        im = ax.contourf(F["x"], F["y"], speed, levels=np.linspace(0, 1.2, 13), cmap=SEQ,
                         extend="max")
        ax.contour(F["x"], F["y"], F["u"], levels=[0.0], colors=[fg.SERIES[1]],
                   linewidths=1.3)
        _streamlines(ax, F, window, density=1.1, n=360)
        _frame(ax, F, window)
        (x0, x1), (y0, y1) = window
        xa, ya = x0 + 0.02 * (x1 - x0), y0 + 0.08 * (y1 - y0)
        ax.annotate("", xy=(xa + 0.1 * (x1 - x0), ya), xytext=(xa, ya),
                    arrowprops={"arrowstyle": "-|>", "color": fg.INK, "lw": 1.2})
        ax.annotate("U∞", (xa + 0.05 * (x1 - x0), ya), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9, color=fg.INK)
        sel = (X := F["x"]) > x0
        sel &= (X < x1) & (F["y"] > y0) & (F["y"] < y1)
        state = "reversed flow, outlined" if (F["u"][sel] < 0).any() else "no reversed flow"
        ax.set_title(f"α = {al}° ({fg.DIMS[g]} grid): {state}", loc="left", fontsize=10)
        _bar(fig, im, ax, "|u| / U∞")
    fig.suptitle("The trailing edge: speed and streamlines (wind axes)", x=0.01, ha="left",
                 fontsize=11, color=fg.INK)
    fig.savefig(FIG / "turbulence_render_naca0012_te.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fg.style()
    plt.rcParams.update({"axes.grid": False})
    flat_plate()
    naca0012(10)
    trailing_edge()
