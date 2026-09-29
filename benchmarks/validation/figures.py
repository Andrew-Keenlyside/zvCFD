"""Figures for docs/benchmarks/validation.md from benchmarks/results/validation/*.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results" / "validation"
FIG = ROOT / "docs" / "_static" / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "font.size": 9.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})


def load(name):
    f = RES / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _slope(ax, x0, y0, x1, p, label):
    xs = np.array([x0, x1])
    ax.plot(xs, y0 * (xs / x0) ** (-p), color=INK2, lw=1, ls="--")
    ax.text(x1 * 1.05, y0 * (x1 / x0) ** (-p), label, color=INK2, fontsize=8.5, va="center")


def convergence():
    duct, pipe, wo, tg, cy = (load(n) for n in ("duct", "pipe", "womersley", "taylor_green",
                                                "carreau_channel"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    k = 0
    if duct:
        r = duct["rows"]
        a.loglog([x["side"] for x in r], [x["l2"] for x in r], "o-", color=SERIES[k], lw=2, ms=6,
                 label="square duct (steady)")
    k += 1
    if wo:
        for alpha in (4.0, 12.0):
            r = [x for x in wo["plane"] if x["alpha"] == alpha]
            a.loglog([2 * x["h"] for x in r], [x["l2"] for x in r], "o-", color=SERIES[k], lw=2,
                     ms=6, label=f"Womersley channel, α = {alpha:g}")
            k += 1
    if tg:
        r = [x for x in tg["rows"] if x["start"] == "consistent"]
        a.loglog([x["N"] for x in r], [x["l2_u"] for x in r], "o-", color=SERIES[k], lw=2, ms=6,
                 label="Taylor–Green velocity")
        a.loglog([x["N"] for x in r], [x["l2_p"] for x in r], "s--", color=SERIES[k], lw=2,
                 ms=6, label="Taylor–Green pressure")
    k += 1
    if cy:
        r = cy["rows"]
        a.loglog([x["H"] for x in r], [x["l2"] for x in r], "o-", color=SERIES[k], lw=2, ms=6,
                 label="Carreau–Yasuda channel")
    _slope(a, 12, 2e-2, 110, 2, "2nd order")
    a.set_xlabel("cells across the flow (or vortex period)")
    a.set_ylabel("relative L2 error")
    a.set_title("Walls on lattice planes, or no walls", loc="left", fontsize=10)
    a.legend(loc="upper right", fontsize=8.2)

    k = 0
    if pipe:
        r = pipe["rows"]
        b.loglog([2 * x["R"] for x in r], [x["l2"] for x in r], "o-", color=SERIES[k], lw=2, ms=6,
                 label="Hagen–Poiseuille pipe")
    k += 1
    if wo:
        for alpha in (4.0, 12.0):
            r = [x for x in wo["pipe"] if x["alpha"] == alpha]
            b.loglog([2 * x["R"] for x in r], [x["l2"] for x in r], "o-", color=SERIES[k], lw=2,
                     ms=6, label=f"Womersley pipe, α = {alpha:g}")
            k += 1
    _slope(b, 12, 1.2e-1, 110, 1, "1st order")
    b.set_xlabel("cells across the pipe (diameter)")
    b.set_title("Curved walls (staircase bounce-back)", loc="left", fontsize=10)
    b.legend(loc="lower left", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(FIG / "validation_convergence.png", dpi=160)
    plt.close(fig)


def womersley_profiles():
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import exact

    wo = load("womersley")
    if not wo:
        return
    nu = (wo["tau"] - 0.5) / 3
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, alpha in zip(axes, (4.0, 12.0)):
        row = next(x for x in wo["plane"] if x["alpha"] == alpha and x.get("profiles"))
        zeta = np.asarray(row["zeta"])
        H = 2 * row["h"]
        scale = max(max(abs(v) for v in p["exact"]) for p in row["profiles"])
        cmap = matplotlib.colormaps["Blues"]
        prof = row["profiles"][::2]
        fine = np.linspace(0, H, 400)
        for i, p in enumerate(prof):
            col = cmap(0.35 + 0.6 * i / max(len(prof) - 1, 1))
            ex = exact.womersley_plane(fine, H, row["F0"], row["omega"], nu,
                                       p["phase"] * row["period"]) / scale
            ax.plot(ex, fine / H, color=col, lw=1.6)
            ax.plot(np.asarray(p["u"]) / scale, zeta / H, "o", color=col, ms=3.2,
                    markeredgecolor=SURFACE, markeredgewidth=0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("u / max |u|")
        ax.set_title(f"α = {alpha:g}: exact (lines), zvCFD (dots), 8 phases", loc="left",
                     fontsize=10)
    axes[0].set_ylabel("position across the channel, z / H")
    fig.tight_layout()
    fig.savefig(FIG / "validation_womersley.png", dpi=160)
    plt.close(fig)


def spheres():
    sa = load("sphere_array")
    if not sa:
        return
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import exact

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    phi = np.pi * exact.SC_CHI ** 3 / 6
    a.semilogy(phi, exact.SC_DRAG, color=INK2, lw=1.5, label="Sangani & Acrivos (1982)")
    for k, L in enumerate((32, 64)):
        r = [x for x in sa["rows"] if x["L"] == L]
        a.semilogy([x["phi"] for x in r], [x["C"] for x in r], "o", color=SERIES[k], ms=6,
                   label=f"zvCFD, cell {L}³")
        b.plot([x["phi"] for x in r], [100 * x["err"] for x in r], "o-", color=SERIES[k], lw=2,
               ms=6, label=f"zvCFD, cell {L}³")
        b.plot(phi, 100 * exact.BOGNER_ERR[L], "x:", color=SERIES[k], lw=1.2, ms=6,
               label=f"Bogner et al. (2015) TRT, {L}³")
    a.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_yticks([1, 2, 5, 10, 20])
    a.set_xlabel("solid volume fraction φ")
    a.set_ylabel("normalised drag C")
    a.set_title("Simple cubic array of spheres, Stokes flow", loc="left", fontsize=10)
    a.legend(loc="upper left", fontsize=8.2)
    b.axhline(0, color=INK2, lw=1)
    b.set_xlabel("solid volume fraction φ")
    b.set_ylabel("error against Sangani & Acrivos (%)")
    b.legend(loc="upper left", fontsize=8.2, ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / "validation_spheres.png", dpi=160)
    plt.close(fig)


def carreau():
    cy = load("carreau_channel")
    if not cy:
        return
    row = next(x for x in cy["rows"] if x.get("profile"))
    p = row["profile"]
    z = np.asarray(p["zeta"]) / row["H"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    s = max(p["exact"])
    ax.plot(np.asarray(p["exact"]) / s, z, color=SERIES[0], lw=2, label="exact (stress balance)")
    ax.plot(np.asarray(p["u"]) / s, z, "o", color=SERIES[0], ms=4.5, markeredgecolor=SURFACE,
            markeredgewidth=0.6, label="zvCFD")
    ax.plot(np.asarray(p["newtonian_nu0"]) / s, z, "--", color=INK2, lw=1.5,
            label="Newtonian at the zero-shear viscosity")
    ax.set_xlabel("u / max u")
    ax.set_ylabel("z / H")
    ax.set_title("Carreau–Yasuda channel (blood exponents), H = 30", loc="left", fontsize=10)
    ax.legend(loc="center", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(FIG / "validation_carreau.png", dpi=160)
    plt.close(fig)


def main():
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    convergence()
    womersley_profiles()
    spheres()
    carreau()


if __name__ == "__main__":
    main()
