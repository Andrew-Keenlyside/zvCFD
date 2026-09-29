"""Figures for docs/benchmarks, drawn from benchmarks/results/*.json.

    python benchmarks/figures.py        # writes docs/_static/figures/*.png

Palette: the dataviz reference categorical slots 1-2 (blue, orange),
validated for the light chart surface; text in neutral ink, never series
colour; one axis per chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "_static" / "figures"

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


def kernels():
    data = json.loads((RES / "bench_lbm_a2000.json").read_text())
    rows = [r for r in data["rows"] if "trt" not in r["case"]]       # BGK series
    cases = [("dense", "open"), ("sparse", "open"), ("dense", "porous"), ("sparse", "porous"),
             ("sparse", "vessels")]
    labels = ["dense, open box", "sparse, open box", "dense, porous φ=0.45",
              "sparse, porous φ=0.45", "sparse, vessels (3.6 % fluid)"]

    def eff(layout, geom, prec):
        for r in rows:
            c = r["case"]
            if c.startswith(f"{layout} {prec}") and geom in c:
                return 100 * r["of_copy"]
        return None

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    h = 0.36
    for i, (lay, geo) in enumerate(cases):
        for j, (prec, col) in enumerate((("fp32", S1), ("fp16", S2))):
            v = eff(lay, geo, prec)
            y = i + (j - 0.5) * (h + 0.04)
            ax.barh(y, v, height=h, color=col, label=prec if i == 0 else None)
            ax.text(v + 1, y, f"{v:.0f} %", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(cases)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("fluid-cell update rate × bytes per update, % of device copy bandwidth")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", title="population storage", title_fontsize=9)
    ax.set_title(f"D3Q19 kernels on an RTX A2000 (copy bandwidth {data['copy_gbs']:.0f} GB/s)",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "kernel_efficiency.png", dpi=160)
    plt.close(fig)


def io():
    rows = [r for r in json.loads((RES / "bench_io_ws.json").read_text())["rows"]
            if r["workers"] == 8]
    names = {"zv-flat-none": "ZV bricks, flat, raw", "zv-flat-zstd": "ZV bricks, flat, zstd",
             "zv-shard2-zstd": "ZV bricks, 2³ shards, zstd",
             "zarr-dense-none": "dense Zarr, raw", "zarr-dense-zstd": "dense Zarr, zstd",
             "icechunk-dense-zstd": "Icechunk dense, zstd"}
    labels = [names[r["variant"]] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, key, title in ((axes[0], "write_gbs", "Write (8 processes), GB/s of payload"),
                           (axes[1], "read_gbs", "Cold read (8 processes), GB/s")):
        vals = [r[key] for r in rows]
        ax.barh(range(len(rows)), vals, height=0.6, color=S1)
        for i, v in enumerate(vals):
            ax.text(v * 1.02 + 0.01, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.25)
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    fig.suptitle("One 2.4 GB snapshot, 4.1 % of bricks active, local NVMe", x=0.01, ha="left",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "io_throughput.png", dpi=160)
    plt.close(fig)


def amg():
    rows = json.loads((RES / "amg_check.json").read_text())
    n = [r["unknowns"] / 1e6 for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for key, col, lab in (("jacobi_cg_iters", S2, "Jacobi-preconditioned CG"),
                          ("amg_cg_iters", S1, "smoothed-aggregation AMG + CG")):
        y = [r[key] for r in rows]
        ax.plot(n, y, color=col, lw=2, marker="o", ms=6, label=lab)
        ax.annotate(f"{y[-1]:,}", (n[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("unknowns (millions); domain length 256 → 512 → 1024 voxels")
    ax.set_ylabel("iterations to 10⁻⁸")
    ax.set_xlim(0, max(n) * 1.18)
    ax.legend(loc="center right")
    ax.set_title("Lubrication pressure solve on the vessel network", loc="left", fontsize=10.5,
                 color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "amg_iterations.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    kernels()
    io()
    amg()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
