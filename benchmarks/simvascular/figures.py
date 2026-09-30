"""Figures for docs/validation/simvascular.md.

    python benchmarks/simvascular/figures.py [--best vmr0066-60um-sv]

Charts only; the 3-D renderings are ``renders.py``. SimVascular is
categorical slot 1 (blue) and zvCFD slot 2 (orange) in every chart, with
direct labels; magnitudes use the single-hue blue ramp and differences the
blue-red diverging pair with a grey midpoint (dataviz reference palette).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import compare as cmp  # noqa: E402
import vmrcase as vc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "_static" / "figures"
RES = cmp.RES

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
SV, ZV = "#2a78d6", "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
TAU_LOW, TAU_HIGH = "#184f95", "#86b6ef"      # ordinal steps of the blue ramp (600, 250)
SEQ = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("blue_red", ["#1c5cab", "#6da7ec", "#f0efec", "#ec8f8e",
                                                     "#c23535"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "font.size": 9.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})


def _label_end(ax, x, y, text, color_text=INK2, dy=0.0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, dy), textcoords="offset points",
                color=color_text, fontsize=8.5, va="center")


def boundary_and_pressure(zv, ref):
    """Inflow imposed at each cut, and the pressure drop each code needs to drive it."""
    pm = cmp.pressure_metrics(zv, ref)
    t = ref["t"] * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6), sharex=True)
    inl = {n: i for i, n in enumerate(zv["names"]) if n in vc.INLET_IDS}
    for c, (name, tree) in enumerate((("LCA_inlet", "LCA"), ("RCA_inlet", "RCA"))):
        a = axes[0, c]
        sel = [vc.outlet_tree(n) == tree for n in ref["outlets"]]
        q_sv = ref["q_out"][:, sel].sum(1) * 1e6
        a.plot(t, q_sv, color=SV, lw=2, label="SimVascular (imposed)")
        a.plot(zv["phase"] * 1e3, zv["q_last"][:, inl[name]] * 1e6, color=ZV, lw=1.5, ls="--",
               label="zvCFD (achieved)")
        a.set_title(f"{tree}: flow through the cut", loc="left", fontsize=10, color=INK)
        a.set_ylabel("mL/s")
        if c == 0:
            a.legend(loc="lower right", fontsize=8.5)
        b = axes[1, c]
        if pm:
            b.plot(t, pm[name]["p_sv"], color=SV, lw=2, label="SimVascular")
            b.plot(t, pm[name]["p_zv"], color=ZV, lw=1.5, label="zvCFD")
            _label_end(b, t, pm[name]["p_sv"], "SimVascular", dy=6)
            _label_end(b, t, pm[name]["p_zv"], "zvCFD", dy=-6)
        b.set_title(f"{tree}: pressure on the cut, relative to the outlets' mean", loc="left",
                    fontsize=10, color=INK)
        b.set_ylabel("Pa")
        b.set_xlabel("time in the cardiac cycle (ms)")
    fig.tight_layout()
    fig.savefig(FIG / "simvascular_boundary.png", dpi=160)
    plt.close(fig)


def outlet_flows(zv, ref):
    om = cmp.outlet_metrics(zv, ref)
    t = ref["t"] * 1e3
    fig = plt.figure(figsize=(12.4, 4.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1, 1, 1])
    a = fig.add_subplot(gs[:, 0])
    ms, mz = om["mean_sv"] * 1e6, om["mean_zv"] * 1e6
    lca = np.array([vc.outlet_tree(n) == "LCA" for n in ref["outlets"]])
    lim = [0, max(ms.max(), mz.max()) * 1.08]
    a.fill_between(lim, [lim[0] * 0.95, lim[1] * 0.95], [lim[0] * 1.05, lim[1] * 1.05],
                   color=GRID, alpha=0.6, lw=0, label="within ±5 %")
    a.plot(lim, lim, color=MUTED, lw=1)
    a.scatter(ms[lca], mz[lca], s=34, color=ZV, edgecolor=SURFACE, lw=1.5, marker="o",
              label="LCA outlets", zorder=3)
    a.scatter(ms[~lca], mz[~lca], s=38, color=ZV, edgecolor=SURFACE, lw=1.5, marker="s",
              label="RCA outlets", zorder=3)
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_aspect("equal")
    a.set_xlabel("SimVascular, cycle-mean outlet flow (mL/s)")
    a.set_ylabel("zvCFD, cycle-mean outlet flow (mL/s)")
    a.legend(loc="upper left", fontsize=8.5)
    a.set_title("24 outlets", loc="left", fontsize=10, color=INK)
    show = ["LAD", "LCX", "RCA", "LAD_b3_b1", "RCA_b3", "LCX_b4_b1"]
    for k, name in enumerate(show):
        b = fig.add_subplot(gs[k % 2, 1 + k // 2])
        i = ref["outlets"].index(name)
        b.plot(t, ref["q_out"][:, i] * 1e6, color=SV, lw=2)
        b.plot(t, om["q_zv"][:, i] * 1e6, color=ZV, lw=1.5)
        b.set_title(name, loc="left", fontsize=9.5, color=INK)
        if k % 2 == 1:
            b.set_xlabel("ms")
        else:
            b.tick_params(labelbottom=False)
        if k // 2 == 0:
            b.set_ylabel("mL/s")
    handles = [plt.Line2D([], [], color=SV, lw=2), plt.Line2D([], [], color=ZV, lw=1.5)]
    fig.legend(handles, ["SimVascular", "zvCFD"], loc="upper right", ncol=2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "simvascular_outlets.png", dpi=160)
    plt.close(fig)


def convergence(summary):
    runs = {k: v for k, v in summary["runs"].items()
            if v["run"]["inlet"] == "sv" and v["run"].get("u_lat", 0.05) == 0.05}
    if len(runs) < 2:
        return
    dx = np.array([v["run"]["voxel_um"] for v in runs.values()])
    o = np.argsort(dx)[::-1]
    dx = dx[o]
    vals = list(runs.values())
    def volume(v):
        f = v.get("fields") or {}
        return np.mean([x["velocity_rel_l2_pct"] for x in f.values()]) if f else np.nan

    series = [("outlet mean flow (mean |difference|)",
               [v["outlets"]["mean_flow_err_pct_mean_abs"] for v in vals]),
              ("pressure drop across the LCA",
               [abs(v.get("pressure", {}).get("LCA_inlet", {}).get("mean_err_pct", np.nan))
                for v in vals]),
              ("velocity, whole volume", [volume(v) for v in vals]),
              ("TAWSS", [v.get("wall", {}).get("tawss_rel_l2_pct", np.nan) for v in vals])]
    fig, a = plt.subplots(figsize=(7.4, 4.0))
    for k, (label, y) in enumerate(series):
        y = np.array(y)[o]
        a.plot(dx, y, "o-", color=SERIES[k], lw=2, ms=7, mec=SURFACE, mew=1.5)
        _label_end(a, dx, y, label, dy=5 if k == 3 else 0)
    a.set_xlabel("voxel size (µm)")
    a.set_ylabel("difference from SimVascular (%)")
    a.set_xticks(dx)
    a.set_xticklabels([f"{d:g}" for d in dx])
    a.set_ylim(0, None)
    a.set_xlim(dx.min() * 0.35, dx.max() * 1.05)
    a.invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIG / "simvascular_convergence.png", dpi=160)
    plt.close(fig)


def sections(zv, ref, ms=530, which=("LM", "LAD proximal", "RCA proximal")):
    pr = np.load(RES / "probes.npz", allow_pickle=True)
    labels = [str(s) for s in pr["section_labels"]]
    sec = pr["kind"] == 3
    sid, uv = pr["section"], pr["section_uv"]
    iz = int(np.argmin(np.abs(zv["probe_t"] - ms / 1e3)))
    isv = int(np.argmin(np.abs(ref["t"] - ms / 1e3)))
    uz = zv["probe_u"][iz][sec]
    us = ref["probe_u"][isv][sec]
    fig, axes = plt.subplots(len(which), 3, figsize=(8.2, 2.6 * len(which)))
    for r, label in enumerate(which):
        k = labels.index(label)
        m = sid == k
        g = uv[m] * 10                                    # mm
        tvec = pr["section_t"][k]
        sz, ss = uz[m] @ tvec, us[m] @ tvec               # axial velocity (m/s)
        vmax = np.nanmax(ss)
        for c, (vals, title, cmap, lim) in enumerate((
                (ss, "SimVascular", SEQ, (0, vmax)), (sz, "zvCFD", SEQ, (0, vmax)),
                (sz - ss, "zvCFD − SimVascular", DIV, (-0.25 * vmax, 0.25 * vmax)))):
            a = axes[r, c]
            sc = a.scatter(g[:, 0], g[:, 1], c=vals, s=7, marker="s", cmap=cmap, vmin=lim[0],
                           vmax=lim[1], lw=0)
            loop = pr["section_loops"][k] * 10
            a.plot(np.r_[loop[:, 0], loop[:1, 0]], np.r_[loop[:, 1], loop[:1, 1]], color=INK2,
                   lw=1)
            a.set_aspect("equal")
            a.axis("off")
            if r == 0:
                a.set_title(title, fontsize=10, color=INK)
            if c == 0:
                a.text(-0.08, 0.5, label, transform=a.transAxes, rotation=90, va="center",
                       ha="right", color=INK, fontsize=10)
            cb = fig.colorbar(sc, ax=a, shrink=0.8, pad=0.02)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=8, colors=INK2)
            cb.set_label("m/s", color=INK2, fontsize=8)
    fig.suptitle(f"Axial velocity at {ms} ms (peak LCA flow)", color=INK, fontsize=10.5, x=0.02,
                 ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "simvascular_sections.png", dpi=160)
    plt.close(fig)


def tawss(zv, ref):
    wm = cmp.wall_metrics(zv, ref)
    if wm is None:
        return
    a_zv, a_sv, own = wm["tawss_zv"], wm["tawss_sv"], wm["tawss_own"]
    ok = np.isfinite(a_zv) & np.isfinite(a_sv)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    top = np.percentile(np.r_[a_sv[ok], a_zv[ok], own[ok]], 99.5)
    for a, x, y, xl, yl, title in (
            (axes[0], a_sv[ok], a_zv[ok], "SimVascular (Pa)", "zvCFD (Pa)",
             "TAWSS, same near-wall estimate for both codes"),
            (axes[1], own[ok], a_sv[ok], "SimVascular, its own gradients (Pa)",
             "SimVascular, near-wall estimate (Pa)", "The estimate against SimVascular's own")):
        a.hexbin(x, y, gridsize=70, extent=(0, top, 0, top), cmap=SEQ, mincnt=1, bins="log",
                 linewidths=0)
        a.plot([0, top], [0, top], color=MUTED, lw=1)
        a.set_xlim(0, top)
        a.set_ylim(0, top)
        a.set_aspect("equal")
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.set_title(title, loc="left", fontsize=10, color=INK)
        r = np.corrcoef(x, y)[0, 1]
        e = np.sqrt(((y - x) ** 2).sum() / (x ** 2).sum()) * 100
        a.text(0.04, 0.93, f"r = {r:.3f}\nrelative L2 {e:.1f} %", transform=a.transAxes,
               color=INK2, fontsize=8.5, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "simvascular_tawss.png", dpi=160)
    plt.close(fig)


def centreline_pressure(zv, ms=530, out="simvascular_centreline_pressure.png", note=""):
    """Pressure along the LAD and RCA centrelines, both codes, relative to p_ref."""
    from renders import VoxelField
    from svref import SVVolume

    sv = SVVolume()
    vf = VoxelField(zv["path"] / f"fields_{ms:04d}.npz")
    s_ = np.load(RES / "svsurface.npz")
    caps = [str(c) for c in s_["caps"]]
    cor = [i for i, c in enumerate(caps) if c not in ("inflow", "aorta")]
    step = int(np.argmin(np.abs(sv.t - ms / 1e3)))
    pref = s_["p"][step, cor].mean()

    def zv_p(pts):
        v = (pts - vf.x0) / vf.h
        base = np.floor(v).astype(np.int64)
        fr = v - base
        acc, tot = np.zeros(len(pts)), np.zeros(len(pts))
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    idx = vf.index(base + np.array([dz, dy, dx]))
                    w = ((fr[:, 0] if dz else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1])
                         * (fr[:, 2] if dx else 1 - fr[:, 2]))
                    w = np.where(idx >= 0, w, 0.0)
                    acc += w * vf.p[np.maximum(idx, 0)]
                    tot += w
        return np.where(tot > 0.3, acc / np.maximum(tot, 1e-12), np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    for a, (path, start, outlet) in zip(axes, (("LAD", 1.85, "LAD"), ("RCA", 1.0, "RCA"))):
        p = vc.read_path(path)
        sarc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        arcs = np.arange(start, sarc[-1], 0.02)
        pts = np.stack([np.interp(arcs, sarc, p[:, k]) for k in range(3)], 1)
        psv = sv.sample(sv.locate(pts), [step])[0][:, 3] - pref
        pzv = zv_p(pts)
        x = (arcs - start) * 10
        a.plot(x, psv, color=SV, lw=2, label="SimVascular")
        a.plot(x, pzv, color=ZV, lw=1.5, label="zvCFD")
        p_cap = s_["p"][step, caps.index(outlet)] - pref
        a.plot([x[-1] + 1.5], [p_cap], marker="o", ms=7, color=INK2, mec=SURFACE, mew=1.5)
        a.annotate("outlet pressure\n(imposed in zvCFD)", (x[-1] + 1.5, p_cap), xytext=(10, 0),
                   textcoords="offset points", ha="left", va="center", color=INK2, fontsize=8.5)
        ok = np.isfinite(psv) & np.isfinite(pzv)
        a.text(0.02, 0.06, f"zvCFD − SimVascular: {np.mean(pzv[ok] - psv[ok]):+.0f} Pa on average",
               transform=a.transAxes, color=INK2, fontsize=8.5)
        a.set_title(f"{path} centreline, {ms} ms{note}", loc="left", fontsize=10, color=INK)
        a.set_xlabel("distance from the inlet cut (mm)")
        a.set_xlim(0, x[-1] + 45)
    axes[0].set_ylabel("pressure relative to p_ref (Pa)")
    axes[1].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=160)
    plt.close(fig)


def oblique_pipe():
    """Pressure lost at straight-pipe patches, before and after the fix, against cap
    orientation and tau."""
    f = RES / "oblique_outlet.json"
    if not f.exists():
        return
    rows = json.loads(f.read_text())
    names = {(0, 0, 1): "along z", (1, 1, 0): "(1,1,0)", (1, 1, 1): "(1,1,1)",
             (1, 2, 3): "(1,2,3)"}
    schemes = [("neem", "Before: extrapolation on patch cells"), ("abb", "Now: patch links")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
    for a, (scheme, title) in zip(axes, schemes):
        labels = []
        for r in rows:
            if r.get("scheme", "neem") != scheme:
                continue
            ax = np.round(np.array(r["axis"]) / max(np.abs(r["axis"])), 3)
            key = min(names, key=lambda k: np.linalg.norm(np.array(k) / max(k) - ax))
            lab = names[key] + (",\ncaps ⊥ z" if r["caps"] == "z" else "")
            if lab not in labels:
                labels.append(lab)
            i = labels.index(lab)
            dx = -0.18 if r["tau"] < 0.7 else 0.18
            a.bar(i + dx, r["jump_in_pct"] + r["jump_out_pct"], width=0.34,
                  color=TAU_LOW if r["tau"] < 0.7 else TAU_HIGH, edgecolor=SURFACE, lw=2)
        a.axhline(0, color=MUTED, lw=1)
        a.set_xticks(range(len(labels)))
        a.set_xticklabels(labels, fontsize=8.5)
        a.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("share of Δp lost at the two patches (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=TAU_LOW),
               plt.Rectangle((0, 0), 1, 1, color=TAU_HIGH)]
    axes[0].legend(handles, ["τ = 0.55", "τ = 0.8"], loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "simvascular_oblique_pipe.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="vmr0066-60um-sv")
    ap.add_argument("--runs", default=str(cmp.RUNS))
    ap.add_argument("--before", default=str(vc.VMR_DIR / "runs_neem" / "vmr0066-100um-sv"),
                    help="a run from before the outlet fix, for the before/after centreline")
    args = ap.parse_args()
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    ref = cmp.sv_reference()
    zv = cmp.load_run(Path(args.runs) / args.best)
    boundary_and_pressure(zv, ref)
    outlet_flows(zv, ref)
    if "probe_u" in zv and "probe_u" in ref:
        sections(zv, ref)
        tawss(zv, ref)
    if zv["fields"]:
        centreline_pressure(zv, note=f", {zv['run']['voxel_um']:g} µm")
    if Path(args.before).exists():
        centreline_pressure(cmp.load_run(Path(args.before)),
                            out="simvascular_centreline_pressure_before.png",
                            note=", 100 µm, before the fix")
    oblique_pipe()
    summary = json.loads((RES / "summary.json").read_text())
    convergence(summary)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
