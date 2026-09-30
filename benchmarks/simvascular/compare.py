"""Compare zvCFD with SimVascular on the cut coronary trees of VMR 0066_H_CORO_H.

    python benchmarks/simvascular/compare.py [--runs /hdd/data/zvcfd_vmr/runs]

Reads SimVascular's reference (``svsurface.npz``, ``svprobes.npz`` from
``svref.py``) and every zvCFD run directory, and writes
``benchmarks/results/simvascular/summary.json``. All comparisons use the
last simulated cycle of each zvCFD run against SimVascular's last cycle
(its 11th), both sampled every 5 ms.

Quantities:

- outlet flows: the cycle-mean flow out of each of the 24 outlets, its
  share of the tree's inflow, and the whole waveform (relative L2 error
  over the cycle, normalised by the outlet's mean flow);
- the pressure drop across each tree: mean pressure on the inlet cut,
  relative to ``p_ref(t)``, over the cycle;
- velocity on six cross-sections, every 5 ms (relative L2 over the
  section points and the cycle), and over the whole fluid volume at the
  snapshot phases;
- wall shear stress: the same near-wall estimate for both codes,
  ``tau = mu (4 u_t(h) - u_t(2h)) / (2h)`` from the tangential velocity at
  h = 0.15 mm and 2h inside the wall (exact for a parabolic profile),
  time-averaged over the cycle (TAWSS); and SimVascular's own TAWSS from
  its finite-element gradients, which checks the estimate itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vmrcase as vc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results" / "simvascular"
RUNS = vc.VMR_DIR / "runs"


def rel_l2(a, b, axis=None):
    """||a - b|| / ||b||, ignoring NaN."""
    ok = np.isfinite(a) & np.isfinite(b)
    d = np.where(ok, a - b, 0.0)
    r = np.where(ok, b, 0.0)
    return np.sqrt((d ** 2).sum(axis)) / np.sqrt((r ** 2).sum(axis))


def sv_reference() -> dict:
    s = np.load(RES / "svsurface.npz")
    caps = [str(c) for c in s["caps"]]
    cor = [i for i, c in enumerate(caps) if c not in ("inflow", "aorta")]
    ref = {"t": s["t"], "outlets": [caps[i] for i in cor], "q_out": s["q_out"][:, cor],
           "p_out": s["p"][:, cor], "p_ref": s["p"][:, cor].mean(1),
           "tawss_own": s["tawss"], "points": s["points"]}
    if (RES / "svwall.npz").exists():             # the result file's own vTAWSS is all zeros
        ref["tawss_own"] = np.load(RES / "svwall.npz")["tawss_fe"]
    pv_path = RES / "svprobes.npz"
    if pv_path.exists():
        v = np.load(pv_path)
        ref.update({"probe_u": v["u"], "probe_p": v["p"], "cut_q": v["cut_q"],
                    "cut_p": v["cut_p"], "cut_names": [str(c) for c in v["cut_names"]]})
    return ref


def load_run(path: Path) -> dict:
    m = np.load(path / "monitors.npz")
    run = json.loads((path / "run.json").read_text())
    names = [str(n) for n in m["names"]]
    t, q = m["t"], m["q"]
    T = t[-1]
    last = t > T - vc.PERIOD - 1e-9
    out = {"path": path, "run": run, "names": names, "kinds": [str(k) for k in m["kinds"]],
           "t": t, "q": q, "tp": m["tp"], "p": m["p"], "phase": t[last] - (T - vc.PERIOD),
           "q_last": q[last], "cycles": int(round(T / vc.PERIOD))}
    lastp = m["tp"] > T - vc.PERIOD - 1e-9
    out["phase_p"], out["p_last"] = m["tp"][lastp] - (T - vc.PERIOD), m["p"][lastp]
    pr = path / "probes.npz"
    if pr.exists():
        z = np.load(pr)
        out["probe_t"], out["probe_u"] = z["t"] - (T - vc.PERIOD), z["u"]
    out["fields"] = sorted(path.glob("fields_*.npz"))
    return out


def outlet_metrics(zv: dict, ref: dict) -> dict:
    t = ref["t"]
    idx = [zv["names"].index(n) for n in ref["outlets"]]
    q_zv = np.stack([np.interp(t, zv["phase"], -zv["q_last"][:, i]) for i in idx], 1)
    q_sv = ref["q_out"]
    mean_zv, mean_sv = q_zv.mean(0), q_sv.mean(0)
    err_mean = (mean_zv - mean_sv) / mean_sv
    wave = np.sqrt(((q_zv - q_sv) ** 2).mean(0)) / mean_sv
    trees = {}
    for tree in ("LCA", "RCA"):
        sel = np.array([vc.outlet_tree(n) == tree for n in ref["outlets"]])
        sz = mean_zv[sel] / mean_zv[sel].sum()
        ss = mean_sv[sel] / mean_sv[sel].sum()
        trees[tree] = {"split_diff_pp_mean": float(np.abs(sz - ss).mean() * 100),
                       "split_diff_pp_max": float(np.abs(sz - ss).max() * 100)}
    # periodicity: the previous cycle's outlet means
    prev = (zv["t"] > zv["t"][-1] - 2 * vc.PERIOD + 1e-9) & (zv["t"] <= zv["t"][-1] - vc.PERIOD)
    per = None
    if zv["cycles"] >= 3 and prev.any():             # cycle 1 holds the start-up ramp
        m_prev = np.array([-zv["q"][prev, i].mean() for i in idx])
        m_last = np.array([-zv["q_last"][:, i].mean() for i in idx])
        per = float(np.abs(m_last - m_prev).max() / np.abs(m_last).mean())
    inl = [i for i, k in enumerate(zv["kinds"]) if k == "velocity"]
    qin = zv["q_last"][:, inl].sum(1)
    return {"q_zv": q_zv, "mean_zv": mean_zv, "mean_sv": mean_sv,
            "summary": {"mean_flow_err_pct_mean_abs": float(np.abs(err_mean).mean() * 100),
                        "mean_flow_err_pct_max_abs": float(np.abs(err_mean).max() * 100),
                        "waveform_err_pct_median": float(np.median(wave) * 100),
                        "waveform_err_pct_max": float(wave.max() * 100),
                        "trees": trees, "periodicity_max_change": per,
                        "imbalance_max_abs": float(np.abs(zv["q_last"].sum(1) / qin).max())},
            "per_outlet": {n: {"sv_mL_s": float(a * 1e6), "zv_mL_s": float(b * 1e6),
                               "err_pct": float(e * 100), "waveform_err_pct": float(w * 100)}
                           for n, a, b, e, w in zip(ref["outlets"], mean_sv, mean_zv, err_mean,
                                                    wave)}}


def pressure_metrics(zv: dict, ref: dict) -> dict | None:
    if "cut_p" not in ref:
        return None
    t = ref["t"]
    out = {}
    for k, name in enumerate(ref["cut_names"]):
        i = zv["names"].index(name)
        p_zv = np.interp(t, zv["phase_p"], zv["p_last"][:, i])
        p_sv = ref["cut_p"][k] - ref["p_ref"]
        out[name] = {"p_zv": p_zv, "p_sv": p_sv,
                     "mean_zv_pa": float(p_zv.mean()), "mean_sv_pa": float(p_sv.mean()),
                     "mean_err_pct": float((p_zv.mean() - p_sv.mean()) / abs(p_sv.mean()) * 100),
                     "waveform_err_pct": float(rel_l2(p_zv, p_sv) * 100)}
    return out


def wss(u1, u2, normal, h_m, mu=vc.MU):
    """Wall shear stress vector from velocities at h and 2h inside the wall: ``(..., N, 3)`` Pa."""
    def tang(u):
        return u - (u * normal).sum(-1, keepdims=True) * normal
    return mu * (4 * tang(u1) - tang(u2)) / (2 * h_m)


def wall_metrics(zv: dict, ref: dict) -> dict | None:
    if "probe_u" not in zv or "probe_u" not in ref:
        return None
    pr = np.load(RES / "probes.npz", allow_pickle=True)
    kind, n, h = pr["kind"], pr["normal"], float(pr["h"]) * 1e-2
    t = ref["t"]
    # zvCFD's probe times fall on SimVascular's (every 5 ms); match by phase
    iz = np.round(zv["probe_t"] / 0.005).astype(int)
    isv = np.round(t / 0.005).astype(int)
    common = np.intersect1d(iz, isv)
    uz = zv["probe_u"][np.searchsorted(iz, common)]
    us = ref["probe_u"][np.searchsorted(isv, common)]
    w_zv = np.linalg.norm(wss(uz[:, kind == 1], uz[:, kind == 2], n, h), axis=-1)
    w_sv = np.linalg.norm(wss(us[:, kind == 1], us[:, kind == 2], n, h), axis=-1)
    ta_zv, ta_sv = np.nanmean(w_zv, 0), np.nanmean(w_sv, 0)
    own = ref["tawss_own"][pr["wall_node"]]
    ok = np.isfinite(ta_zv) & np.isfinite(ta_sv)
    sec = kind == 3
    u_sec_err = rel_l2(uz[:, sec], us[:, sec])
    per_section = {}
    sid = pr["section"]
    for k, label in enumerate(pr["section_labels"]):
        m = sid == k
        per_section[str(label)] = float(rel_l2(uz[:, sec][:, m], us[:, sec][:, m]) * 100)
    return {"tawss_zv": ta_zv, "tawss_sv": ta_sv, "tawss_own": own,
            "summary": {"tawss_rel_l2_pct": float(rel_l2(ta_zv[ok], ta_sv[ok]) * 100),
                        "tawss_median_abs_err_pct": float(
                            np.median(np.abs(ta_zv[ok] - ta_sv[ok]) / ta_sv[ok]) * 100),
                        "tawss_corr": float(np.corrcoef(ta_zv[ok], ta_sv[ok])[0, 1]),
                        "estimate_vs_sv_own_rel_l2_pct": float(rel_l2(ta_sv[ok], own[ok]) * 100),
                        "estimate_vs_sv_own_corr": float(np.corrcoef(ta_sv[ok], own[ok])[0, 1]),
                        "section_velocity_rel_l2_pct": float(u_sec_err * 100),
                        "section_velocity_rel_l2_pct_by_section": per_section}}


def field_metrics(zv: dict) -> dict:
    """Whole-volume velocity error at each snapshot phase (SimVascular sampled at voxel centres)."""
    from svref import SVVolume

    out = {}
    if not zv["fields"]:
        return out
    sv = SVVolume()
    cache = zv["path"] / "sv_at_voxels.npz"
    f0 = np.load(zv["fields"][0])
    if cache.exists():
        c = np.load(cache)
        loc = (c["cells"], c["w"])
    else:
        loc = sv.locate(f0["xyz"].astype(np.float64))
        np.savez_compressed(cache, cells=loc[0], w=loc[1])
    interior = f0["flag"] == 0
    for fpath in zv["fields"]:
        f = np.load(fpath)
        ms = int(fpath.stem.split("_")[1])
        step = int(np.argmin(np.abs(sv.t - ms / 1000)))
        s = sv.sample(loc, [step])[0]
        ok = interior & np.isfinite(s[:, 0])
        du = f["u"][ok] - s[ok, :3]
        err = np.sqrt((du ** 2).sum() / (s[ok, :3] ** 2).sum())
        out[str(ms)] = {"velocity_rel_l2_pct": float(err * 100),
                        "speed_max_zv": float(np.linalg.norm(f["u"][ok], axis=1).max()),
                        "speed_max_sv": float(np.linalg.norm(s[ok, :3], axis=1).max()),
                        "voxels_compared": int(ok.sum()),
                        "voxels_outside_sv": int((interior & ~np.isfinite(s[:, 0])).sum())}
    return out


def self_convergence(runs: Path, ref: dict) -> dict:
    """zvCFD against its own finest run (SimVascular inlet profile, default lattice velocity).

    Separates zvCFD's discretisation error from the difference between
    the codes: if zvCFD has settled while the gap to SimVascular stays, the
    gap is not zvCFD's resolution.
    """
    series = sorted((p for p in runs.glob("vmr0066-*um-sv") if (p / "monitors.npz").exists()),
                    key=lambda p: -float(p.name.split("-")[1][:-2]))
    if len(series) < 2:
        return {}
    fine = load_run(series[-1])
    om_f = outlet_metrics(fine, ref)
    pm_f = pressure_metrics(fine, ref)
    wm_f = wall_metrics(fine, ref)
    out = {"finest": series[-1].name}
    for path in series[:-1]:
        zv = load_run(path)
        om, pm, wm = outlet_metrics(zv, ref), pressure_metrics(zv, ref), wall_metrics(zv, ref)
        d = (om["mean_zv"] - om_f["mean_zv"]) / om_f["mean_zv"]
        rec = {"outlet_mean_flow_diff_pct_mean_abs": float(np.abs(d).mean() * 100),
               "outlet_mean_flow_diff_pct_max_abs": float(np.abs(d).max() * 100)}
        if pm and pm_f:
            rec["inlet_pressure_diff_pct"] = {
                k: float((pm[k]["mean_zv_pa"] / pm_f[k]["mean_zv_pa"] - 1) * 100) for k in pm}
        if wm and wm_f:
            ok = np.isfinite(wm["tawss_zv"]) & np.isfinite(wm_f["tawss_zv"])
            rec["tawss_rel_l2_pct"] = float(rel_l2(wm["tawss_zv"][ok], wm_f["tawss_zv"][ok]) * 100)
        out[path.name] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--no-fields", action="store_true")
    args = ap.parse_args()
    ref = sv_reference()
    summary = {"sv": {"outlets": ref["outlets"],
                      "mean_q_mL_s": (ref["q_out"].mean(0) * 1e6).tolist()}, "runs": {}}
    if "cut_q" in ref:
        summary["sv"]["cut_q_mean_mL_s"] = (ref["cut_q"].mean(1) * 1e6).tolist()
        tree_q = [ref["q_out"][:, [vc.outlet_tree(n) == tr for n in ref["outlets"]]].sum(1)
                  for tr in ("LCA", "RCA")]
        summary["sv"]["cut_vs_outlets_rel_l2_pct"] = [
            float(rel_l2(ref["cut_q"][k], tree_q[k]) * 100) for k in range(2)]
    for path in sorted(Path(args.runs).glob("vmr0066-*")):
        if not (path / "monitors.npz").exists():
            continue
        zv = load_run(path)
        om = outlet_metrics(zv, ref)
        rec = {"run": zv["run"], "cycles": zv["cycles"], "outlets": om["summary"],
               "per_outlet": om["per_outlet"]}
        pm = pressure_metrics(zv, ref)
        if pm:
            rec["pressure"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("p_")}
                               for k, v in pm.items()}
        wm = wall_metrics(zv, ref)
        if wm:
            rec["wall"] = wm["summary"]
        if not args.no_fields:
            rec["fields"] = field_metrics(zv)
        summary["runs"][path.name] = rec
        print(path.name, json.dumps(rec["outlets"], default=float)[:300])
    summary["self_convergence"] = self_convergence(Path(args.runs), ref)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"-> {RES / 'summary.json'}")


if __name__ == "__main__":
    main()
