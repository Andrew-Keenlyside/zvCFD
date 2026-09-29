"""How fast zvCFD reaches a steady Stokes solution as a function of tau (TRT).

With TRT and magic 3/16 the steady solution does not depend on tau, so tau
only sets how quickly the solver gets there. Two processes compete: momentum
diffusion across a pore or vessel (faster at large tau, since nu = (tau - 1/2)/3)
and pressure equilibration along the sample, which in a weakly compressible
solver behaves like diffusion with coefficient c_s^2 K / nu (faster at small
tau). Tubes are limited by the first, porous media by the second.

The lattice pressure drop is held fixed across the scan (the physical
gradient is scaled with 1/(tau - 1/2)^2 and the flux scaled back; exact for
Stokes flow), so every tau sees the same Mach number and fp32 resolution.

    python benchmarks/openfoam/tau_scan.py --cases pipe12,porous,vessels \
        --taus 0.55,0.6,0.7,0.8,1.0,1.5,2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import voxel_compare as vc  # noqa: E402

OUT = vc.RESULTS / "tau_scan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="pipe12,porous,vessels")
    ap.add_argument("--taus", default="0.55,0.6,0.7,0.8,1.0,1.5,2.0")
    ap.add_argument("--max-steps", type=int, default=400_000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    g0 = vc.G_KIN
    for name in args.cases.split(","):
        fluid, _ = vc.case_geometry(name)
        rows = []
        for tau in (float(t) for t in args.taus.split(",")):
            scale = (1.5 / (tau - 0.5)) ** 2          # same lattice pressure drop as tau = 2
            vc.G_KIN = g0 * scale
            r = vc.run_zvcfd(name, fluid, tau=tau, max_steps=args.max_steps)
            vc.G_KIN = g0
            row = {"tau": tau, "flux": r["flux_final"] / scale, "steps": r["steps"],
                   "mlups": r["mlups"], **{k: r[k] for k in r if k.startswith(("wall_", "iters_"))}}
            rows.append(row)
            print(f"{name:<8} tau {tau:5.2f}  flux {row['flux']:.5e}  steps {r['steps']:>7}  "
                  f"to 1%: {r['wall_to_0.01']:7.2f} s  to 0.1%: {r['wall_to_0.001']:7.2f} s",
                  flush=True)
        (OUT / f"{name}.json").write_text(json.dumps({"case": name, "g_kin": g0, "rows": rows},
                                                     indent=1))


if __name__ == "__main__":
    main()
