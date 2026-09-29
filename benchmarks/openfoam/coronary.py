"""The HiP-CT coronary case in OpenFOAM, on the collaborator's body-fitted Fluent mesh.

Steady laminar flow, Newtonian blood (nu = 3.5e-6 m^2/s): flow-rate inlet
(0.19396 mL/s, mean 0.1 m/s over the 1.94 mm^2 inlet, Re ~ 45), all 77
outlets at 0 Pa, no-slip walls (lumen, the two crop planes, the opening).
The same conditions as ``examples/coronary_50um.yaml`` for zvCFD.

    FOAM_SIF=esi2506.sif python benchmarks/openfoam/coronary.py <case dir> [--procs 16]

Needs the imported mesh in ``<case>/constant/polyMesh``
(``foamcase.import_fluent``). Writes ``<case>/result.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import foamcase as fc  # noqa: E402

Q_IN = 1.9396e-7
NU = 3.5e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--end", type=int, default=500,
                    help="SIMPLE iterations; the splits settle by ~250 (0.0016 pp per 100 at 400)")
    args = ap.parse_args()
    case = Path(args.case)
    names = fc.patch_names(case)
    inlet = [n for n in names if "inlet" in n]
    outlets = [n for n in names if "outlet" in n]
    monitored = inlet + outlets
    fc.write_system(case, n_procs=args.procs, end=args.end, flow_patches=monitored, nonorth=1,
                    write_interval=100, pressure_patches=inlet, modifiable=True)
    fc.write_physics(case, NU)
    outlet_bc = {"U": "type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0);",
                 "p": "type fixedValue; value uniform 0;"}
    fc.write_fields(case, {".*wall": fc.WALL, ".*velocity-inlet": fc.flow_rate(Q_IN),
                           ".*pressure-outlet": outlet_bc})
    t0 = time.time()
    timing = fc.run_parallel(case, args.procs)
    log = fc.read_log(case)
    flows = fc.read_flows(case, len(monitored))
    pin = sorted((case / "postProcessing" / "pressure_0").glob("*/surfaceFieldValue*.dat"))
    if pin:
        p_kin = float(np.loadtxt(pin[-1], comments="#")[-1, 1])
        (case / "inlet_pressure.txt").write_text(f"{p_kin * 1060.0} Pa (kinematic {p_kin})\n")
    res = {"patches": monitored, "timing": timing, "wall_s": time.time() - t0,
           "iterations": log["iterations"], "converged": log["converged"],
           "exec_s": log["exec_s"], "flows_final": flows[-1].tolist(),
           "flows_history_every10": flows[::10].tolist(), "procs": args.procs,
           "cells": 14790642}
    (case / "result.json").write_text(json.dumps(res))
    q = flows[-1]
    print(f"{log['iterations']} iterations, converged={log['converged']}, "
          f"simpleFoam {timing['solve_s']:.0f} s; inflow {-q[0]:.4e}, "
          f"outflow {q[1:].sum():.4e} m3/s")


if __name__ == "__main__":
    main()
