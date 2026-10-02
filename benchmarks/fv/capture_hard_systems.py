"""Capture the four hard coupled systems of the linear-solver study, for ``amgx_smoothers.py``.

    PYTHONPATH=. python benchmarks/fv/capture_hard_systems.py OUT_DIR

Each system is the scaled coupled system of an early outer iteration (fixed
advection blend), with its starting guess, as ``OUT_DIR/<name>.npz``:

- ``gateb_tube12``: the gate B tube (``gate_b.case(12)``, 72 k nodes, wedge
  layers at the wall);
- ``dfg_slab``: the DFG 2D-2 channel, one cell deep, Δt = 0.005 s;
- ``layered_tube``: a tube with strongly graded wall layers (growth 0.45);
- ``thin_slab``: a channel of cells 10× thinner than wide.

These are the systems of docs/spec/fv_numerics.md (linear solvers on thin
cells), solved by ``benchmarks/fv/amgx_smoothers.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "fv"))
R = 0.5


def tube_profile(x):
    return np.stack([0 * x[:, 0], 0 * x[:, 0],
                     2 * np.clip(1 - (x[:, 0] ** 2 + x[:, 1] ** 2) / R ** 2, 0, None)], 1)


def save(out: Path, name, m, bcs, fluid, u0, dt=None, its=3):
    import cupy as cp

    from zvcfd.fv.solver import GPUSolver

    s = GPUSolver(m, fluid, bcs, linear="host-direct" if m.n_nodes < 3000 else "acm",
                  advection=1.0, dt=dt)
    s.initialise(U=u0)
    s.solve(max_iterations=its, tol=0)
    A, b = s.assemble()
    x0 = s._state()
    s._scale_rows(A, b)
    np.savez(out / f"{name}.npz", indptr=cp.asnumpy(A.indptr), indices=cp.asnumpy(A.indices),
             data=cp.asnumpy(A.data), b=cp.asnumpy(b), x0=cp.asnumpy(x0))
    print(name, m.n_nodes, flush=True)


def main():
    from gate_b import case

    from zvcfd.fv.reference import Fluid
    from zvcfd.mesh.generate import box, cylinder_channel, tube

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    m, bcs = case(12)
    save(out, "gateb_tube12", m, bcs, Fluid(1.0, 0.01), tube_profile)

    msh = cylinder_channel(16)
    z = {b.name: k for k, b in msh.zones.items()}

    def inflow(x):
        return np.stack([4 * 1.5 * x[:, 1] * (0.41 - x[:, 1]) / 0.41 ** 2, 0 * x[:, 0],
                         0 * x[:, 0]], 1)

    save(out, "dfg_slab", msh, {z["inlet"]: {"kind": "velocity", "value": inflow},
                                z["outlet"]: {"kind": "pressure", "value": 0.0},
                                z["walls"]: {"kind": "wall"}, z["cylinder"]: {"kind": "wall"},
                                z["front"]: {"kind": "symmetry"},
                                z["back"]: {"kind": "symmetry"}},
         Fluid(1.0, 1e-3), inflow, dt=0.005, its=2)

    m = tube(R, 2.0, n_core=8, n_ring=8, n_axial=32, kind="mixed", layers=6, growth=0.45)
    z = {b.name: k for k, b in m.zones.items()}
    save(out, "layered_tube", m, {z["inlet"]: {"kind": "velocity", "value": tube_profile},
                                  z["outlet"]: {"kind": "pressure", "value": 0.0},
                                  z["wall"]: {"kind": "wall"}}, Fluid(1.0, 0.01), tube_profile)

    m = box((48, 16, 1), (3.0, 1.0, 1 / 160), kind="hex")
    z = {b.name: k for k, b in m.zones.items()}

    def inflow2(x):
        return np.stack([4 * x[:, 1] * (1 - x[:, 1]), 0 * x[:, 0], 0 * x[:, 0]], 1)

    save(out, "thin_slab", m, {z["xmin"]: {"kind": "velocity", "value": inflow2},
                               z["xmax"]: {"kind": "pressure", "value": 0.0},
                               z["ymin"]: {"kind": "wall"}, z["ymax"]: {"kind": "wall"},
                               z["zmin"]: {"kind": "symmetry"}, z["zmax"]: {"kind": "symmetry"}},
         Fluid(1.0, 0.01), inflow2)


if __name__ == "__main__":
    main()
