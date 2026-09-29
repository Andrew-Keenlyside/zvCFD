"""Same inputs, same code, same GPU: every physics output byte-identical.

Run ids and wall-clock timings are the only outputs allowed to differ.
"""

import hashlib
import json

import numpy as np
import pytest
from test_fluent_msh import CUBE

pytestmark = [pytest.mark.gpu]


def _porous_run():
    import cupy as cp

    from zvcfd.boundary import Patch, face_patches
    from zvcfd.domain import BrickDomain
    from zvcfd.lbm import SparseLBM

    rng = np.random.default_rng(3)
    flags = (rng.random((16, 16, 32)) < 0.3).astype(np.uint8)
    flags[:, :, :2] = 0
    flags[:, :, -2:] = 0
    dom = BrickDomain.from_flags(flags)
    b = face_patches(dom, {"xmin": Patch("in", "velocity"), "xmax": Patch("out", "pressure")})
    sim = SparseLBM(dom, tau=0.7, collision="trt", boundary=b)
    sim.set_patch(0, u_zyx=(0.0, 0.0, 0.01))
    flows = []
    for _ in range(10):
        sim.step(50)
        q = sim.patch_flux()
        flows.append(q)
        # feed the measured flux back into the inlet, as flow control does
        sim.set_patch(0, u_zyx=(0.0, 0.0, 0.01 * (1 + 0.1 * (q[0] > 0))))
    cp.cuda.Device().synchronize()
    return hashlib.sha256(sim.f0.get().tobytes()).hexdigest(), np.array(flows)


def test_solver_rerun_is_bit_identical():
    h0, q0 = _porous_run()
    h1, q1 = _porous_run()
    assert h0 == h1
    assert np.array_equal(q0, q1)


@pytest.mark.store
def test_run_outputs_are_byte_identical(tmp_path):
    from zvcfd.config import from_dict
    from zvcfd.run import run

    msh = tmp_path / "cube.msh"
    msh.write_text(CUBE)
    cfg = {"name": "det", "source": {"kind": "mesh", "path": str(msh), "unit": "mm",
                                     "voxel_size": 50.0},
           "physics": {"nu": 1e-6, "rho": 1000.0},
           "boundaries": {"patches": [{"match": "inlet", "kind": "velocity", "flow_rate": 1e-10},
                                      {"match": "outlet", "kind": "pressure", "pressure": 0.0}]},
           "solver": {"tau": 0.8, "steps": 3000, "check_every": 500, "tolerance": 1e-9},
           "domain": {"chunk_bricks": 2}, "output": {"every": 1000}}
    dirs = [run(from_dict(cfg), out=str(tmp_path / f"r{k}"), log=lambda s: None) for k in (0, 1)]

    def digests(d):
        return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(d.rglob("*")) if p.is_file()}

    a, b = digests(dirs[0]), digests(dirs[1])
    assert a.keys() == b.keys()
    differ = {k for k in a if a[k] != b[k]}
    assert differ <= {"zarr.json", "monitors.json"}           # run id; wall-clock timings
    m0, m1 = (json.loads((d / "monitors.json").read_text()) for d in dirs)
    for x, y in zip(m0["history"], m1["history"]):
        assert {k: v for k, v in x.items() if k != "wall_s"} == \
               {k: v for k, v in y.items() if k != "wall_s"}
    assert (dirs[0] / "patches.csv").read_bytes() == (dirs[1] / "patches.csv").read_bytes()
