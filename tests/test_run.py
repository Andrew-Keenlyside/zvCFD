"""End to end: a Fluent-format mesh through `zvcfd.run.run` to a run collection."""

import json

import pytest
from test_fluent_msh import CUBE

pytestmark = [pytest.mark.gpu, pytest.mark.store]


def test_mesh_run_end_to_end(tmp_path):
    from zvcfd import collection
    from zvcfd.config import from_dict
    from zvcfd.run import load_patch_table, run

    msh = tmp_path / "cube.msh"
    msh.write_text(CUBE)                      # unit cube in mm: inlet x=0, outlet x=1
    cfg = from_dict({
        "name": "cube",
        "source": {"kind": "mesh", "path": str(msh), "unit": "mm", "voxel_size": 50.0},
        "physics": {"nu": 1e-6, "rho": 1000.0},
        "boundaries": {"patches": [
            {"match": "inlet", "kind": "velocity", "flow_rate": 1e-10},
            {"match": "outlet", "kind": "pressure", "pressure": 0.0}]},
        "solver": {"tau": 1.0, "steps": 20000, "check_every": 500, "tolerance": 1e-4},
        "domain": {"chunk_bricks": 2},
    })
    lines = []
    run_dir = run(cfg, out=str(tmp_path / "runs"), log=lines.append)
    doc = collection.read(run_dir)
    assert collection.check(doc) == []
    assert collection.snapshots(doc)
    mon = json.loads((run_dir / "monitors.json").read_text())
    assert mon["summary"]["converged"]
    rows = load_patch_table(run_dir / "patches.csv")
    q = {r["kind"]: float(r["flow_m3s"]) for r in rows}
    assert q["velocity"] == pytest.approx(1e-10, rel=1e-2)          # flow control
    assert -q["pressure"] == pytest.approx(q["velocity"], rel=2e-3)  # mass balance
