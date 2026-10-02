"""``zvcfd run`` with ``solver.method: fv``: a pipe in a Fluent mesh, end to end.

The mesh is written in millimetres; the run imports it into a Zarr Vectors
mesh collection in metres, solves on the mesh read back from it, maps zones
by patch rules, solves (steady, then a short transient with an
RCR outlet), and writes the run collection: a link to the mesh collection,
a boundary store,
node-field snapshots, TAWSS on the boundary store, monitors and the patch
table. The numbers are checked against the physics: the flow-rate inlet is
exact, the outlet satisfies the RCR relation, mass balances.
"""

import csv
import json

import numpy as np
import pytest

from zvcfd import collection as col
from zvcfd.config import from_dict
from zvcfd.mesh.fluent import write_fluent_mesh
from zvcfd.mesh.generate import tube

pytestmark = pytest.mark.gpu

Q = 2.0e-7                      # m³/s into a 1 mm radius pipe: U_mean 6.4 cm/s
RCR_SI = (1.0e8, 1.0e-10, 4.0e8)


@pytest.fixture(scope="module")
def pipe_msh(tmp_path_factory):
    m = tube(1.0, 6.0, n_core=3, n_ring=3, n_axial=12, kind="mixed", layers=2, growth=0.8)
    path = tmp_path_factory.mktemp("mesh") / "pipe.msh"
    write_fluent_mesh(m, path)
    return path


def _config(path, **fv):
    return from_dict({
        "name": "pipe-fv",
        "source": {"kind": "mesh", "path": str(path), "unit": "mm"},
        "physics": {"nu": 3.5e-6, "rho": 1060.0},
        "boundaries": {"patches": [
            {"match": "inlet", "kind": "velocity", "flow_rate": Q, "profile": "parabolic",
             "waveform": [[0.0, 1.0], [0.5, 1.5], [1.0, 1.0]]},
            {"match": "outlet", "kind": "rcr", "rcr": list(RCR_SI), "pressure": 1000.0}]},
        "solver": {"method": "fv"},
        "fv": {"linear": "acm", "iterations": 300, "tolerance": 1e-8} | fv,
        "output": {"every": 0},
    })


def test_config_validation():
    with pytest.raises(ValueError, match="backflow"):
        from_dict({"boundaries": {"patches": [{"match": "o", "kind": "pressure",
                                               "backflow_stabilisation": 0.2}]}})
    cfg = from_dict({"source": {"kind": "mesh", "path": "x.msh"}, "solver": {"method": "fv"}})
    assert cfg.fv.linear == "auto"            # no voxel size needed


def test_steady_run(pipe_msh, tmp_path):
    from zvcfd.io.mesh_store import read_boundary_store, read_mesh_store
    from zvcfd.io.node_fields import read_node_fields
    from zvcfd.run import run

    run_dir = run(_config(pipe_msh), out=str(tmp_path), log=lambda *a: None)
    doc = col.read(run_dir)
    assert not col.check(doc)
    kinds = {n["id"]: n["type"] for n in doc["nodes"]}
    assert kinds["mesh"] == "zvcfd:fv-mesh" and kinds["boundary"] == "zvcfd:fv-boundary"
    assert kinds["mesh-collection"] == "zvcfd:mesh"
    paths = {n["id"]: n["path"]["path"] for n in doc["nodes"] if "path" in n}
    assert paths["mesh"].startswith("../meshes/pipe-")        # imported once, beside the runs
    m = read_mesh_store(run_dir / paths["mesh"])
    assert m.unit == "meter" and np.isclose(m.nodes[:, 2].max(), 6e-3)
    snaps = col.snapshots(doc)
    f, meta = read_node_fields(run_dir / snaps[-1]["path"]["path"])
    assert f["velocity"].shape == (m.n_nodes, 3) and meta["time_s"] == 0.0
    rows = {r["patch"]: r for r in csv.DictReader(open(run_dir / "patches.csv"))}
    q_in, q_out = float(rows["inlet"]["flow_m3s"]), -float(rows["outlet"]["flow_m3s"])
    assert abs(q_in - Q) < 1e-12 * Q and abs(q_out - Q) < 1e-6 * Q
    rp, c, rd = RCR_SI
    assert abs(float(rows["outlet"]["pressure_pa"]) - (1000.0 + (rp + rd) * Q)) < 1e-4
    mon = json.loads((run_dir / "monitors.json").read_text())
    assert mon["summary"]["steady"]["converged"]
    b = read_boundary_store(run_dir / "boundary.zarrvectors")
    assert {z["name"] for z in b["zones"]} == {"inlet", "outlet", "wall"}


def test_transient_run(pipe_msh, tmp_path):
    from zvcfd.run import run

    cfg = _config(pipe_msh, dt=0.05, periods=1.0, loops=6, loop_tolerance=1e-8)
    run_dir = run(cfg, out=str(tmp_path), log=lambda *a: None)
    mon = json.loads((run_dir / "monitors.json").read_text())
    steps = [h for h in mon["history"] if h["kind"] == "step"]
    assert len(steps) == 20 and abs(steps[-1]["t"] - 1.0) < 1e-12
    q = np.array([h["flow_m3s"]["inlet"] for h in steps])
    t = np.array([h["t"] for h in steps])
    factor = np.interp(t % 1.0, [0.0, 0.5, 1.0], [1.0, 1.5, 1.0])
    np.testing.assert_allclose(-q, Q * factor, rtol=1e-12)
    import zarr_vectors as zv

    meta = dict(zv.open(str(run_dir / "boundary.zarrvectors")).metadata["zvcfd"])
    assert meta["fields"] == ["osi", "rrt", "tawss"]


def test_partitioned_run_matches(pipe_msh, tmp_path):
    """``parallel.partitions: 3`` on one device: the same patch table as one partition."""
    from zvcfd.run import run

    cfg1 = _config(pipe_msh)
    d = cfg1.as_dict()
    d["parallel"] = {"gpus": 1, "partitions": 3}
    d["fv"]["tolerance"] = 1e-9
    one = run(cfg1, out=str(tmp_path / "one"), log=lambda *a: None)
    three = run(from_dict(d), out=str(tmp_path / "three"), log=lambda *a: None)
    rows = [{r["patch"]: r for r in csv.DictReader(open(p / "patches.csv"))}
            for p in (one, three)]
    for name in ("inlet", "outlet"):
        for key in ("flow_m3s", "pressure_pa"):
            a, b = float(rows[0][name][key]), float(rows[1][name][key])
            assert abs(a - b) <= 1e-6 * abs(a), (name, key, a, b)


def test_method_auto():
    """``solver.method`` defaults to auto: finite volume for meshes, the lattice-Boltzmann
    solver for voxel sources and for meshes given a voxel size; stored resolved."""
    mesh = {"kind": "mesh", "path": "x.msh"}
    assert from_dict({"source": mesh}).solver.method == "fv"
    assert from_dict({"source": mesh | {"voxel_size": 50.0}}).solver.method == "lbm"
    assert from_dict({"source": {"kind": "phantom", "name": "pipe"}}).solver.method == "lbm"
    with pytest.raises(ValueError, match="voxel_size"):           # lbm on a mesh needs voxels
        from_dict({"source": mesh, "solver": {"method": "lbm"}})
    explicit = from_dict({"source": mesh | {"voxel_size": 50.0}, "solver": {"method": "lbm"}})
    implicit = from_dict({"source": mesh | {"voxel_size": 50.0}})
    assert explicit.hash == implicit.hash


def test_yaml_exponents_are_numbers(tmp_path):
    """``1.0e8`` and ``2e-7`` in a YAML config are numbers (PyYAML alone reads them as strings)."""
    pytest.importorskip("yaml")
    from zvcfd.config import load

    p = tmp_path / "c.yaml"
    p.write_text("source: {kind: mesh, path: x.msh}\n"
                 "boundaries:\n  patches:\n"
                 "    - {match: inlet, kind: velocity, flow_rate: 2e-7}\n"
                 "    - {match: outlet, kind: rcr, rcr: [1.0e8, 1.0e-10, 4e8], pressure: 1000}\n")
    rules = load(p).boundaries.patches
    assert rules[0]["flow_rate"] == 2e-7 and rules[1]["rcr"] == [1e8, 1e-10, 4e8]
    assert rules[1]["pressure"] == 1000 and isinstance(rules[1]["pressure"], int)
