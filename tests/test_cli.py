import json

from zvcfd.cli import main


def test_plan(capsys):
    assert main(["plan", "--fluid-cells", "6.3e8", "--method", "lbm-fp32,lbm-fp16"]) == 0
    out = capsys.readouterr().out
    assert "lbm-fp16" in out


def test_plan_dense_json(capsys):
    assert main(["plan", "--shape", "512,512,512", "--fluid-fraction", "0.2", "--layout",
                 "dense", "--geometry", "porous", "--gpus", "1", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["stored_cells"] == 512 ** 3


def test_plan_needs_size(capsys):
    assert main(["plan"]) == 2


def test_phantom_list(capsys):
    assert main(["phantom", "list"]) == 0
    assert "porous-256" in capsys.readouterr().out


def test_probe_json(capsys):
    assert main(["probe", "--json"]) == 0
    assert "cupy" in json.loads(capsys.readouterr().out)


def test_mesh_info_volume(tmp_path, capsys):
    from zvcfd.mesh import write_fluent_mesh
    from zvcfd.mesh.generate import tube

    path = tmp_path / "t.msh"
    write_fluent_mesh(tube(0.5, 2.0, kind="mixed", n_axial=4), path)
    assert main(["mesh-info", str(path), "--volume", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["elements"] == {"tet": 1152, "wedge": 256}
    assert out["dual"]["volume_rel_diff"] < 1e-13
    assert out["couplings"]["rows"] == out["summary"]["nodes"]
    assert out["memory_gb"]["fp64_ilu0"] > out["memory_gb"]["mixed_dilu"] > 0
