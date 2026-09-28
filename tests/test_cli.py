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
