import json

import pytest

from zvcfd.config import from_dict, load


def test_defaults_and_hash_are_stable():
    a = from_dict({"name": "x"})
    b = from_dict({"name": "x"})
    assert a.hash == b.hash and len(a.hash) == 12
    assert from_dict({"name": "y"}).hash != a.hash


@pytest.mark.parametrize("bad", [{"nme": 1}, {"solver": {"stps": 3}}, {"solver": {"tau": 0.5}},
                                 {"source": {"kind": "vtk"}}])
def test_strict(bad):
    with pytest.raises(ValueError):
        from_dict(bad)


def test_load_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"name": "demo", "solver": {"steps": 10}}))
    assert load(p).solver.steps == 10
