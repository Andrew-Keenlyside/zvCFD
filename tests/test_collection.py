import pytest

from zvcfd import collection as col


def test_document_round_trip(tmp_path):
    run = tmp_path / "demo.zvcfd"
    doc = col.run_document(run, name="demo", attributes={"k": 1})
    col.add_image(doc, run, tmp_path / "scan.ome.zarr")
    col.add_domain(doc, run, run / "domain.zarrvectors")
    for step in (200, 100):
        col.add_snapshot(doc, run, run / "fields" / f"step-{step:09d}.zarrvectors", step=step)
    col.write(run, doc)
    back = col.read(run)
    assert back == doc
    assert [n["id"] for n in col.snapshots(back)] == ["step-000000100", "step-000000200"]
    image = next(n for n in back["nodes"] if n["id"] == "image")
    assert image["path"]["path"] == "../scan.ome.zarr"
    assert next(n for n in back["nodes"] if n["id"] == "domain")["path"]["path"] == \
        "./domain.zarrvectors"
    assert col.check(back) == []


def test_ids_share_a_namespace_with_coordinate_systems(tmp_path):
    doc = col.run_document(tmp_path, name="d")
    with pytest.raises(ValueError):
        col.add_node(doc, col.node("zvcfd:config", "physical", "./x.json", path_type="json"))


def test_invalid_document_is_not_written(tmp_path):
    doc = col.run_document(tmp_path, name="d")
    doc["nodes"].append({"type": "collection", "id": "bad"})     # neither nodes nor path
    with pytest.raises(ValueError):
        col.write(tmp_path / "r", doc)
