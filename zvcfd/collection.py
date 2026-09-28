"""A run as an OME-NGFF RFC-8 collection.

A run directory ``<name>.zvcfd/`` is a Zarr v3 group whose
``attributes.ome`` is an RFC-8 collection document. Its nodes point at
everything the run reads and writes, by relative path:

- ``image``: the input OME-Zarr (``multiscale``), referenced where it
  lives and never copied;
- ``mask``: the segmentation the domain was built from (``multiscale``,
  with ``labels.source`` pointing at ``image``), when it is not the image;
- ``domain``: the brick store holding voxel flags (``zvcfd:domain``);
- ``fields``: a ``collection`` of snapshot brick stores, one per saved
  step (``zvcfd:fields``);
- ``boundary``: surfaces, patches and centrelines as Zarr Vectors stores
  (``zvcfd:boundary``), e.g. imported from an Ansys ``.msh``;
- ``config``: the resolved run configuration (``json``).

Conventions shared with BRIDGE (``COLLECTION_LAYOUT_PLAN.md``): version
``"0.6"`` with ``type: "collection"``; one namespace for node ids and
coordinate-system ids; paths relative to the document with ``./`` inside
the container; extensions prefixed ``zvcfd:``; **one writer** per
document (the coordinator), and every write an atomic replace. Publishing
a snapshot is adding its node in one such replace, which is what makes a
snapshot visible atomically without a transactional store (``docs/spec/
snapshots.md``). RFC-8 is still a draft ("0.x" in its examples); the
version string follows BRIDGE and will move with it.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

OME_VERSION = "0.6"
PREFIX = "zvcfd"
ZARR_JSON = "zarr.json"


def new_id(kind: str = "run") -> str:
    return f"zvc-{kind}-{uuid.uuid4().hex[:12]}"


def rel(target: str | os.PathLike, doc_dir: str | os.PathLike) -> str:
    """Relative path from the document's directory, ``./``-prefixed inside it."""
    r = os.path.relpath(Path(target).resolve(), Path(doc_dir).resolve())
    return r if r.startswith("..") else f"./{r}"


def physical_system(unit: str = "micrometer") -> dict:
    return {"id": "physical", "name": "physical",
            "axes": [{"name": n, "type": "space", "unit": unit} for n in ("z", "y", "x")]}


def run_document(run_dir: str | os.PathLike, *, name: str, unit: str = "micrometer",
                 attributes: dict | None = None, run_id: str | None = None) -> dict:
    """A new, empty run collection document."""
    return {
        "version": OME_VERSION,
        "type": "collection",
        "id": run_id or new_id(),
        "name": name,
        "attributes": {
            f"{PREFIX}:run": dict(attributes or {}),
            "scene": {"coordinateSystems": [physical_system(unit)],
                      "coordinateTransformations": []},
        },
        "nodes": [],
    }


def node(kind: str, name: str, path: str, *, path_type: str = "zarr", node_id: str | None = None,
         attributes: dict | None = None) -> dict:
    n = {"type": kind, "id": node_id or name, "name": name,
         "path": {"type": path_type, "path": path}}
    if attributes:
        n["attributes"] = attributes
    return n


def add_node(doc: dict, n: dict) -> dict:
    """Add or replace (by id) a top-level node. Ids share one namespace with coordinate systems."""
    ids = {c["id"] for c in doc["attributes"]["scene"]["coordinateSystems"]}
    if n["id"] in ids:
        raise ValueError(f"node id {n['id']!r} collides with a coordinate system id")
    doc["nodes"] = [m for m in doc["nodes"] if m.get("id") != n["id"]] + [n]
    return doc


def add_image(doc: dict, run_dir, image_path, *, name: str = "image") -> dict:
    return add_node(doc, node("multiscale", name, rel(image_path, run_dir)))


def add_domain(doc: dict, run_dir, store_path, *, summary: dict | None = None) -> dict:
    return add_node(doc, node(f"{PREFIX}:domain", "domain", rel(store_path, run_dir),
                              attributes={f"{PREFIX}:domain": summary or {}}))


def add_snapshot(doc: dict, run_dir, store_path, *, step: int, time_s: float | None = None,
                 fields: list[str] | None = None) -> dict:
    """Publish one snapshot under the ``fields`` collection node."""
    fields_node = next((n for n in doc["nodes"] if n.get("id") == "fields"), None)
    if fields_node is None:
        fields_node = {"type": "collection", "id": "fields", "name": "fields", "nodes": []}
        doc["nodes"].append(fields_node)
    sid = f"step-{step:09d}"
    snap = node(f"{PREFIX}:fields", sid, rel(store_path, run_dir),
                attributes={f"{PREFIX}:step": step, f"{PREFIX}:time_s": time_s,
                            f"{PREFIX}:fields": fields or []})
    fields_node["nodes"] = [m for m in fields_node["nodes"] if m["id"] != sid] + [snap]
    fields_node["nodes"].sort(key=lambda m: m["id"])
    return doc


def snapshots(doc: dict) -> list[dict]:
    f = next((n for n in doc["nodes"] if n.get("id") == "fields"), None)
    return [] if f is None else list(f["nodes"])


def check(doc: dict) -> list[str]:
    """Structural problems (empty when fine): unique ids, nodes xor path, known version."""
    problems = []
    seen = {c["id"] for c in doc.get("attributes", {}).get("scene", {})
            .get("coordinateSystems", [])}

    def walk(n, where):
        nid = n.get("id")
        if nid is not None:
            if nid in seen:
                problems.append(f"{where}: duplicate id {nid!r}")
            seen.add(nid)
        if ("nodes" in n) == ("path" in n):
            problems.append(f"{where}: needs exactly one of nodes/path")
        for i, c in enumerate(n.get("nodes", [])):
            walk(c, f"{where}/{c.get('id', i)}")

    if doc.get("version") != OME_VERSION:
        problems.append(f"version {doc.get('version')!r} != {OME_VERSION!r}")
    for i, n in enumerate(doc.get("nodes", [])):
        walk(n, n.get("id", str(i)))
    return problems


def write(run_dir: str | os.PathLike, doc: dict) -> Path:
    """Atomically replace the run group's ``zarr.json`` with ``doc`` as ``attributes.ome``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    problems = check(doc)
    if problems:
        raise ValueError("invalid collection: " + "; ".join(problems))
    body = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": doc}}
    return atomic_write_json(run_dir / ZARR_JSON, body)


def read(run_dir: str | os.PathLike) -> dict:
    with open(Path(run_dir) / ZARR_JSON) as fh:
        return json.load(fh)["attributes"]["ome"]


def atomic_write_json(path: Path, obj: Any) -> Path:
    """Write-then-rename in the same directory, so readers see old or new, never half."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path
