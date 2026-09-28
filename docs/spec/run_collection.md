# Run collections

## Terms

**Run collection**
: A directory `<name>-<config hash>.zvcfd/` that is a Zarr v3 group whose
  `attributes.ome` is an OME-NGFF RFC-8 `collection` document.

**Node**
: An entry of the document's `nodes` list: an object with `type`, `id`,
  `name` and exactly one of `path` (a leaf) or `nodes` (a sub-collection).

**Prefixed type**
: A node or path type outside RFC-8's core set, written `zvcfd:<name>`.
  Readers that do not know the prefix treat the node as opaque.

**Coordinator**
: The one process allowed to write the document.

---

## Introduction

The run collection is the table of contents of a run. It names the input
image, the domain, every published snapshot and checkpoint, and the
resolved configuration, all by path relative to itself. Nothing large lives
in the document: arrays live in the stores it points at. Because RFC-8
allows collections to reference other collections, a study index (such as
BRIDGE's `derivatives/BRIDGE/collection.json`) can list zvCFD runs next to
the BRIDGE subjects they were computed from.

RFC-8 is a draft under review. zvCFD writes `"version": "0.6"` with
`"type": "collection"`, exactly as BRIDGE does today, and will change when
BRIDGE does. All document writing lives in `zvcfd.collection`.

---

## Technical reference

### Document

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "ome": {
      "version": "0.6",
      "type": "collection",
      "id": "zvc-run-8e7437307dd4",
      "name": "network-demo",
      "attributes": {
        "zvcfd:run": {"config_hash": "78c04469a293", "voxel_size_um": 10.0,
                      "dt_s": 4.76e-06, "solver": "lbm"},
        "scene": {
          "coordinateSystems": [{"id": "physical", "name": "physical",
            "axes": [{"name": "z", "type": "space", "unit": "micrometer"},
                     {"name": "y", "type": "space", "unit": "micrometer"},
                     {"name": "x", "type": "space", "unit": "micrometer"}]}],
          "coordinateTransformations": []
        }
      },
      "nodes": [
        {"type": "multiscale", "id": "image", "name": "image",
         "path": {"type": "zarr", "path": "../../scans/sub-01_XPCT.ome.zarr"}},
        {"type": "zvcfd:domain", "id": "domain", "name": "domain",
         "path": {"type": "zarr", "path": "./domain.zarrvectors"},
         "attributes": {"zvcfd:domain": {"shape": [128, 128, 512], "brick": 8,
                        "bricks": 4989, "fluid_cells": 1192679, "fill": 0.467}}},
        {"type": "zvcfd:config", "id": "config", "name": "config",
         "path": {"type": "json", "path": "./config.json"}},
        {"type": "collection", "id": "fields", "name": "fields", "nodes": [
          {"type": "zvcfd:fields", "id": "step-000005000", "name": "step-000005000",
           "path": {"type": "zarr", "path": "./fields/step-000005000.zarrvectors"},
           "attributes": {"zvcfd:step": 5000, "zvcfd:time_s": 0.0238,
                          "zvcfd:fields": ["rho", "ux", "uy", "uz"]}}
        ]}
      ]
    }
  }
}
```

(Taken from `examples/network_demo.yaml`, with an `image` node added to
show the reference form.)

### Node types

| `type` | `path.type` | Target | Required |
|---|---|---|---|
| `multiscale` | `zarr` | the input OME-Zarr image | when the source is an image |
| `multiscale` (with `labels.source`) | `zarr` | the segmentation, when it is not the image | optional |
| `zvcfd:domain` | `zarr` | the domain brick store (flags) | MUST |
| `zvcfd:config` | `json` | the resolved `RunConfig` | MUST |
| `collection` id `fields` | — | sub-collection of `zvcfd:fields` snapshot nodes | when any snapshot exists |
| `collection` id `checkpoints` | — | sub-collection of `zvcfd:checkpoint` nodes | when any checkpoint exists |
| `zvcfd:boundary` | `zarr` | ZV `mesh`/`graph` stores: wall surface, patches, centrelines | optional |

### Rules

1. **Ids.** Node ids and coordinate-system ids share one namespace and MUST
   be unique in the document. The run id is `zvc-run-<12 hex>`. Snapshot
   ids are `step-<9-digit step>`, so sorting ids sorts by step.
2. **Paths.** Paths are relative to the document's directory (RFC 1808):
   `./…` inside the run, `../…` to climb out. Input images are referenced,
   never copied.
3. **Existence.** A node MUST NOT be added before its target is complete on
   disk (after its store's `finalize`).
4. **One writer.** Only the coordinator writes the document.
5. **Atomic replace.** Every write is write-to-temporary, `fsync`,
   `os.replace` in the same directory, so readers see the old document or
   the new one, never a mixture (`zvcfd.collection.write`).
6. **Validation.** `zvcfd.collection.check` rejects duplicate ids, nodes
   with both or neither of `nodes`/`path`, and unknown versions; `write`
   refuses a document that fails it.

### Placement next to BRIDGE outputs

Following BRIDGE's BIDS-like layout, a run on a BRIDGE subject goes under
the subject's derivatives:

```text
derivatives/zvCFD/sub-01/micr/<acquisition>/runs/<name>-<hash>.zvcfd/
```

and references the subject's image and mask with `../` paths, so the study
index can collect runs as it collects BRIDGE collections.
