# Specification

This section is the reference for what zvCFD writes and reads: the brick
store, the run collection, snapshots and checkpoints, how boundary
conditions are encoded, the parallel I/O contract, how pyramid levels are
used, and the numerical schemes. It serves two audiences:

- **Users** who need to know exactly what is on disk, to read results with
  other tools, or to decide chunk, shard and precision settings.
- **Contributors** who will implement the multi-GPU driver, new boundary
  conditions or readers in other languages.

Each page follows the zarr-vectors specification's structure: a **Terms**
section defining the vocabulary, a plain-English **Introduction**, and a
**Technical reference** with layouts, rules and worked examples.

Words in capitals — MUST, SHOULD, MAY — are used in the RFC 2119 sense.

---

## Design goals

**Storage follows the fluid, not the box.** A vessel tree fills well under
1 % of its scan's bounding box. Only bricks holding fluid are stored, in GPU
memory and on disk.

**One storage format, one set of guarantees.** Every array a run writes is
a Zarr Vectors store written through `zarr_vectors.building`. zvCFD
inherits its layout, codecs, sharding, presence handling and GPU read path
rather than inventing its own.

**Parallel by construction.** A worker owns whole store cells (or whole
shards). No two workers ever write the same file, and no lock is needed
beyond the coordinator's single presence rebuild.

**Atomic publication without a transactional store.** A snapshot becomes
visible in one atomic replace of the run collection's metadata document.

**Images stay where they are.** Input OME-Zarr volumes are referenced by
relative path from the run collection, never copied.

---

## Relationship to upstream specifications

| Specification | Relationship |
|---|---|
| [Zarr Vectors](https://zarr-vectors-py.readthedocs.io/en/latest/spec/index.html) (format 0.9.4) | Brick stores are valid Zarr Vectors stores of custom geometry type `zvcfd:bricks` (spec §12.5, custom geometries). Nothing zvCFD writes needs a zvCFD-aware reader to open. |
| [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) | All arrays are Zarr v3 arrays; the run collection is a Zarr v3 group. |
| [OME-NGFF RFC-8](https://ngff.openmicroscopy.org/rfc/8/index.html) (collections, under review) | The run collection is an RFC-8 `collection` node, with `zvcfd:`-prefixed extension node types. |
| [OME-NGFF 0.5 / 0.6](https://ngff.openmicroscopy.org/) | Input images; units and axes follow NGFF (UDUNITS-2 names, axis order t, c, z, y, x). |
| BRIDGE `COLLECTION_LAYOUT_PLAN.md` | Conventions for ids, relative paths, prefixes and the one-writer rule are BRIDGE's, so a BRIDGE subject and a zvCFD run can sit in one study index. |

---

## Pages

```{toctree}
:maxdepth: 1

brick_store
run_collection
snapshots
boundary_conditions
parallel_io
multiresolution
numerics
mesh_store
fv_numerics
```
