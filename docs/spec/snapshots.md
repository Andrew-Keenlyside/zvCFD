# Snapshots and checkpoints

## Terms

**Snapshot**
: A brick store of macroscopic fields at one time step, published under
  the run collection's `fields` node.

**Checkpoint**
: A brick store of the lattice populations at one time step: enough to
  restart bit-for-bit. Published under `checkpoints`.

**Publication**
: The atomic replace of the run collection document that adds a snapshot's
  or checkpoint's node.

---

## Introduction

Time is the one dimension Zarr Vectors does not chunk for fields: a store
has resolution levels, not time steps. zvCFD therefore writes **one store
per saved step** and lets the run collection be the time index. A saved
step is written in full, finalised, and only then published. A reader
following the collection never sees a half-written step. A crash leaves at
most an unpublished store, which the next run can delete. This is the same
guarantee a transactional store gives, obtained with one atomic file
replace.

---

## Technical reference

### Snapshot stores

Path: `fields/step-<9-digit step>.zarrvectors`, a [brick store](brick_store.md)
with fields chosen by `output.fields` (default `rho`, `ux`, `uy`, `uz`) in
`output.dtype` (default `float32`; `float16` halves the size).

| Field | Units in the store | Physical conversion |
|---|---|---|
| `rho` | lattice density | `p − p₀ = (rho − 1)·c_s²·ρ_phys·(dx/dt)²`, `c_s² = 1/3` |
| `ux`, `uy`, `uz` | lattice velocity | `u = u_lattice · dx/dt` |

`dx`, `dt` and the physical density are in the collection's `zvcfd:run`
attributes and the configuration node; `zvcfd.Lattice` performs the
conversions.

The node's attributes MUST carry `zvcfd:step` and `zvcfd:time_s`, and
SHOULD carry `zvcfd:fields`.

### Checkpoint stores

Path: `checkpoints/step-<9-digit step>.zarrvectors`, a brick store with one
attribute per population, `f00` … `f18`. Each is stored in the solver's own
population precision (`float32`, or `float16` offset by the rest weight
`w_q`, exactly as held on the device) so that a restart reproduces the run
bit-for-bit. The node's attributes MUST record `zvcfd:precision` and the
collision parameters. *(Designed; not yet built.)*

### Publication protocol

```text
coordinator                          workers (one per GPU)
───────────                          ─────────────────────
create_brick_store(path)  ───────▶
                                     write_brick_chunks(own chunks)   # no lock
                          ◀───────   done
finalize_brick_store(path)           # one presence rebuild
add_snapshot(doc, path, step)
collection.write(run, doc)           # atomic replace: now visible
```

### Retention

Snapshots are kept unless deleted. Deleting a snapshot means removing its
node (one atomic replace), then its directory. Checkpoints default to the
last two. A checkpoint is deleted only after a newer one is published.

### Why not one store with a time axis?

Zarr Vectors supports a `time`-typed axis only as a chunked, binned
coordinate of the vertices, like space. Fields would then need a vertex per
brick per step, duplicating positions and interleaving steps in cells. One
store per step keeps each step's cells contiguous and deletable, and makes
publication atomic.
