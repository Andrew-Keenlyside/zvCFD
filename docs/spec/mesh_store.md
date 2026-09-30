# Mesh stores

## Terms

**Mesh node**
: A vertex of an unstructured volume mesh. The finite-volume solver keeps
  its unknowns (u, v, w, p) at nodes.

**Element**
: A tetrahedron, pyramid, wedge (prism) or hexahedron. Its nodes are
  listed in the local order of [Local orders](#local-orders), with
  positive volume.

**Element record**
: One row of the store's `links` family: up to six node references.

**Boundary zone**
: A named set of boundary faces with one kind (`wall`,
  `velocity-inlet`, `pressure-outlet`, …), as in a Fluent mesh.

---

## Introduction

The finite-volume solver works on the mesh itself, so a run stores it as
it stores everything else: as Zarr Vectors stores. There are two.

The **volume store** (geometry type `zvcfd:fv-mesh`) holds the nodes as
vertices and the elements as link records. Zarr Vectors chunks vertices
spatially, and elements that span chunks become cross-chunk links, which
it records explicitly. So a store chunk carries a spatial piece of the
mesh, and the same chunks are what GPU partitions will own.

The **boundary store** is an ordinary Zarr Vectors `mesh` store of the
boundary triangles, with one object per zone. Any Zarr Vectors viewer
shows it as a surface. Wall-node results such as wall shear stress are
added to it as vertex attributes.

Neither store needs an addition to Zarr Vectors: the volume store is a
custom geometry type (spec §12.5), and the boundary store is a standard
one.

---

## Technical reference

### Volume store

| Array | dtype | Shape per cell | Content |
|---|---|---|---|
| `vertices` | float64 | (n, 3) | node coordinates, axes `(z, y, x)`, in the mesh unit |
| `vertex_attributes/node` | int64 | (n,) | the node's index in the mesh |
| `links/0` | int64 | records of width 6 | element records (below) |
| `link_attributes/kind` | uint8 | one per record | 0 tet, 1 pyramid, 2 wedge, 3 first half of a hex, 4 second half |
| `link_attributes/element` | int64 | one per record | the element's global index |

Coordinates MUST be double precision, because the control-volume geometry
is computed from them.

**Element records.** zarr-vectors' array link partitioner takes records
of up to six endpoints, so:

- a tetrahedron `(0, 1, 2, 3)` is stored as `(0, 1, 2, 3, 3, 3)`;
- a pyramid as `(0, 1, 2, 3, 4, 4)`;
- a wedge exactly;
- a hexahedron as two records, `(0, 1, 2, 4, 5, 6)` (kind 3) and
  `(0, 2, 3, 4, 6, 7)` (kind 4), with the same `element` value. A reader
  restores `(a0, a1, a2, b2, a3, a4, a5, b5)`.

Links come back in chunk order. A reader MUST sort records by
`(element, kind)` to restore the mesh's element order and to pair hex
halves.

**Global element index.** Elements are numbered by type in the order tet,
pyramid, wedge, hex, then in the mesh's order within each type.

**Metadata** (`zvcfd` namespace): `kind: "fv-mesh"`, `unit`, `chunk` (the
cubic chunk edge, mesh units), `counts` (elements per type), `kinds`,
`width`, `source`.

### Boundary store

A Zarr Vectors `mesh` store (`write_mesh`) of the boundary triangles.
Quadrilateral faces are split along their 0–2 diagonal. Each zone is one
object, and its vertices are duplicated where zones meet.

| Array | Content |
|---|---|
| `vertices` | float64 `(z, y, x)` |
| `links/0` | triangles (width 3) |
| `vertex_attributes/node` | mesh node of each vertex |
| `vertex_attributes/zone` | zone id of each vertex |

The zone table goes in the `zvcfd` namespace as `zones: [{object, zone,
kind, name, faces}]`. To rebuild zones on a volume mesh, a reader matches
the mesh's boundary faces against the stored triangles: a triangle
matches itself, and a quadrilateral matches through one of its four
three-node subsets.

### Local orders

| Type | Nodes | Rule |
|---|---|---|
| tet | 0 1 2 3 | `(x1 − x0) × (x2 − x0) · (x3 − x0) > 0` |
| pyramid | 0 1 2 3 base, 4 apex | base counter-clockwise seen from the apex |
| wedge | 0 1 2 bottom, 3 4 5 top | 3, 4, 5 above 0, 1, 2; bottom counter-clockwise from above |
| hex | 0 1 2 3 bottom, 4 5 6 7 top | 4–7 above 0–3; bottom counter-clockwise from above |

These are VTK's orders for all four types. `zvcfd.mesh.core.FACES` lists
each type's faces with outward normals, and `EDGES` its edges.

### Reading Fluent meshes

`zvcfd.mesh.read_fluent_mesh` reads nodes, every face zone and the zone
table, then rebuilds each cell's element from its faces. It classifies
cells by their triangle and quadrilateral face counts (tet 4 + 0, pyramid
4 + 1, wedge 2 + 3, hex 0 + 6) and orders nodes by the geometry. It
therefore ignores the declared element type (Simpleware writes 7,
polyhedral, for tetrahedra and prisms) and the file's orientation
convention, which it measures instead (`meta["c0_side"]`). ASCII and
binary sections are both read; a mixed binary face section is walked by
pointer doubling. Polygonal faces (more than four nodes) are not
supported.

Measured on the HiP-CT coronary mesh (1.9 GB, ASCII; `benchmarks/fv/mesh_report.py`):

| | |
|---|---|
| Nodes | 5,673,949 |
| Elements | 6,546,228 tetrahedra, 8,244,327 wedges, 87 pyramids (the declared 14,790,642 cells) |
| Faces | 34,539,621, conforming (each interior face shared by two elements with opposite orientations) |
| Boundary | 1,672,260 faces in 82 zones: 4 walls, 77 outlets, 1 inlet |
| Orientation | right-hand normal points into `c0` on every interior face |
| Read time | 53 s on an idle machine (143–191 s when it was shared with other jobs); 11.3 GB peak host memory |

`write_fluent_mesh` writes the same format back (ASCII or binary), so
generated validation meshes can go to other codes.
