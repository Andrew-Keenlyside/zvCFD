# Meshes in Zarr Vectors

Every mesh run in zvCFD starts from Zarr Vectors. A mesh from any tool
(Ansys Fluent, SimVascular, Gmsh, …) is imported once into a **mesh
collection**. Both GPU solvers then take their geometry from its stores
and never read the original file:

- the **finite-volume** solver (custom CUDA kernels, AmgX) solves on the
  nodes, elements and zones read back from the collection's volume store;
- the **lattice-Boltzmann** solver (custom CUDA kernels) voxelises the
  collection's boundary store.

Results go back into Zarr Vectors, in the run collection. So the
geometry, the solver's working mesh and the results all share one
chunked, spatially indexed format. This page shows the workflow on the
two meshes of the validation studies, and what it costs.

---

## Import a mesh

```bash
zvcfd import-mesh "mesh 1.msh" --out coronary.zvmesh --unit mm
```

```text
read mesh 1.msh: 5,673,949 nodes, {'tet': 6546228, 'pyramid': 87, 'wedge': 8244327}, 82 zones (64.3 s)
wrote coronary.zvmesh (70.5 s)
  5,673,949 nodes, {'tet': 6546228, 'pyramid': 87, 'wedge': 8244327}; chunk 16.5526 mm; box [58.481, 58.765, 66.21] mm
  zone    3 wall                     4 faces  from-lumen_…-to-xmax-wall
  …
  zone   84 velocity-inlet       1,822 faces  from-lumen_…-to-cor_inlet_001_amira_amiranode_329-velocity-inlet
```

The 1.9 GB ASCII Fluent mesh becomes a 1.05 GB collection:

```bash
zvcfd info coronary.zvmesh
```

```text
mesh collection coronary.zvmesh: 5,673,949 nodes, {'tet': 6546228, 'pyramid': 87, 'wedge': 8244327}, 82 zones, chunk 0.0165526 m; from /home/andrew/Downloads/mesh 1.msh
  zvcfd:fv-mesh      volume     ./volume.zarrvectors
  zvcfd:fv-boundary  boundary   ./boundary.zarrvectors
```

The volume store keeps the nodes as vertices, chunked in 16.6 mm cubes,
and the elements as link records. The 127,983 elements that straddle a
chunk boundary (0.9 %) are stored as explicit cross-chunk links. The
boundary store is an ordinary Zarr Vectors surface mesh, one object per
zone, so any Zarr Vectors viewer shows the tree with its 77 outlet caps
([Mesh stores](../spec/mesh_store.md#mesh-collections)).

The voxel solver needs only the boundary. `--surface-only` reads just the
Fluent boundary zones and writes a 72 MB collection in 27 s (22 s to read,
5 s to write).

You do not have to import by hand. When `source.path` in a run
configuration is a raw mesh, `zvcfd run` imports it into
`<output.path>/meshes/` on first use and reuses the collection afterwards.

## The lattice-Boltzmann solver from the collection

Point `source.path` at the collection. It is already in metres, so
`unit` is not needed:

```yaml
name: coronary-50um-zvmesh
source: {kind: mesh, path: coronary.zvmesh, voxel_size: 50.0}
physics: {nu: 3.5e-6, rho: 1060.0}
boundaries:
  patches:
    - {match: inlet, kind: velocity, flow_rate: 1.9396e-7}
    - {match: outlet, kind: pressure, pressure: 0.0}
solver: {collision: trt, tau: 0.6, steps: 200000, check_every: 2000, tolerance: 1.0e-4}
domain: {chunk_bricks: 16}
output: {path: runs, every: 0, fields: [rho, ux, uy, uz]}
```

```text
voxelised coronary.zvmesh (Zarr Vectors boundary store) at 50 um: 17.5 s
domain (1336, 1184, 1176): 21849 bricks (0.60% active), 5,028,753 fluid cells, fill 0.45; 78 patches; tau = 0.6000, dt = 2.38e-05 s  (17.5 s)
  step     2000  inflow 1.8760e-07 m3/s  imbalance +2.33e-01  change inf
  ...
  step    18000  inflow 1.9395e-07 m3/s  imbalance -4.44e-06  change 5.70e-05
  converged: change <= 0.0001, |imbalance| <= 0.001
  wrote step-000018000.zarrvectors
done: 18000 steps, 627 MLUPS incl. checks/output, 218.7 s total -> runs/coronary-50um-zvmesh-13b339edcd0d.zvcfd
```

The collection is a lossless intermediate. The run is the same run as
one that voxelises the Fluent file directly:

| Compared with the direct voxelisation | |
|---|---|
| Domain store (531 files): flags, bricks, patch cells and links | byte-identical, apart from `zarr.json` (run id, timestamps) |
| Velocity and density snapshot at step 18,000 (1,059 files) | byte-identical, apart from `zarr.json` |
| Flow and mean pressure at all 78 patches | identical |
| Patch areas | agree to 4 × 10⁻¹⁴ (mm → m rounding) |

`tests/test_mesh_collection.py` checks the same equality voxel for voxel
on a small mesh. The voxel geometry is computed in voxel units, so the
metre coordinates of the store change nothing. Reading and voxelising the
store takes 17.5 s here, against 15 s to voxelise the Fluent file
directly, which also has to be parsed first (22 s for its boundary).

The run collection links the mesh collection it came from (node
`mesh-collection`, and `zvcfd:run.mesh_collection` with the grid origin
in metres), so the voxel domain can be placed back on the mesh.

## The finite-volume solver from the collection

The finite-volume solver takes the whole mesh. `zvcfd run` with
`solver.method: fv` reads nodes, elements and zones back from the
collection, maps zones to boundary conditions, and hands the mesh to the
GPU solver. [Solving on the mesh itself](fv_mesh.md) walks through a
complete run on a pipe.

Reading the coronary collection back takes 58 s: 10 s of Zarr I/O,
the rest rebuilding element order and boundary zones. Parsing the Fluent
file takes 64–172 s, depending on the machine's load. The mesh that comes
back is the Fluent mesh exactly: the same node coordinates and element
connectivity, and in every one of the 82 zones the same faces with the
same owning elements and orientation. Only the order of faces within a
zone may differ.

Snapshots are written as node fields that point at the collection's
volume store as their mesh, so the mesh is stored once. Wall shear
statistics go into the run's own boundary store as vertex attributes.
The full coronary mesh needs about 25 GB of GPU memory in mixed precision
(`zvcfd mesh-info --volume`), which is an H100 job.

On the workstation, the real-data case is the SimVascular comparison
([SimVascular](../validation/simvascular.md)). SimVascular's own
tetrahedral mesh, cropped to the two coronary trees, is written as a mesh
collection, and the CUDA solver steps the cardiac cycle on the mesh read
back from it (`benchmarks/simvascular/run_fv.py`):

| SimVascular case, from `coronary.zvmesh` | |
|---|---|
| Mesh | 398,105 nodes, 2,089,669 tetrahedra, 27 zones |
| Write / read the collection | 17.5 s / 8.4 s |
| Read-back check | nodes, elements and zones identical to the cropped mesh, else the run stops |
| Time step (BDF2, 1 ms, 5 coefficient loops) | 4.9 s on the RTX A2000; about 8.4 GB of GPU memory |
| One cardiac cycle | 1,000 steps in 81 min; outlet flows within 0.5 % of SimVascular on average, velocity within 1.3–1.8 % |

## From Python

```python
from zvcfd.io.mesh_collection import import_mesh, read_mesh, read_surface, voxelize_collection

zvm = import_mesh("model/mesh-complete", "model.zvmesh", unit="cm")   # a SimVascular folder
mesh = read_mesh(zvm)                     # UnstructuredMesh in metres, with zones
tri, zone, zones = read_surface(zvm)      # boundary triangles (T, 3, 3), zone per triangle
domain, boundary, grid, report = voxelize_collection(zvm, 60e-6)      # 60 µm voxels
```

`read_source` reads any supported mesh without writing anything. With
[meshio](https://github.com/nschloe/meshio) installed, Gmsh, Abaqus,
Exodus, MED and Nastran meshes import too. Tagged boundary faces (Gmsh
physical groups, cell sets) become zones, and any untagged boundary face
goes into zone `wall`.
