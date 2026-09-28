# From an Ansys mesh

Collaborators who work in Ansys hand over Fluent meshes. This tutorial
reads one — a HiP-CT coronary lumen meshed in Simpleware ScanIP — and sizes
a voxel run of the same geometry at several resolutions. zvCFD does not
solve on the unstructured cells. It takes the **boundary**: the wall
surface, to voxelise, and the named inlet and outlet zones, to place
boundary conditions.

---

## Summarise the mesh

```bash
zvcfd mesh-info "mesh 1.msh" --voxel-size 0.02,0.01,0.005,0.0025
```

```text
/home/andrew/Downloads/mesh 1.msh
  5,673,949 nodes, 14,790,642 cells, 34,539,621 faces (1,672,260 on the boundary)
  bounding box 58.48 x 58.77 x 66.21 mm; enclosed volume 628.6 mm^3
  pressure-outlet    77 zones     21,411 faces  area 15.07 mm^2
  velocity-inlet      1 zones      1,822 faces  area 1.94 mm^2
  wall                4 zones  1,649,027 faces  area 2269 mm^2
  outlet diameters 0.242 .. 1.12 mm (median 0.417)

     voxel   fluid voxels   box voxels  fluid %  min patch (vox)   GB/GPU  fits   Gupd/s
      0.02       7.86e+07     2.85e+10     0.28             12.1      2.6   yes     48.5
      0.01       6.29e+08     2.28e+11     0.28             24.2     20.6   yes     48.5
     0.005       5.03e+09     1.82e+12     0.28             48.3    164.5    NO     48.5
    0.0025       4.02e+10     1.46e+13     0.28             96.7   1315.9    NO     48.5
```

The reader parses the 1.9 GB ASCII file in about 15 s using 2.8 GB of
memory. It streams the 33 million interior faces past without parsing them
(mmap, C-speed search), and parses only the nodes and the 1.7 million
boundary faces.

Reading the summary:

- **The geometry is 0.28 % of its bounding box.** A dense grid at 20 µm
  would hold 2.85 × 10¹⁰ voxels (4.4 TB in fp32 LBM). The sparse domain
  holds 7.9 × 10⁷ fluid voxels (~22 GB, one H100).
- **Units are millimetres** (inferred: a 58–66 mm box and 0.24–1.6 mm
  vessels are coronary dimensions). Voxel sizes are given in mesh units.
- **The narrowest outlet is 12 voxels across at 20 µm**, and 5 at 50 µm
  (not shown). 20 µm is the coarsest resolution worth running.
- **The 4 wall zones** are the lumen wall (2,269 mm²), two crop planes
  (`xmax`, `ymin`) and one wall-typed opening (`cor_opening_001`). The
  crop planes are where the segmentation was cut, and would be better as
  outlets. That is worth raising with whoever meshed it.
- **At 5 µm the run does not fit** in fp32 (165 GB per GPU at the default
  fill of 0.6). In fp16, at the fill wide vessels achieve at that resolution
  (~0.8), it needs ~62 GB per GPU ([Comparison](../benchmarks/comparison.md)).

The `Gupd/s` column is the same for every row because it is the node's
aggregate update rate; time per step is fluid voxels ÷ rate.

## From Python

```python
from zvcfd.io.fluent_msh import read_fluent_boundary, voxel_estimate

mesh = read_fluent_boundary("mesh 1.msh")
s = mesh.summary()
inlet = next(p for p in s["patches"] if p["kind"] == "velocity-inlet")
print(inlet["name"], round(inlet["area"], 3), round(inlet["equivalent_diameter"], 3))
tris, zone = mesh.triangles()          # fan-triangulated boundary, zone id per triangle
print(tris.shape, len(set(zone.tolist())))
```

```text
from-lumen_bspline_cropped_smoothed_meshmixer-2-to-cor_inlet_001_amira_amiranode_329-velocity-inlet 1.94 1.571
(3339948, 3) 82
```

`triangles()` is the input for voxelisation: every triangle carries its
zone, so wall triangles bound the fluid, and each inlet and outlet zone
becomes a patch of boundary voxels ([Boundary conditions](../spec/boundary_conditions.md)).
Voxelising the surface and mapping zones to patches is milestone 2 on the
[roadmap](../feasibility/roadmap.md). Until then, the fastest route to a
flag volume is the segmentation the mesh was made from.

## Comparing with the Fluent run

The comparison page estimates a pulsatile cardiac cycle at roughly 25 min
for Fluent GPU on this 14.8 M-cell mesh (one H100). zvCFD at 20 µm (5× the
cells, as voxels) is estimated at ~46 min on one H100 or ~7 min on eight.
At 10 µm the estimate is ~1.8 h on eight; a meshed solver would need ~1.2
days on 1,024 cores at that cell count. The same geometry and zones in
both solvers is the validation case of milestone 2.
