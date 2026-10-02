# From an Ansys mesh

Collaborators who work in Ansys hand over Fluent meshes. This tutorial
reads one — a HiP-CT coronary lumen meshed in Simpleware ScanIP — sizes a
voxel run of the same geometry at several resolutions, then voxelises and
runs it. zvCFD does not
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
      0.02       7.86e+07     2.85e+10     0.28             12.1      2.6   yes     52.6
      0.01       6.29e+08     2.28e+11     0.28             24.2     20.6   yes     52.6
     0.005       5.03e+09     1.82e+12     0.28             48.3    164.5    NO     52.6
    0.0025       4.02e+10     1.46e+13     0.28             96.7   1315.9    NO     52.6
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
(1683184, 3) 82
```

`triangles()` is the input for voxelisation. Every triangle carries its
zone, so wall triangles bound the fluid, and each inlet and outlet zone
becomes a patch of boundary voxels ([Boundary conditions](../spec/boundary_conditions.md)).

## Voxelise and run

`examples/coronary_50um.yaml` runs the mesh as it stands: steady flow of
blood as a Newtonian fluid, 0.194 mL/s into the inlet (a mean of 0.1 m/s,
Re ≈ 45), and 0 Pa at every outlet. Patch rules match zone names by
substring, and the inlet's flow rate is held by flow control on the
measured patch flux.

```yaml
name: coronary-50um
source: {kind: mesh, path: "/home/andrew/Downloads/mesh 1.msh", unit: mm, voxel_size: 50.0}
physics: {nu: 3.5e-6, rho: 1060.0}
boundaries:
  patches:
    - {match: inlet, kind: velocity, flow_rate: 1.9396e-7}
    - {match: outlet, kind: pressure, pressure: 0.0}
solver: {collision: trt, tau: 0.6, steps: 200000, check_every: 2000, tolerance: 1.0e-4}
domain: {chunk_bricks: 16}
output: {path: runs, every: 0, fields: [rho, ux, uy, uz]}
```

```bash
zvcfd run examples/coronary_50um.yaml
```

```text
domain (1336, 1184, 1176): 21849 bricks (0.60% active), 5,028,753 fluid cells, fill 0.45; 78 patches; tau = 0.6000, dt = 2.38e-05 s  (15.3 s)
  step     2000  inflow 1.8772e-07 m3/s  imbalance +3.00e-01  change inf
  step     4000  inflow 1.9255e-07 m3/s  imbalance +4.15e-02  change 3.72e-02
  ...
  step    16000  inflow 1.9394e-07 m3/s  imbalance +5.96e-05  change 9.02e-05
  converged: change <= 0.0001, |imbalance| <= 0.001
  wrote step-000016000.zarrvectors
done: 16000 steps, 727 MLUPS incl. checks/output, 128.1 s total -> runs/coronary-50um-b1ee088e1393.zvcfd
```

`zvcfd run` does not voxelise the Fluent file itself. On first use it
imports the file's boundary into a Zarr Vectors mesh collection under
`runs/meshes/` (27 s), and it voxelises that collection's boundary store
([Meshes in Zarr Vectors](zarr_vectors_meshes.md)). Later runs reuse the
collection, and the result is byte for byte the same as a direct
voxelisation. The output above is from a direct voxelisation.

Voxelising takes 15 s. The solve takes under two minutes on one RTX
A2000, which is 0.38 s of flow at a time step of 23.8 µs. Convergence
requires two things: the flows must stop changing (10⁻⁴ between checks),
and inflow must equal outflow (10⁻³).

Every patch's flow, share and mean pressure are in `patches.csv`:

```python
from zvcfd.run import load_patch_table

rows = load_patch_table("runs/coronary-50um-b1ee088e1393.zvcfd/patches.csv")
inlet = next(r for r in rows if r["kind"] == "velocity")
print(f"inlet {float(inlet['flow_m3s']) * 1e6:.4f} mL/s at {float(inlet['pressure_pa']):.1f} Pa")
outlets = sorted((r for r in rows if r["kind"] == "pressure"), key=lambda r: -float(r["split"]))
for r in outlets[:4]:
    print(f"{r['patch'].split('-to-')[1][:14]}  {100 * float(r['split']):5.2f} %  {r['cells']:>4} voxels")
print(f"{len(outlets)} outlets, total {sum(-float(r['flow_m3s']) for r in outlets) * 1e6:.4f} mL/s")
```

```text
inlet 0.1940 mL/s at 62.7 Pa
cor_outlet_074  31.87 %   553 voxels
cor_outlet_029  10.28 %   383 voxels
cor_outlet_022   7.82 %   319 voxels
cor_outlet_035   7.60 %   259 voxels
77 outlets, total 0.1940 mL/s
```

For an outlet, `cells` counts the voxels whose lattice links cross the
outlet's cap: the outlet's pressure is imposed on those links, at the cap
([Boundary conditions](../spec/boundary_conditions.md#pressure-outlets-on-caps-patch-links)).

At 35 µm (`examples/coronary_35um.yaml`: 14.7 M fluid voxels, about the
Ansys mesh's own cell count), the run takes 20,000 steps and 6.6 min. The
splits move by 0.004 percentage points on average and by 0.06 at most;
the dominant outlet takes 31.8 %.

## Comparing with other solvers

The same case on the same workstation, with OpenFOAM on the body-fitted
14.8 M-cell mesh, is in the [OpenFOAM comparison](../benchmarks/openfoam.md).
For Ansys, the comparison page estimates a pulsatile cardiac cycle on this
mesh at 1.3–5.3 h for CFX on 128 cores, and 6–25 min for Fluent's GPU
solver on one H100 (the ranges span 5–20 iterations per time step). zvCFD
at 20 µm (5× the cells, as voxels) is estimated at ~42 min on one H100 or
~6 min on eight. At 10 µm the estimate is ~1.7 h on eight; a meshed solver
would need 7 h to 1.2 days on 1,024 cores at that cell count.
