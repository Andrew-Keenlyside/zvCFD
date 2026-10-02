# Solving on the mesh itself

The voxel solver in [From an Ansys mesh](ansys_mesh.md) takes only the
boundary of a mesh. The finite-volume solver solves on the mesh's own
cells, in the manner of Ansys CFX: unknowns at the nodes, pressure and
velocity coupled, algebraic multigrid on the GPU
([Finite-volume numerics](../spec/fv_numerics.md)). This tutorial runs it
on a small pipe: import the mesh into a Zarr Vectors collection, run one
cardiac-like cycle with a flow waveform at the inlet and an RCR outlet,
and read the results.

Every output shown comes from a real run on an RTX A2000 workstation.

---

## A mesh

Any Fluent `.msh`, `.vtu` or SimVascular `mesh-complete` folder will do.
Here a pipe of 1 mm radius and 6 mm length, built like the coronary mesh
(wedge layers at the wall around a tetrahedral core), written as a Fluent
mesh in millimetres:

```python
from zvcfd.mesh.fluent import write_fluent_mesh
from zvcfd.mesh.generate import tube

m = tube(1.0, 6.0, n_core=6, n_ring=6, n_axial=24, kind="mixed", layers=3, growth=0.8)
write_fluent_mesh(m, "pipe.msh")
```

## Import it

Both solvers run from a mesh collection: a Zarr Vectors volume store and a
boundary store, in metres ([Mesh stores](../spec/mesh_store.md)).

```console
{{import}}
```

`zvcfd run` imports a raw mesh by itself (and caches the collection), so
this step is optional; doing it once is faster for a mesh you will run
many times.

## The configuration

`examples/pipe_fv.yaml`:

```yaml
{{config}}
```

- `boundaries.patches` maps zones to conditions by name, as for the voxel
  solver. The inlet takes a flow rate with a waveform (a periodic table of
  multipliers) and the fully developed profile of the inlet face
  (`profile: parabolic`). The outlet is a three-element Windkessel, coupled
  implicitly: its pressure is `a + r Q` inside every linear solve, so many
  outlets with large resistances stay stable.
- `solver.method: fv` selects the finite-volume solver; its settings are
  in `fv`. `linear: auto` uses AmgX if it is installed (with fallbacks for
  hard systems), else zvCFD's own multigrid. `dt` and `periods` make the
  run transient, after a steady start at `t = 0`.
- Pressure outlets get backflow stabilisation by default (β = 0.2, as in
  SimVascular).

## Run it

```console
{{run}}
```

The steady start converges in {{steady_its}} outer iterations. Then each
time step runs up to five coefficient loops. The run collection holds the
mesh collection's link, node-field snapshots every 25 steps, the monitors
and the patch table.

## Read the results

The patch table has each zone's flow (into the domain) and mean pressure
at the end of the run:

```console
{{patches}}
```

Snapshots are Zarr groups of node arrays in the mesh's node order:

```python
{{read}}
```

After a transient run, the time statistics of wall shear stress (TAWSS,
OSI, RRT over the last period) are vertex attributes of the boundary
store, so a Zarr Vectors viewer shows them on the wall.

## On a real mesh

`examples/coronary_fv.yaml` is the same set-up for the HiP-CT coronary
tree (14.8 M cells, 77 outlets). Its block matrix has 106 M blocks, so it
needs an H100 ([plan](../feasibility/fv_plan.md#progress)). Settings that
matter at that scale:

| Key | Use |
|---|---|
| `fv.precision: mixed` | the AmgX hierarchy in single precision: half its memory, the solution still double |
| `fv.residual_target: 1.0e-5` | stop outer iterations (and a time step's loops) on the RMS residual, as CFX does |
| `parallel.partitions`, `parallel.gpus` | the partitioned solver, one partition per GPU |
| `fv.linear: simple` | the block preconditioner, for meshes whose thin prism layers stall the coupled AMG (`auto` switches to it by itself) |
