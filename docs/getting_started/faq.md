# Frequently asked questions

## General

### Why two solvers?

Because blood-flow geometries arrive in two forms. A mesh, from Ansys,
SimVascular or Gmsh, is best solved on its own cells. The finite-volume
solver does that, in the manner of Ansys CFX: body-fitted walls, prism
layers where the boundary layer needs them, and wall shear stress taken
on the true wall. It is the main solver. An image segmentation with
10⁸–10¹⁰ fluid voxels is a different case: meshing it is a large, fragile
job, and the mesh and its matrix (1–5.6 GB per million cells) would not
fit on one node. There the lattice-Boltzmann solver works on the voxels
directly. Each step is local, with no mesh, matrix or global solve, and
it runs at the GPU's memory bandwidth (98 % of copy bandwidth on open
domains). Its costs are staircase walls, which need finer voxels than a
body-fitted mesh for the same wall-shear accuracy, and slow convergence of
steady problems. [Risks](../feasibility/risks.md) covers both.

### Is this a replacement for Fluent or CFX?

Not yet. The finite-volume solver follows CFX's method and is validated on
meshes that fit one GPU. On SimVascular's coronary model it matches
SimVascular's outlet flows to 0.5 %. Still to do: the HiP-CT coronary
tree against a collaborator's CFX run on the same mesh (it needs an
H100), scaling across GPUs, and much of what a commercial code offers
(turbulence models beyond k-kL, FSI, moving meshes). The lattice-Boltzmann solver is
for the regime a meshed solver cannot reach on one node: image-native
resolutions, straight from OME-Zarr. See
[Comparison](../benchmarks/comparison.md).

### How does it relate to SimVascular?

It complements it. SimVascular is the open-source reference for
cardiovascular modelling: its finite-element solver (svMultiPhysics)
handles FSI, rheology and a full set of outlet models. zvCFD reads
SimVascular's meshes and its svSolver outlet files (`cort.dat`,
`rcrt.dat`). Its finite-volume solver runs on a SimVascular mesh as it
stands, and on the Vascular Model Repository's coronary model it agrees
with SimVascular's published results through a cardiac cycle
([Against SimVascular](../validation/simvascular.md)). Its
lattice-Boltzmann solver reaches voxel domains 30–300× larger than a
typical SimVascular mesh. See
[Comparison](../benchmarks/comparison.md#scenario-4-simvascular).

### What does "clean-room" mean here?

The solver is written from published equations and papers only, not from
the source of Ansys, OpenFOAM, FluidX3D, Palabos, waLBerla or other codes
whose licences would bind it. [Clean-room policy](../how_to/cleanroom.md)
lists what may be consulted.

## Storage

### Why store fields in Zarr Vectors rather than as OME-Zarr images?

Because the domain is sparse. At 4 % brick occupancy, dense float32 arrays
chunked at 64³ put **21 GB** on disk for a 2.4 GB payload uncompressed,
and wrote 2–4× more slowly than brick stores in our benchmark. Zarr
Vectors stores only the bricks that exist, keeps one cell per chunk, and
already has the GPU read path, the parallel-write contract and the
multiscale machinery. For viewing, a dense OME-Zarr export of a region is a
one-line job; it is not the working format.

### Do I need Icechunk?

Not for speed, and not for parallel writes. Plain Zarr with one writer per
chunk (or shard) is safe and was as fast as Icechunk in our benchmark.
Icechunk is worth it for versioned, branchable run stores on object
storage. On a POSIX or Lustre filesystem it is officially not safe for
concurrent commits. See [When to use Icechunk](../how_to/icechunk.md).

### Why is every snapshot a separate store?

So that publishing one is atomic (one replace of the collection document),
deleting one is `rm -r`, and a restart reads exactly one store. The brick
centres are duplicated in each snapshot store, but they cost 12 bytes per
brick against 8 kB per brick for four float32 fields: 0.15 %.

## Scale

### How large a domain fits on one 8 × H100 node?

About 2 × 10⁹ fluid cells in fp32 or 4 × 10⁹ in fp16 at a brick fill of 0.6
(640 GB of HBM, 90 % usable; 157 and 81 bytes per stored cell). The HiP-CT coronary tree fits at 10 µm in fp32
(18 GB per GPU) and at 5 µm only in fp16 (62 GB per GPU). A 2048³ rock
sample at porosity 0.2 does not fit in either precision. `zvcfd plan`
answers this for any domain.

### What happens beyond that?

Three routes, in order of cost: fp16 storage (2× capacity), sub-volume runs
with boundary pressures from a whole-domain lubrication or network solve,
and multi-node runs. Streaming bricks from host memory each step is not
viable: a PCIe link is roughly 60× slower than HBM.
