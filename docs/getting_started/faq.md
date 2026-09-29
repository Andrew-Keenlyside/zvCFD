# Frequently asked questions

## General

### Why lattice-Boltzmann rather than finite volumes?

Because the input is an image. A lattice-Boltzmann step is local — each
voxel reads its 18 neighbours and writes itself — so it maps onto the voxel
grid with no mesh, no matrix and no global solve, and it runs at the GPU's
memory bandwidth. Our kernel reaches 98 % of the device's copy bandwidth on
open domains. A finite-volume solver on an image first needs a body-fitted
mesh, which for a 10⁹-voxel lumen is itself a large, fragile job, and then
holds a system matrix of 1–5.6 GB per million cells. The trade-off:
staircase walls need finer voxels than a body-fitted mesh for the same
wall-shear accuracy, and steady problems converge slowly in time-marching
LBM. [Risks](../feasibility/risks.md) covers both.

### Is this a replacement for Fluent or CFX?

No. At the size of a typical Ansys model it is not faster per GPU and far
less complete. For the HiP-CT coronary case, Fluent's GPU solver on the
14.8 M-cell Simpleware mesh needs an estimated 6–25 min per cardiac cycle on
one H100 (CFX, which is CPU-only, 1.3–5.3 h on 128 cores); zvCFD at 20 µm (5× as many cells) needs ~42 min on one H100 or
~6 min on eight. zvCFD is for the regime a meshed solver cannot reach on
one node: image-native resolutions with 10⁸–10¹⁰ fluid voxels, straight from
OME-Zarr, where a mesh and its matrix would not fit in memory. See
[Comparison](../benchmarks/comparison.md).

### How does it relate to SimVascular?

It complements it. SimVascular is the open-source reference for
cardiovascular modelling: its finite-element solver (svMultiPhysics)
handles FSI, rheology and a full set of outlet models, on meshes of a few
million elements from CT angiography. zvCFD solves image-native voxel
domains 30–300× larger, on GPUs, without a mesh. Its planned outlet
conditions come from SimVascular's own 0-D solver (svZeroDSolver, BSD-3),
coupled every few steps. Its validation cases include SimVascular's test
problems and the Vascular Model Repository. See
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
