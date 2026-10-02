# Tutorials

Worked examples from a pipe to a real coronary tree. Every output shown
comes from a real run on an RTX A2000 workstation.

## On a mesh: the finite-volume solver

| Tutorial | What it does |
|---|---|
| [Solving on the mesh itself](fv_mesh.md) | The finite-volume solver on a mesh's own cells: import, a transient run with an RCR outlet, results |
| [Meshes in Zarr Vectors](zarr_vectors_meshes.md) | Import any mesh into a mesh collection once; run either solver from it |

## On voxels: the lattice-Boltzmann solver

| Tutorial | What it does |
|---|---|
| [Your first simulation](first_simulation.md) | Run the vessel-network example from a configuration file |
| [Solving on an OME-Zarr volume](omezarr_input.md) | Pick a pyramid level of a segmentation and run on it |
| [From an Ansys mesh](ansys_mesh.md) | Voxelise a Fluent mesh with its inlets and outlets, and run the HiP-CT coronary tree |
| [Reading results](reading_results.md) | Find snapshots, read brick fields on the CPU or GPU, convert to physical units |
| [Running on the 8 × H100 node](multi_gpu_node.md) | Partition a domain across GPUs on the cluster |

```{toctree}
:hidden:

fv_mesh
zarr_vectors_meshes
first_simulation
omezarr_input
ansys_mesh
reading_results
multi_gpu_node
```
