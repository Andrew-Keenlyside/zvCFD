<p align="center">
  <img src="assets/zvCFD.png" alt="zvCFD: GPU-based computational fluid dynamics backed by Zarr Vectors" width="520">
</p>

# zvCFD

**GPU-based computational fluid dynamics, backed by Zarr Vectors.**

zvCFD solves blood flow and other laminar flows on GPUs, with every
geometry, mesh and result held in chunked
[Zarr Vectors](https://github.com/AllenInstitute/zarr-vectors-py) stores and
each run published as an OME-NGFF RFC-8 collection. It has two solvers:

- **Finite volume (the main solver)**: a CFX-style, element-based,
  vertex-centred finite-volume method on the mesh itself: tetrahedra,
  prisms, pyramids and hexahedra. Pressure and velocity are solved
  coupled, with algebraic multigrid (NVIDIA AmgX) on the GPU. It is steady
  or transient (BDF2), with flow-rate inlets, RCR and coronary outlets,
  and Carreau–Yasuda blood. Wall shear stress is evaluated as CFX does.
  Any mesh (Ansys Fluent, VTK, SimVascular, Gmsh via meshio) is imported
  once into a Zarr Vectors mesh collection, and the solver runs from it.
- **Lattice Boltzmann (for voxel geometries)**: D3Q19 TRT on sparse 8³
  bricks, straight on the voxels of large OME-Zarr segmentations with no
  meshing step, on one or several GPUs. It suits image-native resolutions
  beyond what a mesh can hold.

📖 **Documentation: [zvcfd.readthedocs.io](https://zvcfd.readthedocs.io/en/latest/index.html)**

> [!NOTE]
> **Status: v0.1.** On SimVascular's published coronary model the
> finite-volume solver matches SimVascular's outlet flows to 0.5 % and its
> velocity field to 1.3–1.8 % over a cardiac cycle, on SimVascular's own
> mesh ([Against SimVascular](https://zvcfd.readthedocs.io/en/latest/validation/simvascular.html)).
> It is verified against exact, manufactured and benchmark solutions and
> OpenFOAM ([validation](https://zvcfd.readthedocs.io/en/latest/validation/index.html)).
> Not yet run: the HiP-CT coronary tree against the collaborator's Ansys CFX
> case (it needs an H100), and multi-GPU timing on 8 × H100.

## Install

```bash
pip install -e ./zarr-vectors-py      # its gpu-backend branch
pip install -e ".[gpu,all]"           # in conda, install cupy from conda-forge instead of [gpu]
zvcfd probe
```

The finite-volume solver uses [AmgX](https://github.com/NVIDIA/AMGX), built
from source (`ZVCFD_AMGX_LIB`); without it, `fv.linear: auto` falls back to
zvCFD's own multigrid. See
[Installation](https://zvcfd.readthedocs.io/en/latest/getting_started/installation.html).

## Quick start

A mesh, solved on its own cells (`solver.method` defaults to the
finite-volume solver for mesh sources):

```bash
zvcfd import-mesh pipe.msh --unit mm       # any mesh -> pipe.zvmesh (Zarr Vectors); optional
zvcfd run examples/pipe_fv.yaml --out runs/
zvcfd info runs/pipe-fv-*.zvcfd
```

An image or voxel geometry, with the lattice-Boltzmann solver:

```bash
zvcfd run examples/network_demo.yaml --out runs/
```

Next: the [quickstart](https://zvcfd.readthedocs.io/en/latest/getting_started/quickstart.html),
the [tutorials](https://zvcfd.readthedocs.io/en/latest/tutorials/index.html),
including [solving on a mesh](https://zvcfd.readthedocs.io/en/latest/tutorials/fv_mesh.html)
and [meshes in Zarr Vectors](https://zvcfd.readthedocs.io/en/latest/tutorials/zarr_vectors_meshes.html),
and the [benchmarks](https://zvcfd.readthedocs.io/en/latest/benchmarks/index.html).

## Citing

If you use zvCFD, please cite it with [`CITATION.cff`](CITATION.cff)
(GitHub's *Cite this repository* button uses it), and cite the methods and
data your results depend on. The
[Credits and citing](https://zvcfd.readthedocs.io/en/latest/credits.html)
page lists them; [References](https://zvcfd.readthedocs.io/en/latest/references.html)
has every source behind the documentation.

## Acknowledgements

zvCFD builds on the work of others:

- **[Zarr Vectors](https://github.com/AllenInstitute/zarr-vectors-py)**
  (Allen Institute; A. Keenlyside, F. Collman, BSD-3) holds every mesh
  and brick store, and **[OME-NGFF](https://ngff.openmicroscopy.org)** RFC-8
  collections define the run layout. Arrays go through
  [zarr-python](https://zarr.readthedocs.io); GPU work through
  [CuPy](https://cupy.dev); CPU work through [NumPy](https://numpy.org),
  [SciPy](https://scipy.org) and [PyAMG](https://github.com/pyamg/pyamg).
- **The finite-volume method** is implemented clean-room from the
  literature: Schneider & Raw (1987) for element-based finite volumes; Rhie
  & Chow (1983) and Choi (1999) for pressure–velocity coupling; the Ansys
  CFX theory and modelling guides for the coupled solution strategy and
  boundary treatment; and [NVIDIA AmgX](https://github.com/NVIDIA/AMGX)
  (Naumov et al. 2015, BSD-3) for algebraic multigrid.
- **The lattice-Boltzmann method** is implemented clean-room from the
  literature: Krüger et al. (2017); Bhatnagar, Gross & Krook (1954) and Qian,
  d'Humières & Lallemand (1992) for BGK and D3Q19; Ginzburg, d'Humières and
  Verhaeghe (2003, 2008) for TRT; Guo, Zheng & Shi (2002) for forcing and
  open boundaries; Mei et al. (2006) for consistent initialisation; and Cho &
  Kensey (1991) for the Carreau–Yasuda blood model.
- **Validation** uses exact solutions and benchmarks by White, Womersley,
  Taylor & Green, Sangani & Acrivos, Bogner et al., and Schäfer & Turek.
- **Comparisons** run [OpenFOAM](https://www.openfoam.com) (OpenCFD) and use
  [SimVascular](https://simvascular.github.io) results from the
  [Vascular Model Repository](https://www.vascularmodel.com) (Wilson et al.
  2013). The Repository's data was provided in whole or in part with Federal
  funds from the National Library of Medicine under Grant No. R01LM013120,
  and the National Heart, Lung, and Blood Institute, National Institutes of
  Health, Department of Health and Human Services, under Contract No.
  HHSN268201100035C.
- **The HiP-CT coronary geometry** comes from hierarchical phase-contrast
  tomography at the ESRF (Walsh et al. 2021), segmented and meshed by a
  collaborator.
