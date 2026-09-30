<p align="center">
  <img src="assets/zvCFD.png" alt="zvCFD: GPU-based computational fluid dynamics backed by Zarr Vectors" width="520">
</p>

# zvCFD

**GPU-based computational fluid dynamics, backed by Zarr Vectors.**

zvCFD solves flow in image-derived geometries, such as vessel lumens and
porous media, directly on the voxels of large OME-Zarr volumes, on one or
several GPUs. There is no meshing step. Only the bricks that contain fluid
are stored, on the GPU and in chunked
[Zarr Vectors](https://github.com/AllenInstitute/zarr-vectors-py) stores, and
each run is an OME-NGFF RFC-8 collection.

📖 **Documentation: [zvcfd.readthedocs.io](https://zvcfd.readthedocs.io/en/latest/index.html)**

> [!NOTE]
> **Status: MVP (v0.1).** Validated against exact solutions and standard
> benchmarks ([validation](https://zvcfd.readthedocs.io/en/latest/validation/index.html)).
> On the HiP-CT coronary tree, one RTX A2000 settles the 77 outlet flow
> splits 10× sooner than OpenFOAM on 16 cores. Multi-GPU runs are verified
> but not yet timed on 8 × H100.

## Install

```bash
pip install -e ./zarr-vectors-py      # its gpu-backend branch
pip install -e ".[gpu,all]"           # in conda, install cupy from conda-forge instead of [gpu]
zvcfd probe
```

See [Installation](https://zvcfd.readthedocs.io/en/latest/getting_started/installation.html).

## Quick start

```bash
zvcfd run examples/network_demo.yaml --out runs/
zvcfd info runs/network-demo-*.zvcfd
```

Next: the [tutorials](https://zvcfd.readthedocs.io/en/latest/tutorials/index.html),
including [running a Fluent mesh](https://zvcfd.readthedocs.io/en/latest/tutorials/ansys_mesh.html),
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
  (Allen Institute; A. Keenlyside, F. Collman, BSD-3) holds every brick
  store, and **[OME-NGFF](https://ngff.openmicroscopy.org)** RFC-8
  collections define the run layout. Arrays go through
  [zarr-python](https://zarr.readthedocs.io); GPU work through
  [CuPy](https://cupy.dev); CPU work through [NumPy](https://numpy.org),
  [SciPy](https://scipy.org) and [PyAMG](https://github.com/pyamg/pyamg).
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
