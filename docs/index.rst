.. zvCFD documentation master file

.. image:: zvcfd.png
   :width: 55%
   :align: center
   :alt: zvCFD — GPU-based Computational Fluid Dynamics backed by Zarr Vectors

----

**zvCFD** solves flow in image-derived geometries — vessel lumens, porous
media, tissue — directly on the voxels of large OME-Zarr volumes, on GPUs,
with every input and output held in chunked `Zarr Vectors
<https://zarr-vectors-py.readthedocs.io>`_ stores. There is no meshing step:
the segmentation *is* the domain. Only the parts of the volume that contain
fluid are stored, on the GPU and on disk, so memory follows the fluid volume
rather than the bounding box — a coronary tree that fills 0.28 % of its
scan's bounding box costs 0.28 % of it.

The package is an **MVP (v0.1)**. It has:

- a sparse-brick D3Q19 lattice-Boltzmann solver (TRT, Carreau–Yasuda
  rheology) on one or several GPUs;
- pressure, flow-controlled velocity and RCR inlets and outlets;
- a voxeliser that turns an Ansys Fluent mesh into a domain and its
  patches;
- brick stores on Zarr Vectors, and run collections following OME-NGFF
  RFC-8;
- the ``zvcfd`` command line.

It is :doc:`validated <validation/index>` against exact solutions
(second order where walls lie on lattice planes, first order on curved
voxel walls), standard benchmarks and OpenFOAM, and reruns are
byte-identical. Not yet done: timing on the 8 × H100 node, and
interpolated (sub-voxel) walls. :doc:`feasibility/index` sets out the
evidence and :doc:`feasibility/roadmap` the order of work.

zvCFD keeps the conventions of the packages it sits on. It writes stores
only through ``zarr_vectors.building``, the supported surface for code that
builds stores, and follows its three-phase parallel-write contract; it
reads OME-Zarr with ``zarr`` directly, as BRIDGE does; its command line
follows ``zvtools`` and ``bridge-sim``; and it probes capabilities rather
than parsing versions.

----

| `Source code on GitHub <https://github.com/Andrew-Keenlyside/zvCFD>`__

Where to start
--------------

.. list-table::
   :widths: 35 65

   * - :doc:`feasibility/index`
     - Is a Zarr-Vectors-backed multi-GPU CFD package feasible? The verdict,
       the measurements behind it, and what it would take.
   * - :doc:`getting_started/quickstart`
     - Build a domain, run the solver, write and read back a snapshot.
   * - :doc:`getting_started/concepts`
     - Bricks, chunks, shards, workers, snapshots and the run collection.
   * - :doc:`validation/index`
     - Exact solutions (Poiseuille, Womersley, Taylor–Green, Carreau–Yasuda),
       standard benchmarks (sphere arrays, DFG cylinder) and OpenFOAM, with
       orders of accuracy and what byte identity does and does not mean.
   * - :doc:`benchmarks/openfoam`
     - Measured against OpenFOAM on the same workstation: identical voxel
       geometries and the HiP-CT coronary tree.
   * - :doc:`benchmarks/comparison`
     - Estimated speed against Fluent, CFX, STAR-CCM+, OpenFOAM, SimVascular and
       GPU-native lattice-Boltzmann codes, with the assumptions behind each number.
   * - :doc:`spec/index`
     - The on-disk layout: brick stores, run collections, snapshots and the
       parallel I/O contract.
   * - :doc:`api/index`
     - Python and command-line reference.


.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   feasibility/index
   spec/index
   validation/index
   tutorials/index
   how_to/index
   benchmarks/index
   api/index
   credits
   references
