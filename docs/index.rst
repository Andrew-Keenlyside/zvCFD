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

The package is at the **feasibility stage**. What exists and is tested: a
sparse-brick D3Q19 lattice-Boltzmann solver on one GPU (bit-identical to its
dense counterpart, within 1 % of analytic Poiseuille flow), the brick-store
layout on Zarr Vectors with GPU-side reads, a run collection following
OME-NGFF RFC-8, a lubrication pressure solver with algebraic multigrid, a
reader for Ansys Fluent meshes, a roofline performance model, and the
``zvcfd`` command line. What does not yet exist: the multi-GPU driver
(8 × H100 with NVLink halo exchange), production boundary conditions and
collision models. :doc:`feasibility/index` sets out the evidence and
:doc:`feasibility/roadmap` the order of work.

zvCFD keeps the conventions of the packages it sits on. It writes stores
only through ``zarr_vectors.building``, the supported surface for code that
builds stores, and follows its three-phase parallel-write contract; it
reads OME-Zarr with ``zarr`` directly, as BRIDGE does; its command line
follows ``zvtools`` and ``bridge-sim``; and it probes capabilities rather
than parsing versions.

----

| `Link to the GitHub repository <https://github.com/BRIDGE-Neuroscience/zvCFD>`__

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
   * - :doc:`benchmarks/comparison`
     - Estimated speed against Fluent, CFX, STAR-CCM+, OpenFOAM and GPU-native
       lattice-Boltzmann codes, with the assumptions behind each number.
   * - :doc:`spec/index`
     - The on-disk layout: brick stores, run collections, snapshots and the
       parallel I/O contract.
   * - :doc:`api/index`
     - Python and command-line reference.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/installation
   getting_started/quickstart
   getting_started/concepts
   getting_started/faq

.. toctree::
   :maxdepth: 1
   :caption: Feasibility Study
   :hidden:

   feasibility/index
   feasibility/architecture
   feasibility/ansys_context
   feasibility/risks
   feasibility/roadmap

.. toctree::
   :maxdepth: 1
   :caption: Specification
   :hidden:

   spec/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/first_simulation
   tutorials/omezarr_input
   tutorials/ansys_mesh
   tutorials/reading_results
   tutorials/multi_gpu_node

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: How-To Guides
   :hidden:

   how_to/plan_a_run
   how_to/choose_brick_and_chunk
   how_to/hpc_sge
   how_to/icechunk
   how_to/cleanroom

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks
   :hidden:

   benchmarks/index
   benchmarks/kernels
   benchmarks/io
   benchmarks/multiresolution
   benchmarks/comparison

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   references
