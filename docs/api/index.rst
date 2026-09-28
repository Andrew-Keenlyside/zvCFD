API Reference
=============

zvCFD is pre-alpha: nothing below carries a compatibility promise yet. The
modules on these pages are the *intended* public surface. Everything else
is internal.

:doc:`python` — the Python surface. ``import zvcfd`` needs only numpy,
scipy and zarr, and gives the domain, configuration, performance model,
unit conversion, capability probe and Fluent reader. Two groups of modules
are imported only when used:

``zvcfd.lbm``
    The GPU solvers; needs cupy and a CUDA device.

``zvcfd.io.fields``
    Brick stores; needs zarr-vectors with the ``building`` surface of the
    ``gpu-backend`` branch.

:doc:`cli` — the ``zvcfd`` command.

Conventions shared with zarr-vectors
------------------------------------

- **Store access only through** ``zarr_vectors.building``. zvCFD never
  imports ``zarr_vectors.core`` or another internal module. Where it needs
  something the building surface lacks, that is filed upstream
  (:doc:`../feasibility/risks`).
- **Keyword-only options** after the positional subject, e.g.
  ``write_brick_chunks(level, domain, fields, *, voxel_size, chunk_bricks, bricks=None)``.
- **Coordinator and worker verbs**: ``create_*`` and ``finalize_*`` are for
  the coordinator, ``write_*`` for workers, as in zarr-vectors' HPC guide.
- **Device arrays**: readers take ``device="cuda"`` and ``decode=``;
  writers accept numpy or cupy.
- **Capabilities, not versions**: ``zvcfd.runtime_capabilities()`` and
  ``zvcfd probe``.

.. toctree::
   :maxdepth: 2

   python
   cli
