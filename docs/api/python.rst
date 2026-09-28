Python
======

Top level
---------

``import zvcfd`` needs only numpy, scipy and zarr. It re-exports
``BrickDomain``, ``Lattice``, ``RunConfig``, ``load_config``,
``read_fluent_boundary`` and the functions below.

.. autofunction:: zvcfd.runtime_capabilities
.. autofunction:: zvcfd.require
.. autofunction:: zvcfd.device_count
.. autofunction:: zvcfd.estimate

Domains
-------

.. automodule:: zvcfd.domain

Solvers
-------

.. automodule:: zvcfd.lbm.solver

.. automodule:: zvcfd.solvers.lubrication

Brick stores
------------

.. automodule:: zvcfd.io.fields

Inputs
------

.. automodule:: zvcfd.io.omezarr

.. automodule:: zvcfd.io.fluent_msh

Run collections
---------------

.. automodule:: zvcfd.collection

Configuration and runs
----------------------

.. automodule:: zvcfd.config

.. automodule:: zvcfd.run

Performance model and units
---------------------------

.. automodule:: zvcfd.perfmodel

.. automodule:: zvcfd.units

Phantoms
--------

.. automodule:: zvcfd.phantoms
