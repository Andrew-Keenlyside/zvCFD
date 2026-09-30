"""Unstructured volume meshes for the finite-volume solver.

:class:`UnstructuredMesh` holds nodes, elements by type and boundary zones;
:func:`read_fluent_mesh` / :func:`write_fluent_mesh` read and write Ansys
Fluent ``.msh`` files; :mod:`zvcfd.mesh.generate` builds validation meshes.
"""

from zvcfd.mesh.core import EDGES, FACES, KINDS, BoundaryZone, UnstructuredMesh, to_vtk
from zvcfd.mesh.fluent import read_fluent_mesh, write_fluent_mesh

__all__ = ["EDGES", "FACES", "KINDS", "BoundaryZone", "UnstructuredMesh", "read_fluent_mesh",
           "to_vtk", "write_fluent_mesh"]
