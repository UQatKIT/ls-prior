import re
from pathlib import Path
from typing import Literal

import meshio
import numpy as np

from prior_fields.prior.dtypes import Array1d
from prior_fields.tensor.mapper import (
    ArrayNx2,
    ArrayNx3,
    get_dict_with_adjacent_faces_for_each_vertex,
    map_categories_from_faces_to_vertices,
    map_vectors_from_faces_to_vertices,
)


def read_raw_atrial_mesh(
    geometry: int | Literal["A"],
) -> tuple[ArrayNx3, ArrayNx3, ArrayNx2, ArrayNx3, Array1d]:
    """Read vertices, faces, UAC, fibers, and anatomical tags from .vtk file.

    Note
    ----
    The data used was downloaded from https://zenodo.org/records/3764917.
    It has to be located in 'data/LGE-MRI-based/{geometry}/LA_Endo_{geometry}.vtk'.

    Parameters
    ----------
    geometry : 1 | 2 | 3 | 4 | 5 | 6 | 7 | 'A'
        Index of geometry to read.

    Returns
    -------
    (ArrayNx3, ArrayNx3, ArrayNx2, ArrayNx3, Array1d)
        Arrays with vertex coordinates, faces,
        universal atrial coordinates of the vertices,
        and fiber orientation and anatomical structure tag for each face.
    """
    mesh = meshio.read(f"data/LGE-MRI-based/{geometry}/LA_Endo_{geometry}.vtk")

    # vertices and faces
    V = mesh.points
    F = mesh.get_cells_type(mesh.cells[0].type)

    # universal atrial coordinates
    alpha = mesh.point_data["alpha"]
    beta = mesh.point_data["beta"]
    uac = np.column_stack([alpha, beta])

    # fibers
    fibers = mesh.cell_data["fibers"][0]

    # tag for anatomical structure assignment
    tag = _extract_anatomical_tags_from_file(geometry)

    return V, F, uac, fibers, tag


def _extract_anatomical_tags_from_file(
    geometry: int | Literal["A"],
) -> Array1d:
    """Read anatomical structure tags at faces from .elem file.

    Parameters
    ----------
    geometry : 1 | 2 | 3 | 4 | 5 | 6 | 7 | 'A'
        Index of geometry.

    Returns
    -------
    Array1d
        Array of anatomical tags at faces.
    """

    pattern = re.compile(rf"Labelled_{geometry}_.*\.elem$")
    elem_file = next(
        f
        for f in Path(f"data/LGE-MRI-based/{geometry}/").iterdir()
        if pattern.match(f.name)
    )

    with open(elem_file) as f:
        # First line contains the number of faces
        num_faces = int(f.readline().strip())

        element_tags = np.zeros(num_faces, dtype=int)
        for i, line in enumerate(f):
            # Last component in each line is the element tag
            element_tags[i] = int(line.split()[-1])

    return element_tags


def read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(
    geometry: int | Literal["A"],
) -> tuple[ArrayNx3, ArrayNx3, ArrayNx2, ArrayNx3, Array1d]:
    """
    Read vertices, faces, UAC, fibers, and anatomical tags from .vtk file and map fibers
    and tags to vertices.

    Parameters
    ----------
    geometry : 1 | 2 | 3 | 4 | 5 | 6 | 7 | 'A'
        Index of geometry.

    Returns
    -------
    (ArrayNx3, ArrayNx3, ArrayNx2, ArrayNx3, Array1d)
        Arrays with vertex coordinates, faces,
        universal atrial coordinates of the vertices,
        and fiber orientation and anatomical structure tag both mapped to the vertices.
    """
    V, F, uac, fibers_on_faces, tag_of_faces = read_raw_atrial_mesh(geometry)

    # Construct mapping of vertex indices to vertex indices of its adjacent faces
    adjacent_faces = get_dict_with_adjacent_faces_for_each_vertex(F)

    # Map fibers to vertices
    fibers = map_vectors_from_faces_to_vertices(
        vecs=fibers_on_faces, adjacent_faces=adjacent_faces
    )

    # Map tag for anatomical structure assignment to vertices
    tag = map_categories_from_faces_to_vertices(
        categories=tag_of_faces, adjacent_faces=adjacent_faces
    )

    return V, F, uac, fibers, tag


def read_all_human_atrial_fiber_meshes() -> tuple[
    dict[int, ArrayNx3],
    dict[int, ArrayNx3],
    dict[int, ArrayNx2],
    dict[int, ArrayNx3],
    dict[int, Array1d],
]:
    """
    Read vertices, faces, UAC, fibers, and anatomical tags from all 7 endocardial meshes
    of the left atrium published in https://zenodo.org/records/3764917.

    Returns
    -------
    (
        dict[int, ArrayNx3],
        dict[int, ArrayNx3],
        dict[int, ArrayNx2],
        dict[int, ArrayNx3],
        dict[int, Array1d],
    )
        Vertices and faces, and UAC, fiber orientation and tags for anatomical structure
        at the vertices for the 7 geometries.
    """
    V: dict[int, ArrayNx3] = dict()
    F: dict[int, ArrayNx3] = dict()
    uac: dict[int, ArrayNx2] = dict()
    fibers: dict[int, ArrayNx3] = dict()
    tags: dict[int, Array1d] = dict()

    for p in Path("data/LGE-MRI-based").iterdir():
        idx = str(p)[-1]
        i = int(idx) if idx.isnumeric() else None

        if i is not None:
            V[i], F[i], uac[i], fibers[i], tags[i] = (
                read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(i)
            )

    return V, F, uac, fibers, tags
