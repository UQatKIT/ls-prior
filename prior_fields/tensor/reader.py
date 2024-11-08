import re
from math import degrees
from pathlib import Path

import meshio
import numpy as np
from loguru import logger
from scipy.stats import mode

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3
from prior_fields.tensor.fiber_grid import DataUAC
from prior_fields.tensor.parameterization import Geometry
from prior_fields.tensor.tangent_space import (
    get_angles_in_tangent_space,
    get_uac_based_coordinates,
)
from prior_fields.tensor.transformer import angles_between_vectors


def read_raw_atrial_mesh(
    geometry: Geometry,
) -> tuple[ArrayNx3, ArrayNx3, ArrayNx2, ArrayNx3, Array1d]:
    """
    Read vertices, faces, UAC, fibers, and anatomical tags from .vtk file.

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
    logger.info(f"Read data for geometry {geometry.value}...")

    mesh = meshio.read(
        f"data/LGE-MRI-based/{geometry.value}/LA_Endo_{geometry.value}.vtk"
    )

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


def _extract_anatomical_tags_from_file(geometry: Geometry) -> Array1d:
    """
    Read anatomical structure tags at faces from .elem file.

    Parameters
    ----------
    geometry : 1 | 2 | 3 | 4 | 5 | 6 | 7 | 'A'
        Index of geometry.

    Returns
    -------
    Array1d
        Array of anatomical tags at faces.
    """

    pattern = re.compile(rf"Labelled_{geometry.value}_.*\.elem$")
    elem_file = next(
        f
        for f in Path(f"data/LGE-MRI-based/{geometry.value}/").iterdir()
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
    geometry: Geometry,
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
    V, F, uac, fibers_on_faces, tags_of_faces = read_raw_atrial_mesh(geometry)

    fibers, tags = _map_fibers_and_tags_to_vertices(F, fibers_on_faces, tags_of_faces)

    return V, F, uac, fibers, tags


def _map_fibers_and_tags_to_vertices(F, fibers_on_faces, tags_of_faces):
    # Construct mapping of vertex indices to vertex indices of its adjacent faces
    adjacent_faces = _get_dict_with_adjacent_faces_for_each_vertex(F)

    # Map fibers to vertices
    fibers = _map_vectors_from_faces_to_vertices(
        vecs=fibers_on_faces, adjacent_faces=adjacent_faces
    )

    # Map tag for anatomical structure assignment to vertices
    tags = _map_categories_from_faces_to_vertices(
        categories=tags_of_faces, adjacent_faces=adjacent_faces
    )

    return fibers, tags


def _get_dict_with_adjacent_faces_for_each_vertex(F: ArrayNx3) -> dict[int, list[int]]:
    """
    Assemble dictionary with vertex indices as keys and lists of indices of the adjacent
    faces as values.

    Parameters
    ----------
    F : ArrayNx3
        Array of vertex indices, where each row represents one triangle of the mesh.

    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping vertex indices to indices of adjacent faces.
    """
    adjacent_faces: dict[int, list[int]] = {i: [] for i in range(F.max() + 1)}

    for face_index, face_vertices in enumerate(F):
        for vertex_id in face_vertices:
            adjacent_faces[vertex_id].append(face_index)

    return adjacent_faces


def _map_vectors_from_faces_to_vertices(
    vecs: ArrayNx3, adjacent_faces: dict[int, list[int]]
) -> ArrayNx3:
    """
    Map vectors defined on face-level to vertices.

    For each vertex, the resulting vector is the component-wise mean over the vectors of
    all adjacent faces.

    Parameters
    ----------
    vecs : ArrayNx3
        Vectors defined on face-level.
    adjacent_faces : dict[int, list[int]]
        Dictionary where the keys are the vertex indices and the values are lists of
        vertex indices of all adjacent faces.

    Returns
    -------
    ArrayNx3
        Vectors mapped to vertex-level.
    """
    return np.array([vecs[i].mean(axis=0) for i in adjacent_faces.values()])


def _map_categories_from_faces_to_vertices(
    categories: Array1d, adjacent_faces: dict[int, list[int]]
) -> Array1d:
    """
    Map categories defined on face-level to vertices.

    For each vertex, the resulting tag is the mode over the tags of the adjacent faces.

    Parameters
    ----------
    categories : Array1d
        Categories on face-level.
    adjacent_faces : dict[int, list[int]]
        Dictionary where the keys are the vertex indices and the values are lists of
        vertex indices of all adjacent faces.

    Returns
    -------
    Array1d
        Categories mapped to vertex-level.
    """
    return np.array([mode(categories[i]).mode for i in adjacent_faces.values()])


def read_all_human_atrial_fiber_meshes() -> (
    tuple[
        dict[int, ArrayNx3],
        dict[int, ArrayNx3],
        dict[int, ArrayNx2],
        dict[int, ArrayNx3],
        dict[int, Array1d],
    ]
):
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
                read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(Geometry(i))
            )

    return V, F, uac, fibers, tags


def collect_data_from_human_atrial_fiber_meshes() -> DataUAC:
    """
    Collect fiber angles and anatomical tags with UACs from the endocardial geometries of
    the left atrium.

    For each geometry, the tangent space at each vertex is described through two vectors:
        1. The direction in which beta doesn't change and alpha increases,
        2. The direction in which alpha doesn't change and beta increases.
    Based on these (not necessarily orthogonal) basis vectors, the fiber angle is defined
    by interpreting the fiber coefficients as opposite and adjacent of a right triangle.
    Doing so equally accounts for the distortion in the coordinate system in alpha- and
    beta-direction.

    Returns
    -------
    DataUAC
        UACs, fiber angles and anatomical tags
    """
    logger.info("Collecting data from human atrial fiber meshes...")

    # Read data (takes about 70 seconds)
    V_dict, F_dict, uac_dict, fibers_dict, tags_dict = (
        read_all_human_atrial_fiber_meshes()
    )
    keys = sorted(V_dict.keys())

    directions_constant_beta_dict: dict[int, ArrayNx3] = dict()
    directions_constant_alpha_dict: dict[int, ArrayNx3] = dict()

    # Get UAC-based coordinates (takes about 80 seconds)
    for i in keys:
        logger.info(f"Get UAC-based tangent space coordinates for geometry {i}...")
        directions_constant_beta_dict[i], directions_constant_alpha_dict[i] = (
            get_uac_based_coordinates(V_dict[i], F_dict[i], uac_dict[i])
        )

    # Unite different geometries
    uac = np.vstack([uac_dict[i] for i in keys])
    fibers = np.vstack([fibers_dict[i] for i in keys])
    tags = np.hstack([tags_dict[i] for i in keys])
    alpha_axes = np.vstack([directions_constant_beta_dict[i] for i in keys])
    beta_axes = np.vstack([directions_constant_alpha_dict[i] for i in keys])

    _validate_uac_bases(alpha_axes, beta_axes)

    fiber_angles = get_angles_in_tangent_space(fibers, alpha_axes, beta_axes)

    return DataUAC(uac, fiber_angles, tags)


def _validate_uac_bases(basis_x, basis_y):
    critical_value = 30
    angles = angles_between_vectors(basis_x, basis_y)
    angles = np.array([degrees(a) for a in angles])
    angles[
        (np.linalg.norm(basis_x, axis=1) == 0) | (np.linalg.norm(basis_y, axis=1) == 0)
    ] = np.nan
    logger.warning(
        f"{100 * (angles == 0).sum() / basis_x.shape[0]:.4f}% of the bases vectors are "
        "parallel."
    )
    logger.warning(
        f"{100 * (angles < critical_value).sum() / basis_x.shape[0]:.4f}% of the bases "
        f"vectors are almost parallel (angle < {critical_value} degrees)."
    )
    logger.warning(
        f"{100 * np.isnan(angles).sum() / basis_x.shape[0]:.4f}% of the vertices are "
        "missing one or both basis vectors."
    )
