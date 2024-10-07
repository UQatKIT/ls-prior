import numpy as np
from scipy.stats import mode

from prior_fields.prior.dtypes import Array1d, ArrayNx3


def map_vectors_from_faces_to_vertices(vecs: ArrayNx3, F: ArrayNx3) -> ArrayNx3:
    """Map vectors defined on face-level to vertices.

    For each vertex, the resulting vector is the component-wise mean over the vectors of
    all adjacent faces.

    Parameters
    ----------
    vecs : ArrayNx3
        Vectors defined on face-level.
    F : ArrayNx3
        Array of vertex indices adjacent to each face.

    Returns
    -------
    ArrayNx3
        Vectors mapped to vertex-level.
    """
    adjacent_faces: dict[int, list[int]] = {i: [] for i in range(F.max() + 1)}

    for face_index, face_vertices in enumerate(F):
        for vertex_id in face_vertices:
            adjacent_faces[vertex_id].append(face_index)

    return np.array([vecs[i].mean(axis=0) for i in adjacent_faces.values()])


def map_categories_from_faces_to_vertices(categories: Array1d, F: ArrayNx3) -> Array1d:
    """Map categories defined on face-level to vertices.

    For each vertex, the resulting tag is the mode over the tags of the adjacent faces.

    Parameters
    ----------
    categories : Array1d
        Categories on face-level.
    F : ArrayNx3
        Array of vertex indices adjacent to each face.

    Returns
    -------
    Array1d
        Categories mapped to vertex-level.
    """
    adjacent_faces: dict[int, list[int]] = {i: [] for i in range(F.max() + 1)}

    for face_index, face_vertices in enumerate(F):
        for vertex_id in face_vertices:
            adjacent_faces[vertex_id].append(face_index)

    return np.array([mode(categories[i]).mode for i in adjacent_faces.values()])


def map_fibers_to_tangent_space(
    fibers: ArrayNx3, x1: ArrayNx3, x2: ArrayNx3
) -> ArrayNx3:
    """Map fibers to tangent spaces spanned by x1 and x2.

    Parameters
    ----------
    fibers : ArrayNx3
        (n, 3) array where each row is a fiber vector.
    x1 : ArrayNx3
        (n, 3) array where each row is a vector in the tangent space.
    x2 : ArrayNx3
        (n, 3) array where each row is another vector in the tangent space.

    Returns
    -------
    ArrayNx3
        Fibers mapped to the tangent spaces.
    """
    fibers_x1, fibers_x2 = get_coefficients(fibers, x1, x2)

    return fibers_x1[:, np.newaxis] * x1 + fibers_x2[:, np.newaxis] * x2


def get_coefficients(
    fibers: ArrayNx3, x1: ArrayNx3, x2: ArrayNx3
) -> tuple[ArrayNx3, ArrayNx3]:
    """Get coefficients to write fibers in tangent space basis.

    Parameters
    ----------
    fibers : ArrayNx3
        (n, 3) array where each row is a fiber vector.
    x1 : ArrayNx3
        (n, 3) array where each row is a vector in the tangent space.
    x2 : ArrayNx3
        (n, 3) array where each row is another vector in the tangent space.

    Returns
    -------
    (ArrayNx3, ArrayNx3)

    """
    return np.einsum("ij,ij->i", fibers, x1), np.einsum("ij,ij->i", fibers, x2)
