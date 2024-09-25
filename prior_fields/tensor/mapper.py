import numpy as np

from prior_fields.prior.dtypes import ArrayNx3


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
