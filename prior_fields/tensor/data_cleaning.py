from sys import setrecursionlimit

import numpy as np

from prior_fields.prior.dtypes import ArrayNx3


def remove_vertex_from_mesh(
    idx: int, V: ArrayNx3, F: ArrayNx3, fibers: ArrayNx3
) -> tuple[ArrayNx3, ArrayNx3, ArrayNx3]:
    """
    Remove vertex, corresponding fibers, and related faces from mesh.

    Parameters
    ----------
    idx : int
        Index of vertex to be removed.
    V : ArrayNx3
        Vertex coordinates.
    F : ArrayNx3
        Triangular faces connecting vertex indices.
    fibers : ArrayNx3
        3d fiber directions at vertices.

    Returns
    -------
    (ArrayNx3, ArrayNx3, ArrayNx3)
        V, F, fibers without vertex with given index.
    """
    org_size = V.shape[0]

    # Remove faces containing the vertex to be removed.
    F = np.delete(F, np.where(F == idx)[0], axis=0)

    # TODO if this works, move to private method to avoid code duplication with io.py
    # Identify and remove unreferenced vertices
    referenced_indices = np.unique(F)
    V = V[referenced_indices]
    fibers = fibers[referenced_indices]

    # Reindex F_endo to ensure correct references after removing vertices
    reindex_map = -1 * np.ones(org_size, dtype=int)
    reindex_map[referenced_indices] = np.arange(referenced_indices.shape[0])
    F = reindex_map[F]

    return V, F, fibers


def _get_duplicate_edges(F: ArrayNx3) -> list[list[int]]:
    """
    Get list of directed edges that belong to more than one face.

    Parameters
    ----------
    F : ArrayNx3
        Array of vertex indices that form the faces of the tessellation.

    Returns
    -------
    list
        List of duplicate edges.
    """
    edges = np.array([[[a, b], [b, c], [c, a]] for a, b, c in F]).reshape(
        F.shape[0] * 3, 2
    )
    e, c = np.unique(edges, axis=0, return_counts=True)
    return e[c > 1].tolist()


def _reorder_neighbours(
    F: ArrayNx3,
    current_face: int,
    faces_to_neighbours_map: dict,
    reordered_faces: list[int],
) -> ArrayNx3:
    """
    Recursively reorder neighbouring faces until there are no duplicate edges.

    Parameters
    ----------
    F : ArrayNx3
        Array of vertex indices that form the faces of the tessellation.
    current_face : int
        Index of the correctly ordered reference face whose neighbors are reordered.
    faces_to_neighbours_map : dict
        Dictionary mapping face indices to the indices of all not yet ordered neighbours.
    reordered_faces : list[int]
        List of faces that are already in correct order.

    Returns
    -------
    ArrayNx3
        Array of faces where the neighbours of `current_face` are ordered.
    """
    neighbours = faces_to_neighbours_map[current_face]
    for neighbour in neighbours:
        violation = _get_duplicate_edges(F[[current_face] + [neighbour]])
        if violation:
            F[neighbour] = [F[neighbour][2], F[neighbour][1], F[neighbour][0]]
        reordered_faces.append(neighbour)
        faces_to_neighbours_map[current_face].remove(neighbour)
        faces_to_neighbours_map[neighbour].remove(current_face)

    for i in reordered_faces:
        if faces_to_neighbours_map[i]:
            F = _reorder_neighbours(F, i, faces_to_neighbours_map, reordered_faces)
            break

    if not any(faces_to_neighbours_map.values()):
        return F
    else:
        raise RecursionError("Could not reach all vertices.")


def _handle_artifacts(
    F: ArrayNx3, faces_to_neighbours_map: dict, reordered_faces: list[int]
) -> tuple[dict, list[int]]:
    """
    Manually handle the boundary artifact that two faces are neighbours to each other but
    not to any further face.
    """
    faces_with_single_neighbour = [
        k for k, v in faces_to_neighbours_map.items() if len(v) == 1
    ]
    neighbours_of_faces_with_single_neighbour = [
        faces_to_neighbours_map[k][0] for k in faces_with_single_neighbour
    ]
    unconnected_faces = set(faces_with_single_neighbour).intersection(
        neighbours_of_faces_with_single_neighbour
    )
    for f in unconnected_faces:
        n = faces_to_neighbours_map[f][0]
        violation = _get_duplicate_edges(F[[f] + [n]])
        if violation:
            F[n] = [F[n][2], F[n][1], F[n][0]]
        reordered_faces.append(n)
        faces_to_neighbours_map[f].remove(n)
    return faces_to_neighbours_map, reordered_faces


def reorder_edges_of_faces(F: ArrayNx3) -> ArrayNx3:
    """
    Reorder edges of faces for `potpourri3d.MeshVectorHeatSolver`.

    Parameters
    ----------
    F : ArrayNx3
        Array of unordered faces

    Returns
    -------
    ArrayNx3
        Array of ordered faces
    """
    setrecursionlimit(5 * F.shape[0])

    empty_list: list[int] = []

    edges = np.array([[[a, b], [b, c], [c, a]] for a, b, c in F]).reshape(
        F.shape[0] * 3, 2
    )
    edges_to_faces_map = {(min(e), max(e)): empty_list.copy() for e in edges}
    for i, f in enumerate(F):
        edges_to_faces_map[(min(f[0], f[1]), max(f[0], f[1]))].append(i)
        edges_to_faces_map[(min(f[1], f[2]), max(f[1], f[2]))].append(i)
        edges_to_faces_map[(min(f[2], f[0]), max(f[2], f[0]))].append(i)

    faces_to_neighbours_map = {i: empty_list.copy() for i in range(F.shape[0])}
    for neighbours in edges_to_faces_map.values():
        if len(neighbours) > 1:
            faces_to_neighbours_map[neighbours[0]].append(neighbours[1])
            faces_to_neighbours_map[neighbours[1]].append(neighbours[0])
        if len(neighbours) > 2:
            faces_to_neighbours_map[neighbours[0]].append(neighbours[2])
            faces_to_neighbours_map[neighbours[2]].append(neighbours[0])
            faces_to_neighbours_map[neighbours[1]].append(neighbours[2])
            faces_to_neighbours_map[neighbours[2]].append(neighbours[1])
        if len(neighbours) > 3:
            raise ValueError("Something weird is happening.")

    reordered_faces = [0]  # Take ordering of face 0 as starting point
    faces_to_neighbours_map, reordered_faces = _handle_artifacts(
        F, faces_to_neighbours_map, reordered_faces
    )

    return _reorder_neighbours(F, 0, faces_to_neighbours_map, reordered_faces)
