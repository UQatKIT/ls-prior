import numpy as np
from loguru import logger
from potpourri3d import MeshVectorHeatSolver

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3
from prior_fields.tensor.mapper import get_coefficients, map_fibers_to_tangent_space
from prior_fields.tensor.transformer import normalize, vector_coefficients_2d_to_angles


def get_reference_coordinates(
    V: ArrayNx3, F: ArrayNx3
) -> tuple[ArrayNx3, ArrayNx3, ArrayNx3]:
    """Use vector heat method to get reference coordinate systems at each vertex.

    The basis vectors (1, 0) and (0, 1) in the tangent space associated with vertex 1 are
    transported to every other vertex in V using the vector heat method.

    Parameters
    ----------
    V : ArrayNx3
        Array of vertex coordinates.
    F : ArrayNx3
        Array of vertex indices that form the faces of the tessellation.

    Returns
    -------
    (ArrayNx3, ArrayNx3, ArrayNx3)
        x- and y-axes of the reference coordinate systems embedded in 3d space
        and normal vectors of the tangent spaces.
    """
    solver = MeshVectorHeatSolver(V, F)
    basis_x, basis_y, normal_vecs = solver.get_tangent_frames()

    # Parallel transport x-axis vector along the surface
    x_axes_2d = solver.transport_tangent_vector(v_ind=1, vector=[1, 0])
    x_axes = (
        x_axes_2d[:, 0, np.newaxis] * basis_x + x_axes_2d[:, 1, np.newaxis] * basis_y
    )

    # Parallel transport y-axis vector along the surface
    y_axes_2d = solver.transport_tangent_vector(v_ind=1, vector=[0, 1])
    y_axes = (
        y_axes_2d[:, 0, np.newaxis] * basis_x + y_axes_2d[:, 1, np.newaxis] * basis_y
    )

    return x_axes, y_axes, normal_vecs


def get_uac_basis_vectors(
    V: ArrayNx3, F: ArrayNx3, uac: ArrayNx2
) -> tuple[ArrayNx3, ArrayNx3]:
    """Get normalized vectors in tangent space in the direction with constant alpha/beta.

    Note
    ----
    At each vertex, this method defines two vectors that span the tangent space, but are
    in general not orthogonal. This is since the UAC system consists of two coordinates
    that are as close to orthogonal as possible. Note that at some vertices, these two
    are even almost parallel.

    Parameters
    ----------
    V : ArrayNx3
        Array of vertex coordinates.
    F : ArrayNx3
        Array of vertex indices that form the faces of the tessellation.
    uac : ArrayNx2
        Array of UAC (alpha, beta) at each vertex.

    Returns
    -------
    (ArrayNx3, ArrayNx3)
        UAC-based basis vectors of the tangent spaces at each vertex
    """
    _, _, basis_n = get_reference_coordinates(V, F)

    vertex_to_faces_map = _get_vertex_to_face_map(F)

    logger.info("Get coordinate of no change in beta...")
    directions_constant_beta = _get_directions_with_no_change_in_one_uac(
        uac[:, 1], uac[:, 0], vertex_to_faces_map, V, F, basis_n
    )

    logger.info("Get coordinate of no change in alpha...")
    directions_constant_alpha = _get_directions_with_no_change_in_one_uac(
        uac[:, 0], uac[:, 1], vertex_to_faces_map, V, F, basis_n
    )

    return normalize(directions_constant_beta), normalize(directions_constant_alpha)


def _get_vertex_to_face_map(F):
    """
    Get dictionary with vertex indices as keys
    and as list of indices of faces that contain the vertex as values.

    Note
    ----
    This is a more efficient equivalent to
    ```python
        {i: np.where(F == i)[0] for i in range(V.shape[0])}
    ```
    """
    face_indices_repeated = np.repeat(np.arange(F.shape[0]), 3)
    F_flat = F.ravel()
    sorted_idx = np.argsort(F_flat)

    face_indices_sorted_by_vertex = face_indices_repeated[sorted_idx]

    # Points at which face indices in `face_indices_sorted_by_vertex` change from
    # belonging to one vertex to the next
    split_points = np.cumsum(np.unique(F_flat, return_counts=True)[1])[:-1]

    vertex_to_faces_map = np.split(face_indices_sorted_by_vertex, split_points)

    return {i: v for i, v in enumerate(vertex_to_faces_map)}


def _get_directions_with_no_change_in_one_uac(
    current_uac: Array1d,
    other_uac: Array1d,
    vertex_to_faces_map: dict,
    V: ArrayNx3,
    F: ArrayNx3,
    basis_n: ArrayNx3,
) -> ArrayNx3:
    vertices_to_uac_change_map = {
        k: {f_idx: 100 * (current_uac[k] - current_uac[F[f_idx]]) for f_idx in faces}
        for k, faces in vertex_to_faces_map.items()
    }

    basis_uac = []
    count_missing = 0

    for v_idx in range(V.shape[0]):
        found_direction = False

        for f_idx, uac_changes in vertices_to_uac_change_map[v_idx].items():
            # c is a list of length 3 (triangular face has three vertices)
            # One entry is always zero (uac(current vertex) - uac(current vertex))
            # The face contains the direction of interest, if
            # 1. there is a second change = 0 (same uac at neighboring vertex), or
            # 2. change is positive towards one and negative towards the other neighbor

            v_indices_face = F[f_idx]

            # 1.
            if (uac_changes == 0).sum() > 1:
                v_idx_neighbor_no_change = v_indices_face[
                    (uac_changes == 0) & (v_indices_face != v_idx)
                ][0]
                direction_no_change = V[v_idx_neighbor_no_change] - V[v_idx]

                found_direction = True

            # 2.
            elif (uac_changes.max() > 0) & (uac_changes.min() < 0):
                edge_lengths = (
                    np.linalg.norm(V[v_indices_face] - V[v_idx], axis=1) / 1000
                )
                uac_changes_per_distance = np.divide(
                    uac_changes, edge_lengths, where=edge_lengths != 0
                )

                # compute direction in which uac coordinate does not change
                weights_no_change = abs(
                    np.divide(
                        1, uac_changes_per_distance, where=uac_changes_per_distance != 0
                    )
                )
                weights_no_change = weights_no_change / np.linalg.norm(weights_no_change)
                direction_no_change = (
                    weights_no_change * (V[v_indices_face] - V[v_idx]).T
                ).T.sum(axis=0)

                found_direction = True

            # post-processing of no-change direction
            if found_direction:
                # map direction to tangent space
                direction_no_change = (
                    direction_no_change
                    - (direction_no_change @ basis_n[v_idx]) * basis_n[v_idx]
                )

                # Choose direction with positive change in other UAC
                change_other_uac = (
                    weights_no_change * (other_uac[v_indices_face] - other_uac[v_idx])
                ).sum()
                direction_no_change = np.sign(change_other_uac) * direction_no_change

                break

        if found_direction:
            basis_uac.append(direction_no_change)
        else:
            count_missing += 1
            basis_uac.append(np.zeros(3))

    if count_missing > 0:
        logger.warning(
            "No face with no-change found for "
            f"{count_missing} / {V.shape[0]} of the vertices."
        )

    return np.vstack(basis_uac)


def get_angles_in_tangent_space(fibers, basis_x, basis_y):
    fibers_tangent_space = map_fibers_to_tangent_space(fibers, basis_x, basis_y)
    coeffs_x, coeffs_y = get_coefficients(fibers_tangent_space, basis_x, basis_y)
    fiber_angles = vector_coefficients_2d_to_angles(coeffs_x, coeffs_y)
    return fiber_angles
