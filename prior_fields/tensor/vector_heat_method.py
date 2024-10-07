import numpy as np
from potpourri3d import MeshVectorHeatSolver
from scipy.linalg import norm

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3


def get_reference_coordinates(
    V: ArrayNx3, F: ArrayNx3
) -> tuple[ArrayNx3, ArrayNx3, ArrayNx3]:
    """Get reference coordinate systems at each vertex.

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
    """Get vectors in tangent space in the direction with constant alpha/beta.

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
    basis_x, _, basis_n = get_reference_coordinates(V, F)

    vertex_to_faces_map = _get_vertex_to_face_map(F)

    print("Get coordinate of no change in beta...")
    directions_constant_beta = _get_directions_with_no_change_in_one_uac_coordinate(
        uac[:, 1], vertex_to_faces_map, V, F, basis_n, basis_x
    )

    print("Get coordinate of no change in alpha...")
    directions_constant_alpha = _get_directions_with_no_change_in_one_uac_coordinate(
        uac[:, 0], vertex_to_faces_map, V, F, basis_n, basis_x
    )

    return directions_constant_beta, directions_constant_alpha


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


def _get_directions_with_no_change_in_one_uac_coordinate(
    uac_coordinate: Array1d,
    vertex_to_faces_map: dict,
    V: ArrayNx3,
    F: ArrayNx3,
    basis_n: ArrayNx3,
    basis_x: ArrayNx3,
) -> ArrayNx3:
    vertices_to_uac_change_map = {
        k: {f_idx: uac_coordinate[k] - uac_coordinate[F[f_idx]] for f_idx in faces}
        for k, faces in vertex_to_faces_map.items()
    }

    vertex_to_face_with_no_uac_change_map = {
        k: {
            f_idx: changes
            for f_idx, changes in face_to_change_map.items()
            if (changes.max() > 0) & (changes.min() < 0)
        }
        for k, face_to_change_map in vertices_to_uac_change_map.items()
    }

    basis_uac = []
    for v_idx in range(V.shape[0]):
        # NOTE: There is no such face for
        # 2.1% of the no-alpha-changes and
        # 3.9% of the no-beta-changes
        if len(vertex_to_face_with_no_uac_change_map[v_idx].keys()) > 0:
            f_idx = list(vertex_to_face_with_no_uac_change_map[v_idx].keys())[0]
            uac_changes = list(vertex_to_face_with_no_uac_change_map[v_idx].values())[0]
            vertex_indices_in_face = F[f_idx]

            # compute direction in which uac coordinate does not change
            weights_no_change = abs(np.divide(1, uac_changes, where=uac_changes != 0))
            direction_no_change = (
                weights_no_change * (V[vertex_indices_in_face] - V[v_idx]).T
            ).T.sum(axis=0)
            # map direction to tangent space
            direction_no_change = (
                direction_no_change
                - (direction_no_change @ basis_n[v_idx]) * basis_n[v_idx]
            )
            # normalize
            direction_no_change = direction_no_change / norm(direction_no_change)
            # Choose direction with acute angle to the VHM x-axis
            direction_no_change = (
                np.sign(direction_no_change @ basis_x[v_idx]) * direction_no_change
            )

            basis_uac.append(direction_no_change)

        else:
            basis_uac.append(np.zeros(3))

    return np.vstack(basis_uac)
