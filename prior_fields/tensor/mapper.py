from sys import stderr
from warnings import warn

import numpy as np
from scipy.stats import mode

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3
from prior_fields.tensor.fiber_grid import FiberGrid
from prior_fields.tensor.transformer import normalize


def get_dict_with_adjacent_faces_for_each_vertex(F: ArrayNx3) -> dict[int, list[int]]:
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


def map_vectors_from_faces_to_vertices(
    vecs: ArrayNx3, adjacent_faces: dict[int, list[int]]
) -> ArrayNx3:
    """Map vectors defined on face-level to vertices.

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


def map_categories_from_faces_to_vertices(
    categories: Array1d, adjacent_faces: dict[int, list[int]]
) -> Array1d:
    """Map categories defined on face-level to vertices.

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


def map_fibers_to_tangent_space(fibers: ArrayNx3, x: ArrayNx3, y: ArrayNx3) -> ArrayNx3:
    """Get normalized fibers in tangent spaces spanned by x and y.

    Note
    ----
    Analogous to the angle restriction to (-pi/2, pi/2], the fibers in the tangent spaces
    are transformed so that they form an acute angle with the x-axis.

    Parameters
    ----------
    fibers : ArrayNx3
        (n, 3) array where each row is a fiber vector.
    x : ArrayNx3
        (n, 3) array where each row is a vector in the tangent space.
    y : ArrayNx3
        (n, 3) array where each row is another vector in the tangent space.

    Returns
    -------
    ArrayNx3
        Fibers mapped to the tangent spaces.
    """
    fiber_coeff_x, fiber_coeff_y = get_coefficients(normalize(fibers), x, y)

    # Exclude fibers which are almost orthogonal to tangent space
    # thresh = 0.25  # corresponds to approximately 75 degrees
    thresh = 0.5  # corresponds to 60 degrees
    mask_orthogonal = (abs(fiber_coeff_x) < thresh) & (abs(fiber_coeff_y) < thresh)
    warn(
        f"\nExcluding {100 * mask_orthogonal.sum() / fibers.shape[0]:.2f}% of the fibers"
        " as they are almost orthogonal to the tangent space."
    )
    stderr.flush()

    fibers_mapped = fiber_coeff_x[:, np.newaxis] * x + fiber_coeff_y[:, np.newaxis] * y

    mask_reverse = fiber_coeff_x < 0
    fibers_mapped[mask_reverse] *= -1

    return normalize(fibers_mapped)


def get_coefficients(
    fibers: ArrayNx3, x: ArrayNx3, y: ArrayNx3
) -> tuple[ArrayNx3, ArrayNx3]:
    """Get coefficients to write fibers in tangent space basis.

    Parameters
    ----------
    fibers : ArrayNx3
        (n, 3) array where each row is a fiber vector.
    x : ArrayNx3
        (n, 3) array where each row is a vector in the tangent space.
    y : ArrayNx3
        (n, 3) array where each row is another vector in the tangent space.

    Returns
    -------
    (ArrayNx3, ArrayNx3)

    """
    return np.einsum("ij,ij->i", fibers, x), np.einsum("ij,ij->i", fibers, y)


def get_fiber_parameters(uac: ArrayNx2) -> tuple[Array1d, Array1d]:
    """
    Map mean and standard deviation of fiber angles from `FiberGrid` to vertices based on
    the given UACs.

    Parameters
    ----------
    uac : ArrayNx2
        Universal atrial coordinates of vertices.

    Returns
    -------
    (Array1d, Array1d)
        Arrays with mean and standard deviation of fiber angles.
    """
    fiber_grid = FiberGrid.read_from_binary_file(
        "data/LGE-MRI-based/fiber_grid_max_depth7_point_threshold100.npy"
    )

    fiber_mean = np.zeros(uac.shape[0])
    fiber_std = np.zeros(uac.shape[0])
    unmatched_vertices = []

    for i in range(fiber_mean.shape[0]):
        j = np.where(
            (uac[i, 0] >= fiber_grid.grid_x[:, 0])
            & (uac[i, 0] < fiber_grid.grid_x[:, 1])
            & (uac[i, 1] >= fiber_grid.grid_y[:, 0])
            & (uac[i, 1] < fiber_grid.grid_y[:, 1])
        )[0]
        try:
            fiber_mean[i] = fiber_grid.fiber_angle_circmean[j[0]]
            fiber_std[i] = fiber_grid.fiber_angle_circstd[j[0]]
        except IndexError:
            fiber_std[i] = np.nan
            unmatched_vertices.append(i)

    if len(unmatched_vertices) > 0:
        warn(
            "\nCouldn't find grid cell for "
            f"{100 * len(unmatched_vertices) / uac.shape[0]:.2f}%"
            " of the vertices"
        )
        stderr.flush()

    return fiber_mean, fiber_std
