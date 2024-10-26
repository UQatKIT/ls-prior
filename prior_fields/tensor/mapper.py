import numpy as np
from loguru import logger
from scipy.stats import mode

from prior_fields.prior.dtypes import Array1d, ArrayNx3
from prior_fields.tensor.transformer import normalize


def map_vectors_from_faces_to_vertices(
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


def map_categories_from_faces_to_vertices(
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


def get_coefficients(
    fibers: ArrayNx3, x: ArrayNx3, y: ArrayNx3
) -> tuple[ArrayNx3, ArrayNx3]:
    """
    Get coefficients to write fibers in (x, y) basis.

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


def map_fibers_to_tangent_space(fibers: ArrayNx3, x: ArrayNx3, y: ArrayNx3) -> ArrayNx3:
    """
    Get normalized fibers in tangent spaces spanned by x and y.

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
    thresh = 0.25  # corresponds to approximately 75 degrees
    # thresh = 0.5  # corresponds to 60 degrees
    mask_orthogonal = (abs(fiber_coeff_x) < thresh) & (abs(fiber_coeff_y) < thresh)

    logger.warning(
        f"Excluding {100 * mask_orthogonal.sum() / fibers.shape[0]:.2f}% of the fibers"
        " as they are almost orthogonal to the tangent space."
    )
    fiber_coeff_x[mask_orthogonal] = np.nan
    fiber_coeff_y[mask_orthogonal] = np.nan

    fibers_mapped = fiber_coeff_x[:, np.newaxis] * x + fiber_coeff_y[:, np.newaxis] * y

    mask_reverse = fiber_coeff_x < 0
    fibers_mapped[mask_reverse] *= -1

    return normalize(fibers_mapped)
