import numpy as np

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3


def normalize(vecs: ArrayNx3 | ArrayNx2) -> ArrayNx3 | ArrayNx2:
    """Normalize vectors to length 1."""
    length = np.linalg.norm(vecs, axis=1)
    return np.divide(vecs.T, length, where=length != 0).T


def angles_to_3d_vector(angles: Array1d, x_axes: ArrayNx3, y_axes: ArrayNx3) -> ArrayNx3:
    """
    Compute 3d vectors of directions from given angles and reference coordinate systems.

    Parameters
    ----------
    angles : Array1d
        Array of angles between vectors and x-axes
    x_axes : ArrayNx3
        Vectors representing the x-axes of the reference coordinates systems.
    y_axes : ArrayNx3
        Vectors representing the y-axes of the reference coordinates systems.

    Returns
    -------
    ArrayNx3
        Directions corresponding to the given angles
    """
    coeff_x, coeff_y = angles_to_2d_vector_coefficients(angles)
    return (coeff_x * x_axes.T).T + (coeff_y * y_axes.T).T


def angles_between_vectors(a: ArrayNx3, b: ArrayNx3) -> Array1d:
    """Compute the angle between a and b as :math:`arccos(a * b / (|a| * |b|))`."""
    dot_product = np.einsum("ij,ij->i", normalize(a), normalize(b))
    dot_product = np.clip(dot_product, -1.0, 1.0)  # handle floating point issues
    return np.arccos(dot_product)


def vector_coefficients_2d_to_angles(coeff_x: Array1d, coeff_y: Array1d) -> Array1d:
    """
    Compute angles from vector coefficients.

    Interpret vector coefficients as opposite and adjacent in the triangle determining
    the angle between x-axis and vector. With that define the angle between the x-axis
    and the vector in a straightened coordinate system in which x- and y-axis are
    orthogonal as :math:`arctan(coeff_y / coeff_x)`.

    Note
    ----
    This is more robust than just computing the angle between the x-axis and the vector
    for the given case, that the basis is not orthogonal.

    Parameters
    ----------
    coeff_x : Array1d
        Lengths of vectors in x-direction.
    coeff_y : Array1d
        Lengths of vectors in y-direction.

    Returns
    -------
    Array1d
        Angles between vectors and x-axes
    """
    angles = np.zeros_like(coeff_x)

    # arctan(opposite / adjacent)
    mask = coeff_x != 0
    angles[mask] = np.arctan(coeff_y[mask] / coeff_x[mask])

    # pi/2
    mask = (coeff_x == 0) & (coeff_y != 0)
    angles[mask] = np.pi / 2

    # No fiber orientation known
    mask = (coeff_x == 0) & (coeff_y == 0)
    angles[mask] = np.nan

    return angles


def angles_to_2d_vector_coefficients(angles: Array1d) -> tuple[Array1d, Array1d]:
    """
    Get coefficients of normalized vectors at given angle with the x-axis.

    Note
    ----
    This is the inverse of `vector_coefficients_2d_to_angles()`.
    """
    coeff_x = np.ones_like(angles)
    coeff_y = np.tan(angles)

    normalization_constant = np.sqrt(abs(coeff_x) ** 2 + abs(coeff_y) ** 2)
    coeff_x = coeff_x / normalization_constant
    coeff_y = coeff_y / normalization_constant

    return coeff_x, coeff_y


def angles_to_sample(angles: Array1d) -> Array1d:
    """Inverse of the sigmoid-like transformation from (-pi/2, pi/2) to (-inf, inf)."""
    epsilon = 1e-9
    y = np.clip(angles, -np.pi / 2 + epsilon, np.pi / 2 - epsilon)
    return 4 * np.log((0.5 * np.pi + y) / (0.5 * np.pi - y))


def sample_to_angles(x: Array1d) -> Array1d:
    """Sigmoid-like transformations of values in (-inf, inf) to (-pi/2, pi/2)."""
    return np.pi * (1 / (1 + np.exp(-x / 4)) - 0.5)


def shift_angles_by_mean(
    angles: Array1d, mean: Array1d, adjust_range: bool = False
) -> Array1d:
    """
    Add mean to angles.

    Parameters
    ----------
    angles : Array1d
        Array of angles within (-pi/2, pi/2].
    mean : Array1d
        Array of mean angles within (-pi/2, pi/2] to add to `angles`.
    adjust_range : bool, optional
        Whether to map angles to interval (-pi/2, pi/2], defaults to False.
        Note that the transformation is discontinuous for `adjust_range=True`.
    Returns
    -------
    Array1d
        Array of angles shifted by the mean.
    """
    angles_shifted = angles + mean

    if adjust_range:
        if (angles_shifted.max() > np.pi) or (angles_shifted.min() <= -np.pi):
            raise ValueError("'angles' and 'mean' have to lie within (-pi/2, pi/2].")

        angles_shifted[angles_shifted > np.pi / 2] -= np.pi
        angles_shifted[angles_shifted <= -np.pi / 2] += np.pi

    return angles_shifted
