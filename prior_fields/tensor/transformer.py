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


def angles_between_vectors(a, b):
    dot_product = np.einsum("ij,ij->i", normalize(a), normalize(b))
    dot_product = np.clip(dot_product, -1.0, 1.0)  # handle floating point issues
    return np.arccos(dot_product)


def vector_coefficients_2d_to_angles(coeff_x, coeff_y):
    """
    Interpret vector coefficients as opposite and adjacent in the triangle determining
    the angle between x-axis and vector. This is more robust than just computing the
    angle between the x-axis and the vector for the given case, that the basis is not
    orthogonal.
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
    """Get coefficients of normalized vectors at given angle with the x-axis.

    Note
    ----
    This is the inverse of `vector_coefficients_2d_to_angles()`.
    """
    coeff_x = np.ones_like(angles)
    coeff_y = np.tan(angles)

    coeff_sum = abs(coeff_x) + abs(coeff_y)
    coeff_x = coeff_x / coeff_sum
    coeff_y = coeff_y / coeff_sum

    return coeff_x, coeff_y


def angles_to_sample(angles: np.ndarray) -> np.ndarray:
    """Inverse of the sigmoid-like transformation from (-pi/2, pi/2) to (-inf, inf)."""
    epsilon = 1e-6
    y = np.clip(angles, -np.pi / 2 + epsilon, np.pi / 2 - epsilon)
    return np.log((0.5 * np.pi + y) / (0.5 * np.pi - y))


def sample_to_angles(x: Array1d) -> Array1d:
    """Sigmoid-like transformations of values in (-inf, inf) to (-pi/2, pi/2)."""
    return np.pi * (1 / (1 + np.exp(-x)) - 0.5)
