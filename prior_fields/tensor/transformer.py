import numpy as np

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3


def normalize(vecs: ArrayNx3 | ArrayNx2) -> ArrayNx3 | ArrayNx2:
    """Normalize vectors to length 1."""
    length = np.linalg.norm(vecs, axis=1)
    return np.divide(vecs.T, length, where=length != 0).T


def angles_to_3d_vector(alphas: Array1d, x_axes: ArrayNx3, y_axes: ArrayNx3) -> ArrayNx3:
    """
    Compute 3d vectors of directions from given angles and reference coordinate systems.

    Parameters
    ----------
    alphas : Array1d
        Array of angles between vectors and x-axes
    x_axes : ArrayNx3
        Vectors representing the x-axes of the reference coordinates systems.
    y_axes : ArrayNx3
        Vectors representing the y-axes of the reference coordinates systems.

    Returns
    -------
    ArrayNx3
        Directions corresponding to the given alphas
    """
    return (np.cos(alphas) * x_axes.T).T + (np.sin(alphas) * y_axes.T).T


def angles_between_vectors(a, b):
    return np.arccos(np.sum(normalize(a) * normalize(b), axis=1))


def vectors_3d_to_angles(
    directions: ArrayNx3, x_axes: ArrayNx3, y_axes: ArrayNx3
) -> Array1d:
    """
    Compute angles in reference coordinate systems for given 3d vectors of directions.

    Parameters
    ----------
    directions : ArrayNx3
        3d vectors of directions in the coordinate systems.
    x_axes : ArrayNx3
        Vectors representing the x-axes of the reference coordinates systems.
    y_axes : ArrayNx3
        Vectors representing the y-axes of the reference coordinates systems.

    Returns
    -------
    Array1D
         Angles between :math:`-\\pi` and :math:`\\pi`
    """
    alphas_x = angles_between_vectors(x_axes, directions)
    alphas_y = angles_between_vectors(y_axes, directions)
    alphas = np.array(
        [ax if ay <= np.pi / 2 else -ax for ax, ay in zip(alphas_x, alphas_y)]
    )
    return alphas


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


def alpha_to_sample(alpha: np.ndarray) -> np.ndarray:
    """Inverse of the sigmoid-like transformation from (-pi, pi) to (-infty, infty)."""
    z = alpha / np.pi
    y = (z + 1) / 2
    return np.log(y / (1 - y))


def sample_to_alpha(x: Array1d) -> Array1d:
    """Sigmoid-like transformations of values in (-infty, infty) to (-pi, pi)."""
    return np.pi * (2 / (1 + np.exp(-x)) - 1)
