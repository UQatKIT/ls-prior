import numpy as np

from prior_fields.prior.dtypes import Array1d, ArrayNx3


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


def _angles_between_vectors(a, b):
    return np.arccos(
        np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    )


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
    alphas_x = _angles_between_vectors(x_axes, directions)
    alphas_y = _angles_between_vectors(y_axes, directions)
    alphas = np.array(
        [ax if ay <= np.pi / 2 else -ax for ax, ay in zip(alphas_x, alphas_y)]
    )
    return alphas


def alpha_to_sample(alpha: np.ndarray) -> np.ndarray:
    """Inverse of the sigmoid-like transformation from (-pi, pi) to (-infty, infty)."""
    z = alpha / np.pi
    y = (z + 1) / 2
    return np.log(y / (1 - y))


def sample_to_alpha(x: Array1d) -> Array1d:
    """Sigmoid-like transformations of values in (-infty, infty) to (-pi, pi)."""
    return np.pi * (2 / (1 + np.exp(-x)) - 1)
