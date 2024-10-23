import numpy as np

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3, ArrayNx3x3


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


def get_conduction_velocity_tensor_from_angles_and_velocities(
    angles: Array1d,
    velocities_l: Array1d,
    velocities_t: Array1d,
    basis_x: ArrayNx3,
    basis_y: ArrayNx3,
) -> ArrayNx3x3:
    """
    Compute conduction velocity tensors from angles and velocities.

    The conduction velocity tensor is defined as
        :math:`v_l^2(x) l(x) \\otimes l(x) + v_t^2(x) t(x) \\otimes t(x)`,
    where fiber and transversal direction are given as
        :math:`l(x) = cos(\\alpha) e_1(x) + sin(\\alpha) e_2(x)`
        :math:`l(x) = -sin(\\alpha) e_1(x) + cos(\\alpha) e_2(x)`
    for basis vectors (e_1, e_2).

    Note
    ----
    For the UAC-based bases of the tangent spaces, `basis_x` and `basis_y` are not
    orthogonal. Therefore, the angles are interpreted based on `basis_x` and `basis_y` as
    explained in `vector_coefficients_2d_to_angles()` and :math:`cos(\\alpha)` and
    :math:`sin(\\alpha)` are replaced by the coefficients obtained from
    `angles_to_2d_vector_coefficients()`.

    Parameters
    ----------
    angles : Array1d
        Array of fiber angles.
    velocities_l : Array1d
        Array of velocities along the fiber direction (longitudinal).
    velocities_t : Array1d
        Array of velocities transversal to the fiber direction.
    basis_x : ArrayNx3
        Array where the n-th row is a vector in the tangent space at vertex n.
    basis_y : ArrayNx3
        Array where the n-th row is another vector in the tangent space at vertex n.
    """
    coeff_x, coeff_y = angles_to_2d_vector_coefficients(angles)
    direction_l = (coeff_x * basis_x.T + coeff_y * basis_y.T).T
    direction_t = (-1 * coeff_y * basis_x.T + coeff_x * basis_y.T).T

    tensor_l = np.einsum("i,ij,ik->ijk", velocities_l**2, direction_l, direction_l)
    tensor_t = np.einsum("i,ij,ik->ijk", velocities_t**2, direction_t, direction_t)

    return tensor_l + tensor_t
