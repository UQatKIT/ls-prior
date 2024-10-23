import numpy as np

from prior_fields.prior.dtypes import Array1d, ArrayNx3, ArrayNx3x3
from prior_fields.tensor.transformer import angles_to_2d_vector_coefficients


def conduction_velocity_to_longitudinal_velocity(cv: float | Array1d, k: float = 3.75):
    """
    Compute longitudinal velocity from conduction velocity and anisotropy factor.

    Parameters
    ----------
    cv : float | Array1d
        Conduction velocity
    k : float, optional
        Anisotropy factor,
        defaults to 3.75 which is reported for the left atrium in the literature, see
        https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.01910

    Returns
    -------
    float | Array1d
        Velocity along the fiber direction.
    """
    return cv / np.sqrt(1 + (1 / k**2))


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
