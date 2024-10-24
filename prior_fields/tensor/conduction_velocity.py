from enum import IntEnum
from typing import overload

import numpy as np

from prior_fields.prior.dtypes import Array1d, ArrayNx3, ArrayNx3x3
from prior_fields.tensor.transformer import angles_to_2d_vector_coefficients


class AnatomicalTag(IntEnum):
    LA = 11
    LAA = 13
    LIPV = 21
    LSPV = 23
    RIPV = 25
    RSPV = 27


# Parameters from Supplementary Table 1 in
# https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.01910
CV = {
    AnatomicalTag.LA: dict(A=600.6, B=3.38 * 1e6, BCL=500, C=30.3, k=3.75),
    AnatomicalTag.LAA: dict(A=600.6, B=4.1 * 1e6, BCL=500, C=29.9, k=3.75),
    AnatomicalTag.LIPV: dict(A=600.6, B=1.41 * 1e5, BCL=500, C=40.7, k=3.75),
    AnatomicalTag.LSPV: dict(A=600.6, B=1.41 * 1e5, BCL=500, C=40.7, k=3.75),
    AnatomicalTag.RIPV: dict(A=600.6, B=1.41 * 1e5, BCL=500, C=40.7, k=3.75),
    AnatomicalTag.RSPV: dict(A=600.6, B=1.41 * 1e5, BCL=500, C=40.7, k=3.75),
}


def get_conduction_velocity_for_tag(tag: AnatomicalTag) -> float:
    """
    Based on values from the literature compute the conduction velocity in an anatomical
    region as :math:`A - B * exp(-BCL/C)`, where BCL is the basis cycle length.

    Parameters
    ----------
    tag : AnatomicalTag
        Specifies the anatomical region.

    Returns
    -------
    float
        Conduction velocity
    """
    return CV[tag]["A"] - CV[tag]["B"] * np.exp(-1 * CV[tag]["BCL"] / CV[tag]["C"])


def get_conduction_velocities_for_tags(tags: Array1d) -> Array1d:
    """Get array with conduction velocities for array of anatomical regions."""
    return np.array([get_conduction_velocity_for_tag(tag) for tag in tags])


def get_anisotropy_factors_for_tags(tags: Array1d) -> Array1d:
    """Get array with anisotropy factors for array of anatomical regions."""
    return np.array([CV[tag]["k"] for tag in tags])


@overload
def conduction_velocity_to_longitudinal_velocity(cv: float, k: float) -> float: ...


@overload
def conduction_velocity_to_longitudinal_velocity(cv: Array1d, k: float) -> Array1d: ...


@overload
def conduction_velocity_to_longitudinal_velocity(cv: Array1d, k: Array1d) -> Array1d: ...


def conduction_velocity_to_longitudinal_velocity(cv, k=3.75):
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
