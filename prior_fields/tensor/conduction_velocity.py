from dataclasses import dataclass
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


@dataclass
class ConductionVelocityParameters:
    A: float = 600.6
    B: float = 3.38 * 1e6
    C: float = 30.3
    k: float = 3.75


# Parameters from Supplementary Table 1 in
# https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.01910
CV: dict[AnatomicalTag, ConductionVelocityParameters] = {
    AnatomicalTag.LA: ConductionVelocityParameters(),
    AnatomicalTag.LAA: ConductionVelocityParameters(B=4.1 * 1e6, C=29.9),
    AnatomicalTag.LIPV: ConductionVelocityParameters(B=1.41 * 1e5, C=40.7),
    AnatomicalTag.LSPV: ConductionVelocityParameters(B=1.41 * 1e5, C=40.7),
    AnatomicalTag.RIPV: ConductionVelocityParameters(B=1.41 * 1e5, C=40.7),
    AnatomicalTag.RSPV: ConductionVelocityParameters(B=1.41 * 1e5, C=40.7),
}


def _get_conduction_velocity_for_tag(
    tag: AnatomicalTag, BCL: int | None = None
) -> float:
    """
    Based on values from the literature compute the conduction velocity in an anatomical
    region as :math:`A - B * exp(-BCL/C)`, where BCL is the basis cycle length.

    Parameters
    ----------
    tag : AnatomicalTag
        Specifies the anatomical region.
    BCL : int | None, optional
        Basic cycle length, defaults to None which means a BCL long enough to not reduce
        conduction velocity due to restitution of the muscle fiber.
        Reasonable values are in the order of 200 to 1,000 [ms].

    Returns
    -------
    float
        Conduction velocity
    """
    if BCL:
        return CV[tag].A - CV[tag].B * np.exp(-1 * BCL / CV[tag].C)
    else:
        return CV[tag].A


def _get_conduction_velocities_for_tags(
    tags: Array1d, BCL: int | None = None
) -> Array1d:
    """Get array with conduction velocities for array of anatomical regions."""
    return np.array([_get_conduction_velocity_for_tag(tag, BCL) for tag in tags])


def _get_anisotropy_factors_for_tags(tags: Array1d) -> Array1d:
    """Get array with anisotropy factors for array of anatomical regions."""
    return np.array([CV[tag].k for tag in tags])


@overload
def _conduction_velocity_to_longitudinal_velocity(cv: float, k: float) -> float: ...


@overload
def _conduction_velocity_to_longitudinal_velocity(cv: Array1d, k: float) -> Array1d: ...


@overload
def _conduction_velocity_to_longitudinal_velocity(
    cv: Array1d, k: Array1d
) -> Array1d: ...


def _conduction_velocity_to_longitudinal_velocity(cv, k=3.75):
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


def get_longitudinal_and_transversal_velocities_for_tags(
    tags: Array1d,
) -> tuple[Array1d, Array1d]:
    conduction_velocities = _get_conduction_velocities_for_tags(tags)
    anisotropy_factors = _get_anisotropy_factors_for_tags(tags)

    velocities_l = _conduction_velocity_to_longitudinal_velocity(
        conduction_velocities, k=anisotropy_factors
    )
    velocities_t = velocities_l / anisotropy_factors

    return velocities_l, velocities_t


def get_conduction_velocity(
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
        Array where the n-th row is another vector in the tangent space at vertex n which
        is orthogonal to `basis_x`.
    """
    coeff_x, coeff_y = angles_to_2d_vector_coefficients(angles)
    direction_l = (coeff_x * basis_x.T + coeff_y * basis_y.T).T
    direction_t = (-1 * coeff_y * basis_x.T + coeff_x * basis_y.T).T

    tensor_l = np.einsum("i,ij,ik->ijk", velocities_l**2, direction_l, direction_l)
    tensor_t = np.einsum("i,ij,ik->ijk", velocities_t**2, direction_t, direction_t)

    return tensor_l + tensor_t
