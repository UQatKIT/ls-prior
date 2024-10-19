import numpy as np

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3
from prior_fields.tensor.mapper import get_coefficients, map_fibers_to_tangent_space
from prior_fields.tensor.reader import read_all_human_atrial_fiber_meshes
from prior_fields.tensor.tangent_space_coordinates import get_uac_basis_vectors
from prior_fields.tensor.transformer import vector_coefficients_2d_to_angles


def collect_data_from_human_atrial_fiber_meshes() -> tuple[ArrayNx2, Array1d, Array1d]:
    """
    Collect fiber angles and anatomical tags with UACs from the endocardial geometries of
    the left atrium.

    For each geometry, the tangent space at each vertex is described through two vectors:
        1. The direction in which beta doesn't change and alpha increases,
        2. The direction in which alpha doesn't change and beta increases.
    Based on these (not necessarily orthogonal) basis vectors, the fiber angle is defined
    by interpreting the fiber coefficients as opposite and adjacent of a right triangle.
    Doing so equally accounts for the distortion in the coordinate system in alpha- and
    beta-direction.

    Returns
    -------
    (ArrayNx2, Array1d, Array1d)
        Arrays with UACs, fiber angles and anatomical tags
    """
    # Read data (takes about 70 seconds)
    V_dict, F_dict, uac_dict, fibers_dict, tags_dict = (
        read_all_human_atrial_fiber_meshes()
    )
    keys = sorted(V_dict.keys())

    directions_constant_beta_dict: dict[int, ArrayNx3] = dict()
    directions_constant_alpha_dict: dict[int, ArrayNx3] = dict()

    # Get UAC-based coordinates (takes about 80 seconds)
    for i in keys:
        print(f"Geometry {i}: Get UAC-based tangent space coordinates.")
        directions_constant_beta_dict[i], directions_constant_alpha_dict[i] = (
            get_uac_basis_vectors(V_dict[i], F_dict[i], uac_dict[i])
        )
        print()

    # Unite different geometries
    uac = np.vstack([uac_dict[i] for i in keys])
    fibers = np.vstack([fibers_dict[i] for i in keys])
    tags = np.hstack([tags_dict[i] for i in keys])
    directions_constant_beta = np.vstack(
        [directions_constant_beta_dict[i] for i in keys]
    )
    directions_constant_alpha = np.vstack(
        [directions_constant_alpha_dict[i] for i in keys]
    )

    # Map fibers to tangent space
    fibers_in_tangent_space = map_fibers_to_tangent_space(
        fibers, directions_constant_beta, directions_constant_alpha
    )

    # Get coefficients of fibers in tangent space coordinates
    fiber_coeffs_x, fiber_coeffs_y = get_coefficients(
        fibers_in_tangent_space, directions_constant_beta, directions_constant_alpha
    )

    # Get fiber angle within (-pi/2, pi/2] in UAC system
    fiber_angles = vector_coefficients_2d_to_angles(fiber_coeffs_x, fiber_coeffs_y)

    return uac, fiber_angles, tags
