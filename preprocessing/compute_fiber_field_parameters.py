# %%
import numpy as np

from prior_fields.prior.dtypes import ArrayNx3
from prior_fields.tensor.io import read_meshes_from_lge_mri_data
from prior_fields.tensor.mapper import (
    FiberGrid,
    FiberGridComputer,
    get_coefficients,
    map_fibers_to_tangent_space,
)
from prior_fields.tensor.transformer import vector_coefficients_2d_to_angles
from prior_fields.tensor.vector_heat_method import get_uac_basis_vectors

# %%
# takes about 70 seconds
V, F, uac, fibers, tags = read_meshes_from_lge_mri_data()

# %%
# takes about 90 seconds
fibers_in_tangent_space: dict[int, ArrayNx3] = dict()
fiber_coeffs_x: dict[int, ArrayNx3] = dict()
fiber_coeffs_y: dict[int, ArrayNx3] = dict()
fiber_angles: dict[int, ArrayNx3] = dict()

for i in sorted(V.keys()):
    print(f"Geometry {i}:")
    print("Get UAC-based tangent space coordinates.")
    directions_constant_beta, directions_constant_alpha = get_uac_basis_vectors(
        V[i], F[i], uac[i]
    )
    print("Map fibers to tangent space.")
    fibers_in_tangent_space[i] = map_fibers_to_tangent_space(
        fibers[i], directions_constant_beta, directions_constant_alpha
    )
    print("Get coefficients of fibers in tangent space coordinates.")
    fiber_coeffs_x[i], fiber_coeffs_y[i] = get_coefficients(
        fibers_in_tangent_space[i], directions_constant_beta, directions_constant_alpha
    )
    print("Get fiber angle within (-pi/2, pi/2] in UAC system.")
    fiber_angles[i] = vector_coefficients_2d_to_angles(
        fiber_coeffs_x[i], fiber_coeffs_y[i]
    )
    print()

# %%
# Collect data from different geometries in single array for mean computation
uac_array = np.vstack([uac[i] for i in np.arange(1, 8)])
fiber_coeffs_x_array = np.hstack([fiber_coeffs_x[i] for i in np.arange(1, 8)])
fiber_coeffs_y_array = np.hstack([fiber_coeffs_y[i] for i in np.arange(1, 8)])
fiber_angles_array = np.hstack([fiber_angles[i] for i in np.arange(1, 8)])
tags_array = np.hstack([tags[i] for i in np.arange(1, 8)])

# %%
# takes about 30 seconds
FiberGridComputer(
    uac=uac_array,
    fiber_coeffs_x=fiber_coeffs_x_array,
    fiber_coeffs_y=fiber_coeffs_y_array,
    fiber_angles=fiber_angles_array,
    anatomical_structure_tags=tags_array,
    max_depth=7,
    point_threshold=100,
).get_fiber_grid().save()

fiber_grid = FiberGrid.read_from_binary_file(
    "data/LGE-MRI-based/fiber_grid_max_depth7_point_threshold100.npy"
)
fiber_grid.plot("tag")
fiber_grid.plot("mean")
fiber_grid.plot("std")

# %%
