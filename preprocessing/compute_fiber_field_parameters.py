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
V_dict, F_dict, uac_dict, fibers_dict, tags_dict = read_meshes_from_lge_mri_data()
keys = sorted(V_dict.keys())

# %%
# takes about 80 seconds
directions_constant_beta_dict: dict[int, ArrayNx3] = dict()
directions_constant_alpha_dict: dict[int, ArrayNx3] = dict()

for i in keys:
    print(f"Geometry {i}: Get UAC-based tangent space coordinates.")
    directions_constant_beta_dict[i], directions_constant_alpha_dict[i] = (
        get_uac_basis_vectors(V_dict[i], F_dict[i], uac_dict[i])
    )
    print()

# %%
# Unite different geometries
uac = np.vstack([uac_dict[i] for i in keys])
fibers = np.vstack([fibers_dict[i] for i in keys])
tags = np.hstack([tags_dict[i] for i in keys])
directions_constant_beta = np.vstack([directions_constant_beta_dict[i] for i in keys])
directions_constant_alpha = np.vstack([directions_constant_alpha_dict[i] for i in keys])

# %%
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

# %%
# takes about 30 seconds
FiberGridComputer(
    uac=uac,
    fiber_coeffs_x=fiber_coeffs_x,
    fiber_coeffs_y=fiber_coeffs_y,
    fiber_angles=fiber_angles,
    anatomical_structure_tags=tags,
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
