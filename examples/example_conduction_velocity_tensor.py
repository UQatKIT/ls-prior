# %%
import numpy as np

from prior_fields.prior.converter import scale_mesh_to_unit_cube
from prior_fields.tensor.conduction_velocity import (
    conduction_velocity_to_longitudinal_velocity,
    get_conduction_velocity_tensor_from_angles_and_velocities,
)
from prior_fields.tensor.fiber_grid import get_fiber_parameters_from_uac_grid
from prior_fields.tensor.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)
from prior_fields.tensor.tangent_space import get_uac_basis_vectors

# %%
V_raw, F, uac, fibers, _ = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices("A")
V = scale_mesh_to_unit_cube(V_raw)
basis_x, basis_y = get_uac_basis_vectors(V, F, uac)
angles, _ = get_fiber_parameters_from_uac_grid(uac)

# %%
# Parameters for RA/LA from https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.01910
# TODO: let depend on anatomical tag
A = 600.6
B = 3.38 * 1e6
BCL = 500  # basis cycle length
C = 30.3
conduction_velocity = A - B * np.exp(-1 * BCL / C)

anisotropy_factor = 3.75

velocities_l = conduction_velocity_to_longitudinal_velocity(
    conduction_velocity, k=anisotropy_factor
) * np.ones_like(angles)
velocities_t = velocities_l / anisotropy_factor

# %%
cv_tensor = get_conduction_velocity_tensor_from_angles_and_velocities(
    angles, velocities_l, velocities_t, basis_x, basis_y
)

# %%
