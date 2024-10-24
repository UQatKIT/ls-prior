# %%

from prior_fields.prior.converter import scale_mesh_to_unit_cube
from prior_fields.tensor.conduction_velocity import (
    get_conduction_velocity,
    get_longitudinal_and_transversal_velocities_for_tags,
)
from prior_fields.tensor.fiber_grid import get_fiber_parameters_from_uac_grid
from prior_fields.tensor.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)
from prior_fields.tensor.tangent_space import get_uac_basis_vectors

# %%
V, F, uac, fibers, tags = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices("A")
V = scale_mesh_to_unit_cube(V)
basis_x, basis_y = get_uac_basis_vectors(V, F, uac)
angles, _ = get_fiber_parameters_from_uac_grid(uac)

# %%
velocities_l, velocities_t = get_longitudinal_and_transversal_velocities_for_tags(tags)

cv_tensor = get_conduction_velocity(angles, velocities_l, velocities_t, basis_x, basis_y)

# %%
