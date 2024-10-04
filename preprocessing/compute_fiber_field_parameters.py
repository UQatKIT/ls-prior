# %%
from prior_fields.tensor.io import read_meshes_from_lge_mri_data
from prior_fields.tensor.vector_heat_method import get_uac_basis_vectors

# %%
V, F, uac, fibers = read_meshes_from_lge_mri_data()

# %%
directions_constant_alpha, directions_constant_beta = get_uac_basis_vectors(
    V[0], F[0], uac[0]
)

# %%
