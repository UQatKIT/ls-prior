# %%
from prior_fields.prior.dtypes import ArrayNx3
from prior_fields.tensor.io import read_meshes_from_lge_mri_data
from prior_fields.tensor.vector_heat_method import get_reference_coordinates

##################
# Read mesh data #
##################
# %%
V, F, uac, fibers = read_meshes_from_lge_mri_data()

coords_vhm: dict[int, tuple[ArrayNx3, ArrayNx3]] = dict()
for i in range(8):
    coords_vhm[i] = get_reference_coordinates(V[0], F[0])

# %%
