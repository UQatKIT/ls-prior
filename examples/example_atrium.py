# %%
from prior_fields.prior.io import read_endocardial_mesh_with_fibers
from prior_fields.prior.plots import plot_vertex_values_on_surface
from prior_fields.prior.prior import BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.plots import plot_vector_field
from prior_fields.tensor.transformer import (
    alpha_to_sample,
    angles_to_3d_vector,
    sample_to_alpha,
    vectors_3d_to_angles,
)
from prior_fields.tensor.vector_heat_method import get_reference_coordinates

##################
# Read mesh data #
##################
# %%
i = 1  # choose preprocessed geometry (data/shape_model/LA_with_fibers_{i}.vtk)
V, F, fibers = read_endocardial_mesh_with_fibers(i)

##################
# Set parameters #
##################
# %%
sigma = 0.2
ell = 10.0

x_axes, y_axes = get_reference_coordinates(V, F)
mean = alpha_to_sample(
    vectors_3d_to_angles(directions=fibers, x_axes=x_axes, y_axes=y_axes)
)

######################
# Bi-Laplacian Prior #
######################
# %%
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=sigma, ell=ell, mean=mean)

# %%
sample = prior.sample()
plot_vertex_values_on_surface(sample, V)


#########################################
# Prior field as angles of vector field #
#########################################
# %%
alphas = sample_to_alpha(sample)
vector_field = angles_to_3d_vector(alphas=alphas, x_axes=x_axes, y_axes=y_axes)

plot_vector_field(vector_field, V)

# %%
