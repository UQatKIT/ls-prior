# %%
from pathlib import Path

from pyvista import Plotter

from prior_fields.prior.converter import numpy_to_function
from prior_fields.prior.plots import plot_function
from prior_fields.prior.prior import BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.io import get_mesh_and_point_data_from_lge_mri_based_data
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
print("Reading mesh...")
V_raw, F, uac, fibers, tags = get_mesh_and_point_data_from_lge_mri_based_data(
    Path("data/LGE-MRI-based/2")
)
Vmin = V_raw.min()
Vmax = V_raw.max()
V = V_raw / (Vmax - Vmin)

##################
# Set parameters #
##################
# %%
print("Get parameters...")
sigma = 1.0
ell = 1.0

x_axes, y_axes, _ = get_reference_coordinates(V, F)
mean = alpha_to_sample(
    vectors_3d_to_angles(directions=fibers, x_axes=x_axes, y_axes=y_axes)
)

######################
# Bi-Laplacian Prior #
######################
# %%
print("Initialize prior...")
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=sigma, ell=ell, mean=mean)

# %%
print("Sample from prior...")
sample = prior.sample()
plot_function(numpy_to_function(sample, prior._prior.Vh))

#########################################
# Prior field as angles of vector field #
#########################################
# %%
print("Transform sample to vector field...")
alphas = sample_to_alpha(sample)
vector_field = angles_to_3d_vector(alphas=alphas, x_axes=x_axes, y_axes=y_axes)

plotter = Plotter()
plotter.add_arrows(V, vector_field, mag=0.01)
plotter.remove_scalar_bar()
plotter.show(window_size=(800, 500))

# %%
