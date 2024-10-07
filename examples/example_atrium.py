# %%
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from vedo.dolfin import plot as vedo_plot

from prior_fields.prior.converter import numpy_to_function
from prior_fields.prior.plots import plot_function
from prior_fields.prior.prior import BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.io import (
    get_mesh_and_point_data_from_lge_mri_based_data,
    read_endocardial_mesh_from_bilayer_model,
)
from prior_fields.tensor.plots import plot_vector_field
from prior_fields.tensor.transformer import (
    alpha_to_sample,
    angles_to_3d_vector,
    sample_to_alpha,
    vectors_3d_to_angles,
)
from prior_fields.tensor.vector_heat_method import get_reference_coordinates

#########################
# Example on small mesh #
#########################
# %%
V, F, _ = read_endocardial_mesh_from_bilayer_model(1)

prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=10.0, ell=10.0)
sample = prior.sample()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = vedo_plot(
    numpy_to_function(sample, prior._prior.Vh),
    lw=False,
    style=1,
    axes=0,
    zoom=2,
    size=(475, 400),
    viewup=[1, 1, -1],
)
ax.set_axis_off()
ax.imshow(np.asarray(im))
plt.savefig("figures/sample_atrium.svg")
plt.show()

##################
# Read mesh data #
##################
# %%
print("Reading mesh...")
V_raw, F, uac, fibers = get_mesh_and_point_data_from_lge_mri_based_data(
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
sigma = 10.0
ell = 0.3

x_axes, y_axes, _ = get_reference_coordinates(V, F)
mean = alpha_to_sample(
    vectors_3d_to_angles(directions=fibers, x_axes=x_axes, y_axes=y_axes)
)

######################
# Bi-Laplacian Prior #
######################
# %%
print("Initialize prior...")
# prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=sigma, ell=ell, mean=mean)
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=sigma, ell=ell)

# %%
print("Sample from prior...")
sample = prior.sample()
# plot_vertex_values_on_surface(sample, V)
plot_function(numpy_to_function(sample, prior._prior.Vh))

#########################################
# Prior field as angles of vector field #
#########################################
# %%
print("Transform sample to vector field...")
alphas = sample_to_alpha(sample)
vector_field = angles_to_3d_vector(alphas=alphas, x_axes=x_axes, y_axes=y_axes)

plot_vector_field(vector_field, V, length=0.01)

# %%
