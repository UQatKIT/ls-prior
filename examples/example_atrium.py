# %%

import numpy as np
from pyvista import Plotter

from prior_fields.prior.plots import get_poly_data
from prior_fields.prior.prior import BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.mapper import get_fiber_parameters
from prior_fields.tensor.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)
from prior_fields.tensor.tangent_space_coordinates import get_reference_coordinates
from prior_fields.tensor.transformer import (
    angles_to_3d_vector,
    angles_to_sample,
    sample_to_angles,
)

##################
# Read mesh data #
##################
# %%
print("Reading mesh...")
V_raw, F, uac, fibers, _ = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(1)
Vmin = V_raw.min()
Vmax = V_raw.max()
V = V_raw / (Vmax - Vmin)

##################
# Set parameters #
##################
# %%
print("Get parameters...")
mean_fiber_angle, std_fiber_angle = get_fiber_parameters(V, uac)

sample_mean = angles_to_sample(mean_fiber_angle)
sigma = std_fiber_angle * std_fiber_angle
ell = np.ones_like(sigma)

plotter = Plotter(shape=(1, 3))

plotter.subplot(0, 0)
plotter.add_text("Mean fiber angle")
plotter.add_mesh(
    get_poly_data(V, F),
    scalars=mean_fiber_angle,
    cmap="twilight",
    scalar_bar_args=dict(title="angle", n_labels=2, label_font_size=12),
)

plotter.subplot(0, 1)
plotter.add_text("Pointwise variance")
plotter.add_mesh(
    get_poly_data(V, F),
    scalars=sigma,
    scalar_bar_args=dict(title="sigma", n_labels=2, label_font_size=12),
)

plotter.subplot(0, 2)
plotter.add_text("Correlation length")
plotter.add_mesh(
    get_poly_data(V, F),
    scalars=ell,
    scalar_bar_args=dict(title="ell", n_labels=2, label_font_size=12),
)

plotter.show(window_size=(900, 400))

######################
# Bi-Laplacian Prior #
######################
# %%
print("Initialize prior...")
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=sigma, ell=ell, mean=sample_mean)

# %%
print("Sample from prior...")
sample = prior.sample()

#########################################
# Prior field as angles of vector field #
#########################################
# %%
print("Transform sample to vector field...")
angles = sample_to_angles(sample)
x_axes, y_axes, _ = get_reference_coordinates(V, F)
vector_field = angles_to_3d_vector(angles=angles, x_axes=x_axes, y_axes=y_axes)

plotter = Plotter()
plotter.add_mesh(get_poly_data(V, F), color="lightgrey", opacity=0.99)
plotter.add_arrows(V, vector_field, mag=0.01, color="tab:blue")
plotter.show(window_size=(800, 500))

###################
# Compare samples #
###################
# %%
plotter = Plotter()
plotter.add_mesh(get_poly_data(V, F), color="lightgrey", opacity=0.99)
plotter.add_arrows(
    V,
    angles_to_3d_vector(
        angles=sample_to_angles(prior.sample()), x_axes=x_axes, y_axes=y_axes
    ),
    mag=0.01,
    color="tab:blue",
)
plotter.add_arrows(
    V,
    angles_to_3d_vector(
        angles=sample_to_angles(prior.sample()), x_axes=x_axes, y_axes=y_axes
    ),
    mag=0.01,
    color="tab:orange",
)
plotter.show(window_size=(800, 500))

# %%
