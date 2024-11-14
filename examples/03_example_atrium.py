# %%
from pathlib import Path

import numpy as np
from pyvista import Plotter
from scipy.spatial import KDTree

from prior_fields.prior.converter import scale_mesh_to_unit_cube
from prior_fields.prior.plots import get_poly_data
from prior_fields.prior.prior import BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.parameterization import Geometry, PriorParameters
from prior_fields.tensor.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)
from prior_fields.tensor.tangent_space import get_vhm_based_coordinates
from prior_fields.tensor.transformer import (
    angles_to_3d_vector,
    sample_to_angles,
    shift_angles_by_mean,
)

geometry = Geometry(1)

#############
# Read data #
#############
# %%
V, F, uac, _, _ = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(geometry)
V = scale_mesh_to_unit_cube(V)
basis_x, basis_y, _ = get_vhm_based_coordinates(V, F)
params = PriorParameters.load(Path(f"data/parameters/params_{geometry.value}.npy"))

# %%
plotter = Plotter(shape=(1, 3), window_size=(900, 400))

plotter.subplot(0, 0)
plotter.add_text("Prior mean")
plotter.add_mesh(
    get_poly_data(V, F),
    scalars=params.mean,
    scalar_bar_args=dict(title="mean", n_labels=2, label_font_size=12),
)

plotter.subplot(0, 1)
plotter.add_text("Pointwise variance")
plotter.add_mesh(
    get_poly_data(V, F),
    scalars=params.sigma,
    scalar_bar_args=dict(title="sigma^2", n_labels=2, label_font_size=12),
)

plotter.subplot(0, 2)
plotter.add_text("Correlation length")
plotter.add_mesh(
    get_poly_data(V, F),
    scalars=params.ell,
    scalar_bar_args=dict(title="ell", n_labels=2, label_font_size=12),
)

plotter.show()

######################
# Bi-Laplacian Prior #
######################
# %%
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=params.sigma, ell=params.ell)

#########################################
# Prior field as angles of vector field #
#########################################
# %%
sample = prior.sample()
angles_around_zero = sample_to_angles(sample)
angles_mean = sample_to_angles(params.mean)
angles = shift_angles_by_mean(angles_around_zero, angles_mean)

x_axes, y_axes, _ = get_vhm_based_coordinates(V, F)
vector_field = angles_to_3d_vector(angles=angles, x_axes=x_axes, y_axes=y_axes)

plotter = Plotter(window_size=(800, 500))
plotter.add_text("Vector field sample")
plotter.add_mesh(get_poly_data(V, F), color="lightgrey", opacity=0.99)
plotter.add_arrows(V, vector_field, mag=0.01, color="tab:blue")
plotter.show()

###################
# Compare samples #
###################
# %%
# Subsample vectors for plotting
axis = np.linspace(0, 1, 100, endpoint=True)
x, y = np.meshgrid(axis.tolist(), axis.tolist())
grid = np.c_[x.ravel(), y.ravel()]

tree = KDTree(uac)
_, idx = tree.query(grid, k=1)

plotter = Plotter(window_size=(800, 500))
plotter.add_text("Comparison of vector field samples and mean")
plotter.add_mesh(get_poly_data(V, F), color="lightgrey", opacity=0.99)
for _ in range(10):  # type: ignore
    plotter.add_arrows(
        V[idx],
        angles_to_3d_vector(
            angles=shift_angles_by_mean(sample_to_angles(prior.sample()), angles_mean)[
                idx
            ],
            x_axes=x_axes[idx],
            y_axes=y_axes[idx],
        ),
        mag=0.01,
        color="tab:blue",
    )
plotter.add_arrows(
    V[idx],
    angles_to_3d_vector(
        angles=sample_to_angles(params.mean[idx]), x_axes=x_axes[idx], y_axes=y_axes[idx]
    ),
    mag=0.015,
    color="tab:orange",
)
plotter.add_legend(labels=[["samples", "tab:blue"], ["mean", "tab:orange"]])  # type: ignore
plotter.show()

# %%
