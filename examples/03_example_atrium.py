"""
This file shows how to use the `BiLaplacianPriorNumpyWrapper` class for triangular meshes
of the left atrium. We focus on the data-informed parameterization computed using
```console
pixi run -- python prior_fields/tensor/scripts/02a_compute_prior_parameters.py 1
```
As in `02_example_sphere.py`, we transform the scalar-valued prior to vector-valued
outputs, which we use as fiber fields in the context of modeling the conduction velocity
tensor on the atrial surface.
"""

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
# Read vertices, faces and UACs
V, F, uac, _, _ = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(geometry)
V = scale_mesh_to_unit_cube(V)

# Load data-informed parameters computed in
# `prior_fields/tensor/scripts/02a_compute_prior_parameters.py`
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
# Initialize zero-mean prior with data-informed pointwise variance and constant
# correlation length of 0.2
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=params.sigma, ell=params.ell)

#########################################
# Prior field as angles of vector field #
#########################################
# %%
# Sample from the prior
sample = prior.sample()

# Transform the sample and the data-informed mean to the proper range of angles
angles_around_zero = sample_to_angles(sample)
angles_mean = sample_to_angles(params.mean)

# Shift the angles by the mean parameter field
# Note that we do not use the mean field as mean of the `BiLaplacianPrior` since then
# we would not properly account for periodicity in the angles and distributions for mean
# values close to -pi/2 or pi/2 would be significantly skewed.
angles = shift_angles_by_mean(angles_around_zero, angles_mean)

# Compute VHM-based reference coordinates
x_axes, y_axes, _ = get_vhm_based_coordinates(V, F)

# Interpret the sample as vector field using the VHM-based coordinates
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
# Subsample vectors for plotting:
# 1. Split the UAC unit square in 100x100 equal squares
axis = np.linspace(0, 1, 100, endpoint=True)
x, y = np.meshgrid(axis.tolist(), axis.tolist())
grid = np.c_[x.ravel(), y.ravel()]

# 2. In each grid cell, find a vertex of the atrial geometry based on the UACs
tree = KDTree(uac)
_, idx = tree.query(grid, k=1)

# Plot the mean fiber field together with 10 transformed samples from the prior
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
