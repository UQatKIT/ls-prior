# %%
import matplotlib.pyplot as plt
import numpy as np
from pyvista import Plotter
from scipy.spatial import KDTree

from prior_fields.prior.converter import scale_mesh_to_unit_cube
from prior_fields.prior.plots import get_poly_data
from prior_fields.prior.prior import BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.fiber_grid import get_fiber_parameters_from_uac_data
from prior_fields.tensor.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)
from prior_fields.tensor.tangent_space import get_uac_basis_vectors
from prior_fields.tensor.transformer import (
    angles_to_3d_vector,
    angles_to_sample,
    sample_to_angles,
)

##################
# Read mesh data #
##################
# %%
V_raw, F, uac, _, _ = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(1)
V = scale_mesh_to_unit_cube(V_raw)

##################
# Set parameters #
##################
# %%
mean_fiber_angle, var_fiber_angle, _ = get_fiber_parameters_from_uac_data(uac, k=100)

sample_mean = angles_to_sample(mean_fiber_angle)
sigma = np.sqrt(var_fiber_angle)
ell = 0.5 * np.ones_like(sigma)

# %%
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
    scalars=var_fiber_angle,
    scalar_bar_args=dict(title="sigma^2", n_labels=2, label_font_size=12),
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
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=sigma, ell=ell, mean=sample_mean)

# %%
# validate correlation length
nrow, ncol = 4, 3
fig, ax = plt.subplots(nrow, ncol, figsize=(12, 12))
for i in range(nrow):
    for j in range(ncol):
        ax[i][j].hist(
            [mean_fiber_angle, sample_to_angles(prior.sample())],
            bins=50,
            label=["mean", "sample"],
        )
        ax[i][j].legend(prop={"size": 8})
plt.show()

#########################################
# Prior field as angles of vector field #
#########################################
# %%
sample = prior.sample()
angles = sample_to_angles(sample)
x_axes, y_axes = get_uac_basis_vectors(V, F, uac)
vector_field = angles_to_3d_vector(angles=angles, x_axes=x_axes, y_axes=y_axes)

plotter = Plotter()
plotter.add_text("Vector field sample")
plotter.add_mesh(get_poly_data(V, F), color="lightgrey", opacity=0.99)
plotter.add_arrows(V, vector_field, mag=0.01, color="tab:blue")
plotter.show(window_size=(800, 500))

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

plotter = Plotter()
plotter.add_text("Comparison of vector field samples and mean")
plotter.add_mesh(get_poly_data(V, F), color="lightgrey", opacity=0.99)
for _ in range(10):
    plotter.add_arrows(
        V[idx],
        angles_to_3d_vector(
            angles=sample_to_angles(prior.sample())[idx],
            x_axes=x_axes[idx],
            y_axes=y_axes[idx],
        ),
        mag=0.01,
        color="tab:blue",
    )
plotter.add_arrows(
    V[idx],
    angles_to_3d_vector(
        angles=mean_fiber_angle[idx], x_axes=x_axes[idx], y_axes=y_axes[idx]
    ),
    mag=0.015,
    color="tab:orange",
)
plotter.add_legend(labels=[["samples", "tab:blue"], ["mean", "tab:orange"]])
plotter.show(window_size=(800, 500))

# %%
