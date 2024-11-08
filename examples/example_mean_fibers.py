# %%


import matplotlib.pyplot as plt
import numpy as np
from pyvista import Plotter
from scipy.spatial import KDTree
from scipy.stats import circmean, circvar

from prior_fields.prior.converter import scale_mesh_to_unit_cube
from prior_fields.prior.plots import get_poly_data
from prior_fields.tensor.parameterization import (
    Geometry,
    get_fiber_parameters_from_uac_grid,
)
from prior_fields.tensor.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)
from prior_fields.tensor.tangent_space import (
    get_uac_based_coordinates,
    map_fibers_to_tangent_space,
)
from prior_fields.tensor.transformer import angles_between_vectors, angles_to_3d_vector

# %%
# Read atlas data
V_raw, F, uac, fibers_atlas, _ = (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(Geometry("A"))
)
V = scale_mesh_to_unit_cube(V_raw)

basis_x, basis_y = get_uac_based_coordinates(V, F, uac)
atlas_fibers = map_fibers_to_tangent_space(fibers_atlas, basis_x, basis_y)

# Get mean fibers and angle mean/variance
fiber_angle_mean, fiber_angle_var = get_fiber_parameters_from_uac_grid(
    uac, file="data/fiber_grid_max_depth8_point_threshold120.npy"
)
mean_fibers = angles_to_3d_vector(fiber_angle_mean, basis_x, basis_y)

# %%
print("Angles between fibers from atlas data and mean fibers mapped to atlas geometry:")
angles_between_atlas_and_mean_fiber = angles_between_vectors(atlas_fibers, mean_fibers)
min_angle = angles_between_atlas_and_mean_fiber.min()
max_angle = angles_between_atlas_and_mean_fiber.max()
circular_mean = circmean(
    angles_between_atlas_and_mean_fiber, low=0, high=np.pi, nan_policy="omit"
)
circular_var = circvar(
    angles_between_atlas_and_mean_fiber, low=0, high=np.pi, nan_policy="omit"
)

print(f"Range:\t\t({min_angle:.4f}, {max_angle:.4f})")
print(f"Circular mean:\t{circular_mean:.4f}")
print(f"Circular var:\t{circular_var:.4f}")

plt.figure(figsize=(6, 4))
plt.hist(angles_between_atlas_and_mean_fiber, bins=25)
plt.title("Histogram of angles between atlas fibers and mean fibers")
plt.show()

# %%
# Subsample vectors for plotting
axis = np.linspace(0, 1, 30, endpoint=True)
x, y = np.meshgrid(axis.tolist(), axis.tolist())
grid = np.c_[x.ravel(), y.ravel()]

tree = KDTree(uac)
_, idx = tree.query(grid, k=1)

# %%
# Interactive plot of fiber fields
plotter = Plotter(window_size=(700, 700))
plotter.add_text("Comparison of fiber fields (subsampled)")
plotter.add_mesh(
    get_poly_data(V, F),
    scalars=fiber_angle_var,
    scalar_bar_args=dict(title="Pointwise variance"),
    cmap="Blues",
)
plotter.add_arrows(
    V[idx], mean_fibers[idx], mag=0.04, color="yellow", label="Mean fibers"
)
plotter.add_arrows(
    V[idx], atlas_fibers[idx], mag=0.04, color="orange", label="Atlas fibers"
)

plotter.add_legend(size=(0.3, 0.1), loc="upper left")  # type: ignore
plotter.show()

# %%
# Interactive plot of mean angle of mapped fibers
plotter = Plotter(window_size=(700, 700))
plotter.add_text("Mean fiber angle")
plotter.add_mesh(
    get_poly_data(V, F),
    scalars=fiber_angle_mean,
    scalar_bar_args=dict(title="Mean angle"),
    cmap="twilight",
)
plotter.show()

# %%
