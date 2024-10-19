# %%

import numpy as np
from matplotlib import pyplot as plt
from pyvista import Plotter
from scipy.spatial import KDTree
from scipy.stats import circmean, circstd

from prior_fields.prior.plots import get_poly_data
from prior_fields.tensor.fiber_grid import FiberGrid
from prior_fields.tensor.mapper import map_fibers_to_tangent_space
from prior_fields.tensor.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)
from prior_fields.tensor.tangent_space_coordinates import get_uac_basis_vectors
from prior_fields.tensor.transformer import angles_between_vectors, angles_to_3d_vector

# %%
print("Read atlas mesh...\n")
V_raw, F, uac, fibers_atlas, _ = (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices("A")
)
Vmin = V_raw.min()
Vmax = V_raw.max()
V = V_raw / (Vmax - Vmin)

basis_x, basis_y = get_uac_basis_vectors(V, F, uac)
atlas_fibers = map_fibers_to_tangent_space(fibers_atlas, basis_x, basis_y)

print("\nRead mean fiber grid...\n")
fiber_grid = FiberGrid.read_from_binary_file(
    "data/LGE-MRI-based/fiber_grid_max_depth7_point_threshold100.npy"
)

# %%
print("Map mean fibers to vertices...\n")
fiber_angle_mean = np.zeros(atlas_fibers.shape[0])
fiber_angle_std = np.zeros(atlas_fibers.shape[0])
unmatched_vertices = []

for i in range(fiber_angle_mean.shape[0]):
    j = np.where(
        (uac[i, 0] >= fiber_grid.grid_x[:, 0])
        & (uac[i, 0] < fiber_grid.grid_x[:, 1])
        & (uac[i, 1] >= fiber_grid.grid_y[:, 0])
        & (uac[i, 1] < fiber_grid.grid_y[:, 1])
    )[0]
    try:
        fiber_angle_mean[i] = fiber_grid.fiber_angle_circmean[j[0]]
        fiber_angle_std[i] = fiber_grid.fiber_angle_circstd[j[0]]
    except IndexError:
        fiber_angle_std[i] = np.nan
        unmatched_vertices.append(i)

if len(unmatched_vertices) > 0:
    print(
        f"The UACs of the following vertices {len(unmatched_vertices)} / {V.shape[0]} "
        "are not covered by the fiber grid:\n",
        unmatched_vertices,
        "\nSetting their fiber mean to zero and the pointwise variance to nan.\n",
    )

mean_fibers = angles_to_3d_vector(fiber_angle_mean, basis_x, basis_y)

# %%
print("Comparison of fibers from atlas data and mean fibers mapped to atlas geometry:")
idx_notnan = (
    (np.linalg.norm(mean_fibers, axis=1) != 0)
    & (np.linalg.norm(atlas_fibers, axis=1) != 0)
    & ~np.isnan(mean_fibers).any(axis=1)
    & ~np.isnan(atlas_fibers).any(axis=1)
)
angles_between_atlas_and_mean_fiber = angles_between_vectors(atlas_fibers, mean_fibers)
min_angle = angles_between_atlas_and_mean_fiber[idx_notnan].min()
max_angle = angles_between_atlas_and_mean_fiber[idx_notnan].max()
circular_mean = circmean(
    angles_between_atlas_and_mean_fiber[idx_notnan], low=0, high=np.pi
)
circular_std = circstd(
    angles_between_atlas_and_mean_fiber[idx_notnan], low=0, high=np.pi
)

print(f"Range:\t\t({min_angle:.4f}, {max_angle:.4f})")
print(f"Circular mean:\t{circular_mean:.4f}")
print(f"Circular std:\t{circular_std:.4f}")

plt.figure(figsize=(6, 4))
plt.hist(angles_between_atlas_and_mean_fiber[idx_notnan], bins=25)
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
plotter = Plotter()
plotter.add_text("Comparison of fiber fields (subsampled)")
plotter.add_mesh(
    get_poly_data(V, F),
    scalars=fiber_angle_std,
    scalar_bar_args=dict(title="Pointwise standard deviation"),
    cmap="Blues",
)
plotter.add_arrows(
    V[idx], mean_fibers[idx], mag=0.04, color="yellow", label="Mean fibers"
)
plotter.add_arrows(
    V[idx], atlas_fibers[idx], mag=0.04, color="orange", label="Atlas fibers"
)

plotter.add_legend(size=(0.3, 0.1), loc="upper left")
plotter.show(window_size=(700, 700))

# %%
# Interactive plot of mean angle of mapped fibers
plotter = Plotter()
plotter.add_text("Mean fiber angle")
plotter.add_mesh(
    get_poly_data(V, F),
    scalars=fiber_angle_mean,
    scalar_bar_args=dict(title="Mean angle"),
    cmap="twilight",
)
plotter.show(window_size=(700, 700))

# %%
