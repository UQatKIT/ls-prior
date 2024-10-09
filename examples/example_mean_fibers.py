# %%
from pathlib import Path

import numpy as np
from ipywidgets import IntSlider, interact
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import circmean, circstd

from prior_fields.tensor.io import get_mesh_and_point_data_from_lge_mri_based_data
from prior_fields.tensor.mapper import FiberGrid, map_fibers_to_tangent_space
from prior_fields.tensor.plots import add_3d_vectors_to_plot
from prior_fields.tensor.transformer import _angles_between_vectors, vectors_3d_to_angles
from prior_fields.tensor.vector_heat_method import get_uac_basis_vectors

# %%
print("Read atlas mesh...\n")
V_raw, F, uac, fibers_atlas, tags = get_mesh_and_point_data_from_lge_mri_based_data(
    Path("data/LGE-MRI-based/A")
)
Vmin = V_raw.min()
Vmax = V_raw.max()
V = V_raw / (Vmax - Vmin)

basis_x, basis_y = get_uac_basis_vectors(V, F, uac)
fibers = map_fibers_to_tangent_space(fibers_atlas, basis_x, basis_y)

print("\nRead mean fiber grid...\n")
fiber_grid = FiberGrid.read_from_binary_file(
    "data/LGE-MRI-based/fiber_grid_normalized_max_depth7_point_threshold100.npy"
)

# %%
print("Map mean fibers to vertices...\n")
fiber_mean = np.zeros_like(fibers)
unmatched_vertices = []

for i in range(fibers.shape[0]):
    j = np.where(
        (uac[i, 0] >= fiber_grid.grid_x[:, 0])
        & (uac[i, 0] < fiber_grid.grid_x[:, 1])
        & (uac[i, 1] >= fiber_grid.grid_y[:, 0])
        & (uac[i, 1] < fiber_grid.grid_y[:, 1])
    )[0]
    try:
        fiber_mean[i] = (
            fiber_grid.fiber_coeff_x_mean[j[0]] * basis_x[i]
            + fiber_grid.fiber_coeff_y_mean[j[0]] * basis_y[i]
        )
    except IndexError:
        unmatched_vertices.append(i)

print(
    f"The UACs of the following vertices {len(unmatched_vertices)} "
    "are not covered by the fiber grid:\n",
    unmatched_vertices,
    "\nSetting their fiber mean to zero.\n",
)

# %%
print("Comparison of fibers from atlas data and mean fibers mapped to atlas geometry:")
idx_non_zero = (fiber_mean.mean(axis=1) != 0) & (fibers.mean(axis=1) != 0)
angles_between_atlas_and_mean_fiber = _angles_between_vectors(
    fibers[idx_non_zero], fiber_mean[idx_non_zero]
)
angles_between_atlas_and_mean_fiber = angles_between_atlas_and_mean_fiber[
    ~np.isnan(angles_between_atlas_and_mean_fiber)
]
print(
    f"Range:\t\t({angles_between_atlas_and_mean_fiber.min():.4f}, "
    f"{angles_between_atlas_and_mean_fiber.max():.4f})"
)
print(f"Circular mean:\t{circmean(angles_between_atlas_and_mean_fiber):.4f}")
print(f"Circular std:\t{circstd(angles_between_atlas_and_mean_fiber):.4f}")

fig = plt.figure(figsize=(10, 10))

for i in range(9):
    s = np.random.randint(0, V.shape[0])
    V_plot = V[s].reshape(1, -1)

    ax = fig.add_subplot(3, 3, i + 1, projection="3d")
    ax.set_title(f"Vertex {s}")
    ax.set_xlim(V_plot[:, 0] - 120, V_plot[:, 0] + 120)
    ax.set_ylim(V_plot[:, 1] - 120, V_plot[:, 1] + 120)
    ax.set_zlim(V_plot[:, 2] - 120, V_plot[:, 2] + 120)

    add_3d_vectors_to_plot(
        V_plot,
        fibers[s].reshape(1, -1),
        ax,
        length=100,
        lw=1,
        label="Atlas fiber",
    )
    add_3d_vectors_to_plot(
        V_plot,
        fiber_mean[s].reshape(1, -1),
        ax,
        length=100,
        lw=1,
        color="tab:orange",
        label="Mean fiber",
    )

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles[:2], labels[:2], loc="upper right")
fig.suptitle("Fiber from atlas vs. UAC-based fiber mean over 7 geometries")
plt.show()

# %%
# Interactive plot of subsamble of mapped fibers
axis = np.linspace(0, 1, 50, endpoint=True)
x, y = np.meshgrid(axis.tolist(), axis.tolist())
grid = np.c_[x.ravel(), y.ravel()]

tree = KDTree(uac)
_, idx = tree.query(grid, k=1)


@interact(
    elev=IntSlider(value=20, min=-180, max=180, step=1, description="elev"),
    azim=IntSlider(value=-60, min=-180, max=180, step=1, description="azim"),
)
def plot_mean_fiber_field(elev, azim):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")
    ax.view_init(elev=elev, azim=azim)

    add_3d_vectors_to_plot(V[idx], fiber_mean[idx], ax, length=0.05, lw=0.5)

    plt.title("Mean fiber field (subsampled)")
    plt.show()


# %%
# Interactive plot of mean angle of mapped fibers
mean_angle = vectors_3d_to_angles(fiber_mean, basis_x, basis_y)


@interact(azim=IntSlider(value=80, min=-180, max=180, step=5, description="azim"))
def plot_mean_angle(azim):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")
    ax.view_init(elev=20, azim=azim)

    c = ax.scatter(*[V[:, i] for i in range(3)], c=mean_angle, s=0.1)

    plt.colorbar(c)
    plt.title("Mean fiber angle")
    plt.show()


# %%
