# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import circmean, circstd, mode

from prior_fields.tensor.io import get_mesh_and_point_data_from_lge_mri_based_data
from prior_fields.tensor.mapper import get_coefficients, map_fibers_to_tangent_space
from prior_fields.tensor.plots import add_3d_vectors_to_plot
from prior_fields.tensor.transformer import vector_coefficients_2d_to_angles
from prior_fields.tensor.vector_heat_method import get_uac_basis_vectors

# %%
V, F, uac, fibers, tags = get_mesh_and_point_data_from_lge_mri_based_data(
    Path("data/LGE-MRI-based/A")
)

# %%
# This takes about 15 - 20 seconds (not fully vectorized)
directions_constant_beta, directions_constant_alpha = get_uac_basis_vectors(V, F, uac)

# %%
fibers_in_tangent_space = map_fibers_to_tangent_space(
    fibers, directions_constant_beta, directions_constant_alpha
)

# %%
fig = plt.figure(figsize=(8, 8))

for i in range(4):
    s = np.random.randint(0, V.shape[0])
    V_plot = V[s].reshape(1, -1)

    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    ax.set_title(f"Vertex {s}")
    ax.set_xlim(V_plot[:, 0] - 120, V_plot[:, 0] + 120)
    ax.set_ylim(V_plot[:, 1] - 120, V_plot[:, 1] + 120)
    ax.set_zlim(V_plot[:, 2] - 120, V_plot[:, 2] + 120)

    add_3d_vectors_to_plot(
        V_plot,
        directions_constant_beta[s].reshape(1, -1),
        ax,
        length=100,
        lw=1,
        label="constant beta",
    )
    add_3d_vectors_to_plot(
        V_plot,
        directions_constant_alpha[s].reshape(1, -1),
        ax,
        length=100,
        lw=1,
        color="tab:green",
        label="constant alpha",
    )
    add_3d_vectors_to_plot(
        V_plot,
        fibers[s].reshape(1, -1),
        ax,
        length=100,
        lw=1,
        color="tab:orange",
        label="fiber",
    )
    add_3d_vectors_to_plot(
        V_plot,
        fibers_in_tangent_space[s].reshape(1, -1),
        ax,
        length=100,
        lw=1,
        color="tab:red",
        label="fiber mapped to tangent space",
    )

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles[:4], labels[:4], loc="lower center")
fig.suptitle("UAC based tangent space coordinates and fibers")
plt.show()

# %%
critical_value = 0.8
count_almost_parallel = 0
for i in range(V.shape[0]):
    if abs(directions_constant_beta[i] @ directions_constant_alpha[i]) > critical_value:
        count_almost_parallel += 1
print(
    f"{100 * count_almost_parallel / V.shape[0]:.2f}% of the bases vectors are "
    f"almost parallel (scalar product > {critical_value})."
)

# %%
fiber_coeffs_x1, fiber_coeffs_x2 = get_coefficients(
    fibers, directions_constant_beta, directions_constant_alpha
)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect("equal")
c = ax.scatter(uac[:, 0], uac[:, 1], c=fiber_coeffs_x1, s=0.5)
fig.colorbar(c)
plt.title("alpha-coefficients of fibers in UAC basis")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect("equal")
c = ax.scatter(uac[:, 0], uac[:, 1], c=fiber_coeffs_x2, s=0.5)
fig.colorbar(c)
plt.title("beta-coefficients of fibers in UAC basis")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect("equal")
ax.quiver(
    uac[:, 0],
    uac[:, 1],
    fiber_coeffs_x1,
    fiber_coeffs_x2,
    angles="xy",
    scale_units="xy",
    scale=80,
    width=0.001,
)
plt.title("Fibers mapped to UAC")
plt.savefig("figures/uac_fibers.svg")
plt.show()

# %%
# Get fiber angle within (-pi, pi] in UAC system
phi = vector_coefficients_2d_to_angles(fiber_coeffs_x1, fiber_coeffs_x2)

# %%
# Split UAC unit square into grid and compute component-wise parameters (fibers/tags)
h = 100
grid = np.linspace(0, 1, h, endpoint=True)
grid[-1] += 1e-6  # include alpha, beta = 1
fiber_coeff_mean_x1 = np.zeros((h, h))
fiber_coeff_mean_x2 = np.zeros((h, h))
phi_circmean = np.zeros((h, h))
phi_circstd = np.zeros((h, h))
tag_mode = np.zeros((h, h))

for i in range(len(grid) - 1):
    for j in range(len(grid) - 1):
        mask = (
            (uac[:, 0] >= grid[i])
            & (uac[:, 0] < grid[i + 1])
            & (uac[:, 1] >= grid[j])
            & (uac[:, 1] < grid[j + 1])
        )
        fiber_coeff_mean_x1[j, i] = fiber_coeffs_x1[mask].mean()
        fiber_coeff_mean_x2[j, i] = fiber_coeffs_x2[mask].mean()
        phi_circmean[j, i] = circmean(phi[mask], low=-np.pi / 2, high=np.pi / 2)
        phi_circstd[j, i] = circstd(phi[mask], low=-np.pi / 2, high=np.pi / 2)
        tag_mode[j, i] = mode(tags[mask]).mode

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect("equal")
plt.scatter(*np.meshgrid(grid, grid), c=tag_mode, s=10, marker="s", alpha=0.3)
ax.quiver(
    *np.meshgrid(grid, grid),
    fiber_coeff_mean_x1,
    fiber_coeff_mean_x2,
    angles="xy",
    scale_units="xy",
    scale=80,
    width=0.001,
)
plt.title("Fibers in UAC over grid with anatomical structures")
plt.savefig("figures/uac_fibers_discretized.svg")
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect("equal")
plt.scatter(*np.meshgrid(grid, grid), c=phi_circmean, s=10, marker="s", alpha=0.3)
ax.quiver(
    *np.meshgrid(grid, grid),
    fiber_coeff_mean_x1,
    fiber_coeff_mean_x2,
    angles="xy",
    scale_units="xy",
    scale=80,
    width=0.001,
)
plt.title("Circular mean of fiber angle")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect("equal")
plt.scatter(*np.meshgrid(grid, grid), c=phi_circstd, s=10, marker="s", alpha=0.3)
ax.quiver(
    *np.meshgrid(grid, grid),
    fiber_coeff_mean_x1,
    fiber_coeff_mean_x2,
    angles="xy",
    scale_units="xy",
    scale=80,
    width=0.001,
)
plt.title("Circular standard deviation of fiber angle")
plt.show()
