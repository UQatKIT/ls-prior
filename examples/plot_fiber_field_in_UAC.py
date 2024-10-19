# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyvista import Plotter, PolyData
from scipy.stats import circmean, circstd, mode

from prior_fields.tensor.io import get_mesh_and_point_data_from_lge_mri_based_data
from prior_fields.tensor.mapper import get_coefficients, map_fibers_to_tangent_space
from prior_fields.tensor.tangent_space_coordinates import get_uac_basis_vectors
from prior_fields.tensor.transformer import vector_coefficients_2d_to_angles

# %%
V, F, uac, fibers, tags = get_mesh_and_point_data_from_lge_mri_based_data(
    Path("data/LGE-MRI-based/A")
)

directions_constant_beta, directions_constant_alpha = get_uac_basis_vectors(V, F, uac)
fibers_in_tangent_space = map_fibers_to_tangent_space(
    fibers, directions_constant_beta, directions_constant_alpha
)

# %%
# Sanity check
plotter = Plotter()
plotter.add_text("Original fibers vs. fibers mapped to tangent space")

mesh = PolyData(V, np.hstack((np.full((F.shape[0], 1), 3), F)))
plotter.add_mesh(mesh, color="lightgrey", opacity=0.9)

length = 200
plotter.add_arrows(
    V, directions_constant_beta, mag=length, color="tab:blue", label="d_alpha"
)
plotter.add_arrows(
    V, directions_constant_alpha, mag=length, color="tab:green", label="d_beta"
)
plotter.add_arrows(V, fibers, mag=length, color="tab:orange", label="fibers")
plotter.add_arrows(
    V,
    fibers_in_tangent_space,
    mag=length,
    color="tab:red",
    label="fibers mapped to tangent space",
)

plotter.add_legend(size=(0.5, 0.2), loc="lower left")
plotter.show(window_size=(800, 500))

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
# Get fiber angle within (-pi/2, pi/2] in UAC system
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
