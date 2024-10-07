# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import circmean, circstd, mode

from prior_fields.prior.dtypes import ArrayNx3
from prior_fields.tensor.io import read_meshes_from_lge_mri_data
from prior_fields.tensor.mapper import get_coefficients, map_fibers_to_tangent_space
from prior_fields.tensor.transformer import vector_coefficients_2d_to_angles
from prior_fields.tensor.vector_heat_method import get_uac_basis_vectors

# %%
V, F, uac, fibers, tags = read_meshes_from_lge_mri_data()

# %%
fibers_in_tangent_space: dict[int, ArrayNx3] = dict()
fiber_coeffs_x1: dict[int, ArrayNx3] = dict()
fiber_coeffs_x2: dict[int, ArrayNx3] = dict()
fiber_angles: dict[int, ArrayNx3] = dict()

# This takes about 75 - 90 seconds
for i in V.keys():
    print(f"Geometry {i}:")
    print("Get UAC-based tangent space coordinates.")
    directions_constant_beta, directions_constant_alpha = get_uac_basis_vectors(
        V[i], F[i], uac[i]
    )
    print("Map fibers to tangent space.")
    fibers_in_tangent_space[i] = map_fibers_to_tangent_space(
        fibers[i], directions_constant_beta, directions_constant_alpha
    )
    print("Get coefficients of fibers in tangent space coordinates.")
    fiber_coeffs_x1[i], fiber_coeffs_x2[i] = get_coefficients(
        fibers[i], directions_constant_beta, directions_constant_alpha
    )
    print("Get fiber angle within (-pi, pi] in UAC system.")
    fiber_angles[i] = vector_coefficients_2d_to_angles(
        fiber_coeffs_x1[i], fiber_coeffs_x2[i]
    )
    print()

# %%
# Collect data from different geometries in single array for mean computation
uac_array = np.vstack([uac[i] for i in np.arange(1, 8)])
fiber_coeffs_x1_array = np.hstack([fiber_coeffs_x1[i] for i in np.arange(1, 8)])
fiber_coeffs_x2_array = np.hstack([fiber_coeffs_x2[i] for i in np.arange(1, 8)])
fiber_angles_array = np.hstack([fiber_angles[i] for i in np.arange(1, 8)])
tags_array = np.hstack([tags[i] for i in np.arange(1, 8)])

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
            (uac_array[:, 0] >= grid[i])
            & (uac_array[:, 0] < grid[i + 1])
            & (uac_array[:, 1] >= grid[j])
            & (uac_array[:, 1] < grid[j + 1])
        )
        fiber_coeff_mean_x1[j, i] = fiber_coeffs_x1_array[mask].mean()
        fiber_coeff_mean_x2[j, i] = fiber_coeffs_x2_array[mask].mean()
        phi_circmean[j, i] = circmean(fiber_angles_array[mask], high=np.pi, low=-np.pi)
        phi_circstd[j, i] = circstd(fiber_angles_array[mask], high=np.pi, low=-np.pi)
        tag_mode[j, i] = mode(tags_array[mask]).mode

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect("equal")
plt.scatter(*np.meshgrid(grid, grid), c=tag_mode, s=1000 / h, marker="s", alpha=0.3)
ax.quiver(
    *np.meshgrid(grid, grid),
    fiber_coeff_mean_x1,
    fiber_coeff_mean_x2,
    angles="xy",
    scale_units="xy",
    scale=50,
    width=0.001,
)
plt.title("Mean of fibers in UAC over 7 geometries")
plt.savefig("figures/uac_fibers_mean.svg")
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect("equal")
c = plt.scatter(
    *np.meshgrid(grid, grid), c=phi_circmean, s=1000 / h, marker="s", alpha=0.3
)
ax.quiver(
    *np.meshgrid(grid, grid),
    fiber_coeff_mean_x1,
    fiber_coeff_mean_x2,
    angles="xy",
    scale_units="xy",
    scale=50,
    width=0.001,
)
plt.colorbar(c)
plt.title("Circular mean of fiber angle")
plt.savefig("figures/uac_fibers_with_circmean.svg")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect("equal")
c = plt.scatter(
    *np.meshgrid(grid, grid), c=phi_circstd, s=1000 / h, marker="s", alpha=0.3
)
ax.quiver(
    *np.meshgrid(grid, grid),
    fiber_coeff_mean_x1,
    fiber_coeff_mean_x2,
    angles="xy",
    scale_units="xy",
    scale=50,
    width=0.001,
)
plt.colorbar(c)
plt.title("Circular standard deviation of fiber angle")
plt.savefig("figures/uac_fibers_with_circstd.svg")
plt.show()
