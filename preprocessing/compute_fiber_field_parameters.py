# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prior_fields.tensor.io import (
    get_mesh_and_point_data_from_lge_mri_based_data,
)
from prior_fields.tensor.plots import add_3d_vectors_to_plot
from prior_fields.tensor.vector_heat_method import get_uac_basis_vectors

# %%
V, F, uac, fibers = get_mesh_and_point_data_from_lge_mri_based_data(
    Path("data/LGE-MRI-based/A")
)

# %%
# This takes about 15 - 20 seconds (not fully vectorized)
directions_constant_alpha, directions_constant_beta = get_uac_basis_vectors(V, F, uac)

# %%
fig = plt.figure(figsize=(8, 8))

for i in range(4):
    s = np.random.randint(0, V.shape[0])
    V_plot = V[s].reshape(1, -1)

    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    ax = plt.gca()
    ax.set_title(f"Vertex {s}")
    ax.set_xlim(V_plot[:, 0] - 120, V_plot[:, 0] + 120)
    ax.set_ylim(V_plot[:, 1] - 120, V_plot[:, 1] + 120)
    ax.set_zlim(V_plot[:, 2] - 120, V_plot[:, 2] + 120)

    add_3d_vectors_to_plot(
        V_plot,
        directions_constant_alpha[s].reshape(1, -1),
        ax,
        length=100,
        lw=1,
        label="constant alpha",
    )
    add_3d_vectors_to_plot(
        V_plot,
        directions_constant_beta[s].reshape(1, -1),
        ax,
        length=100,
        lw=1,
        color="tab:green",
        label="constant beta",
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

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles[:3], labels[:3], loc="lower center")
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
