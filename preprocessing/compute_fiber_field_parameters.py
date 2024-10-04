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
s = np.random.randint(0, V.shape[0] - 20)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
ax = plt.gca()
ax.set_aspect("equal")

add_3d_vectors_to_plot(
    V[s : (s + 20)], directions_constant_alpha[s : (s + 20)], ax, length=100, lw=1
)
add_3d_vectors_to_plot(
    V[s : (s + 20)],
    directions_constant_beta[s : (s + 20)],
    ax,
    length=100,
    lw=1,
    color="tab:orange",
)

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
