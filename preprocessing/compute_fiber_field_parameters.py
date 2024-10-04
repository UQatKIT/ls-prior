# %%
import matplotlib.pyplot as plt
import numpy as np

from prior_fields.tensor.io import read_meshes_from_lge_mri_data
from prior_fields.tensor.plots import add_3d_vectors_to_plot
from prior_fields.tensor.vector_heat_method import get_uac_basis_vectors

# %%
V, F, uac, fibers = read_meshes_from_lge_mri_data()

# %%
directions_constant_alpha, directions_constant_beta = get_uac_basis_vectors(
    V[0], F[0], uac[0]
)

# %%
s = np.random.randint(0, V[0].shape[0] - 20)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
ax = plt.gca()
ax.set_aspect("equal")

add_3d_vectors_to_plot(
    V[0][s : (s + 20)], directions_constant_alpha[s : (s + 20)], ax, length=100, lw=1
)
add_3d_vectors_to_plot(
    V[0][s : (s + 20)],
    directions_constant_beta[s : (s + 20)],
    ax,
    length=100,
    lw=1,
    color="tab:orange",
)

plt.show()

# %%
