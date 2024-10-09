# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, interact
from scipy.spatial import KDTree

from prior_fields.prior.converter import numpy_to_function
from prior_fields.prior.plots import plot_function
from prior_fields.prior.prior import BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.io import get_mesh_and_point_data_from_lge_mri_based_data

# %%
V_raw, F, uac, fibers_atlas, tags = get_mesh_and_point_data_from_lge_mri_based_data(
    Path("data/LGE-MRI-based/A")
)
Vmin = V_raw.min()
Vmax = V_raw.max()
V = V_raw / (Vmax - Vmin)

# %%
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=0.2, ell=0.2)
sample = prior.sample()
plot_function(numpy_to_function(sample, prior._prior.Vh))

# %%
V_raw, F1, uac1, fibers1, tags1 = get_mesh_and_point_data_from_lge_mri_based_data(
    Path("data/LGE-MRI-based/1")
)
Vmin = V_raw.min()
Vmax = V_raw.max()
V1 = V_raw / (Vmax - Vmin)

# %%
axis = np.linspace(0, 1, 100, endpoint=True)
x, y = np.meshgrid(axis.tolist(), axis.tolist())
grid = np.c_[x.ravel(), y.ravel()]

tree = KDTree(uac)
_, idx = tree.query(grid, k=1)

tree1 = KDTree(uac1)
_, idx1 = tree1.query(grid, k=1)


# %%
@interact(azim=IntSlider(value=20, min=-180, max=180, step=5, description="azim"))
def plot_sample_orginial_geometry(azim):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")
    ax.view_init(azim=azim, elev=20)

    c = ax.scatter(*[V[idx, i] for i in range(3)], c=sample[idx], s=1)  # type: ignore

    plt.colorbar(c)
    plt.show()


# %%
@interact(azim=IntSlider(value=20, min=-180, max=180, step=5, description="azim"))
def plot_sample_different_geometry(azim):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")
    ax.view_init(azim=azim, elev=20)

    c = ax.scatter(*[V1[idx1, i] for i in range(3)], c=sample[idx], s=1)  # type: ignore

    plt.colorbar(c)
    plt.show()


# %%
