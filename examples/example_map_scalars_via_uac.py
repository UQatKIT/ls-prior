# %%
from pathlib import Path

from pyvista import Plotter
from scipy.spatial import KDTree

from prior_fields.prior.plots import get_poly_data
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
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=0.2, ell=1.0)
sample = prior.sample()

# %%
V_raw, F1, uac1, fibers1, tags1 = get_mesh_and_point_data_from_lge_mri_based_data(
    Path("data/LGE-MRI-based/1")
)
Vmin = V_raw.min()
Vmax = V_raw.max()
V1 = V_raw / (Vmax - Vmin)

tree = KDTree(uac)
_, idx_neighbors = tree.query(uac1, k=1)
sample1 = sample[idx_neighbors]

# %%
mesh = get_poly_data(V, F)
mesh["sample"] = sample

mesh1 = get_poly_data(V1, F1)
mesh1["sample"] = sample1

plotter = Plotter(shape=(1, 2))

plotter.subplot(0, 0)
plotter.add_text("Sample on atlas geometry")
plotter.add_mesh(mesh)

plotter.subplot(0, 1)
plotter.add_text("Sample on geometry 1")
plotter.add_mesh(mesh1)

plotter.show(window_size=(800, 400))
