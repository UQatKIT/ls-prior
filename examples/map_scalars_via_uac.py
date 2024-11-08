# %%
from pyvista import Plotter
from scipy.spatial import KDTree

from prior_fields.prior.converter import scale_mesh_to_unit_cube
from prior_fields.prior.plots import get_poly_data
from prior_fields.prior.prior import BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)

# %%
V_raw, F, uac, fibers_atlas, _ = (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices("A")
)
V = scale_mesh_to_unit_cube(V_raw)

# %%
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=0.2, ell=1.0)
sample = prior.sample()

# %%
V1_raw, F1, uac1, fibers1, _ = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(
    1
)
V1 = scale_mesh_to_unit_cube(V1_raw)

tree = KDTree(uac)
_, idx_neighbors = tree.query(uac1, k=1)
sample1 = sample[idx_neighbors]

# %%
mesh = get_poly_data(V, F)
mesh["sample"] = sample

mesh1 = get_poly_data(V1, F1)
mesh1["sample"] = sample1

plotter = Plotter(shape=(1, 2), window_size=(800, 400))

plotter.subplot(0, 0)
plotter.add_text("Sample on atlas geometry")
plotter.add_mesh(mesh)

plotter.subplot(0, 1)
plotter.add_text("Sample on geometry 1")
plotter.add_mesh(mesh1)

plotter.show()

# %%
