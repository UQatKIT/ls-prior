# %%
import numpy as np
from dolfin import BoundaryMesh, Mesh
from pyvista import Plotter, PolyData

from prior_fields.prior.converter import function_to_numpy
from prior_fields.prior.plots import plot_function
from prior_fields.prior.prior import BiLaplacianPrior
from prior_fields.tensor.transformer import angles_to_3d_vector, sample_to_alpha
from prior_fields.tensor.vector_heat_method import get_reference_coordinates

# %%
sigma = 0.1
ell = 0.5

mesh = Mesh("data/sphere.xml")  # tetrahedrons
sphere_mesh = BoundaryMesh(mesh, "exterior", order=False)  # triangles
sphere_mesh_ordered = BoundaryMesh(mesh, "exterior", order=True)

prior = BiLaplacianPrior(sphere_mesh_ordered, sigma=sigma, ell=ell, seed=1)

# %%
sample = prior.sample()
plot_function(sample, title="BiLaplacianPrior sample")


############################################
# Gaussian field as angles of vector field #
############################################
# %%
alphas = sample_to_alpha(function_to_numpy(sample))

V = sphere_mesh.coordinates()
F = sphere_mesh.cells()

x_axes, y_axes, _ = get_reference_coordinates(V, F)
vector_field = angles_to_3d_vector(alphas=alphas, x_axes=x_axes, y_axes=y_axes)

# %%
plotter = Plotter()
plotter.add_text("Vector field on sphere with reference coordinate systems")

mesh = PolyData(V, np.hstack((np.full((F.shape[0], 1), 3), F)))
plotter.add_mesh(mesh, color="lightgrey", opacity=0.9)

plotter.add_arrows(V, x_axes, mag=0.1, color="tab:green", label="x-axes")
plotter.add_arrows(V, y_axes, mag=0.1, color="tab:green", label="y-axes")
plotter.add_arrows(V, vector_field, mag=0.1, color="tab:red", label="vector field")

plotter.add_legend(size=(0.3, 0.1), loc="lower left")
plotter.show(window_size=(800, 500))

# %%
