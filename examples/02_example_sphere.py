"""
This file shows how to use the `BiLaplacianPrior` class for triangular meshes of the unit
sphere. Particularly, we introduce a sigmoid transformation and VHM-based coordinate
systems of the tangent spaces that enable us to use the bi-Laplacian prior to model
vector-valued random fields.
"""

# %%
from dolfin import BoundaryMesh, Mesh
from pyvista import Plotter

from prior_fields.parameterization.tangent_space import get_vhm_based_coordinates
from prior_fields.parameterization.transformer import (
    angles_to_3d_vector,
    sample_to_angles,
)
from prior_fields.prior.converter import function_to_numpy
from prior_fields.prior.plots import get_poly_data, plot_function
from prior_fields.prior.prior import BiLaplacianPrior

# %%
# baseline parameters
sigma = 5.0
ell = 0.5

# read mesh
mesh = Mesh("data/sphere.xml")  # tetrahedrons

# extract outer triangular surface of the mesh
sphere_mesh = BoundaryMesh(mesh, "exterior", order=False)  # triangles

# for the use in the `BiLaplacianPrior`, we have to order the faces
sphere_mesh_ordered = BoundaryMesh(mesh, "exterior", order=True)

#########################################
# bi-Laplacian prior on the unit sphere #
#########################################
prior = BiLaplacianPrior(sphere_mesh_ordered, sigma=sigma, ell=ell, seed=1)

# %%
sample = prior.sample()
plot_function(sample, title="BiLaplacianPrior sample")


############################################
# Gaussian field as angles of vector field #
############################################
# %%
# Apply sigmoid transformation to the prior sample to obtain values in (-pi/2, pi/2)
angles = sample_to_angles(function_to_numpy(sample, get_vertex_values=True))

# Extract vertices and faces from mesh
V = sphere_mesh.coordinates()
F = sphere_mesh.cells()

# Compute reference coordinates of the tangent spaces using the vector heat method
x_axes, y_axes, _ = get_vhm_based_coordinates(V, F)

# Interpret the transformed sample as angles in the VHM-based coordinate systems
vector_field = angles_to_3d_vector(angles=angles, x_axes=x_axes, y_axes=y_axes)

# %%
plotter = Plotter(window_size=(800, 500))
plotter.add_text("Vector field on sphere with reference coordinate systems")

plotter.add_mesh(get_poly_data(V, F), color="lightgrey", opacity=0.9)

plotter.add_arrows(V, x_axes, mag=0.1, color="tab:green", label="x-axes")
plotter.add_arrows(V, y_axes, mag=0.1, color="tab:green", label="y-axes")
plotter.add_arrows(V, vector_field, mag=0.1, color="tab:red", label="vector field")

plotter.add_legend(size=(0.3, 0.1), loc="lower left")  # type: ignore
plotter.show()

# %%
