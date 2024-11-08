# %%
from dolfin import BoundaryMesh, Mesh
from pyvista import Plotter

from prior_fields.prior.converter import function_to_numpy
from prior_fields.prior.plots import get_poly_data, plot_function
from prior_fields.prior.prior import BiLaplacianPrior
from prior_fields.tensor.tangent_space import get_vhm_based_coordinates
from prior_fields.tensor.transformer import angles_to_3d_vector, sample_to_angles

# %%
sigma = 5.0
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
angles = sample_to_angles(function_to_numpy(sample, get_vertex_values=True))

V = sphere_mesh.coordinates()
F = sphere_mesh.cells()

x_axes, y_axes, _ = get_vhm_based_coordinates(V, F)
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
