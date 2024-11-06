# %%
from pathlib import Path

from dolfin import BoundaryMesh, Expression, FunctionSpace, Mesh, UnitSquareMesh
from matplotlib import rc
from pyvista import Plotter, global_theme

from prior_fields.prior.converter import (
    expression_to_vector,
    scale_mesh_to_unit_cube,
    str_to_function,
    str_to_vector,
)
from prior_fields.prior.plots import get_poly_data, plot_function, plot_numpy_sample
from prior_fields.prior.prior import BiLaplacianPrior, BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.parameters import Geometry, PriorParameters
from prior_fields.tensor.plots import initialize_vector_field_plotter
from prior_fields.tensor.reader import (
    read_all_human_atrial_fiber_meshes,
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)
from prior_fields.tensor.tangent_space import get_reference_coordinates
from prior_fields.tensor.transformer import angles_to_3d_vector, sample_to_angles

global_theme.font.family = "times"
global_theme.font.size = 20
global_theme.font.title_size = 20
global_theme.font.label_size = 20

font = {"family": "times", "size": 20}
rc("font", **font)

###############
# Unit square #
###############
# %%
# Minimal working example:
# zero-mean, stationary, isotropic BiLaplacianPrior on UnitSquareMesh with 4x4 vertices
mesh = UnitSquareMesh(3, 3)
prior = BiLaplacianPrior(mesh, sigma=0.1, ell=0.1, seed=1)
sample = prior.sample()
plot_function(
    sample, show_mesh=True, file="figures/priors/square_with_mesh.png", labsize=15
)

# %%
mesh = UnitSquareMesh(64, 64)
mean_zero = str_to_vector("0", mesh)

# %%
# Baseline: zero-mean, stationary, isotropic BiLaplacianPrior
prior = BiLaplacianPrior(mesh, sigma=0.2, ell=0.1, mean=mean_zero, seed=1)
sample = prior.sample()
plot_function(sample, file="figures/priors/square_baseline.png", vmin=-2.0, vmax=2.0)

prior = BiLaplacianPrior(mesh, sigma=0.1, ell=0.1, mean=mean_zero, seed=1)
sample = prior.sample()
plot_function(
    sample, file="figures/priors/square_baseline_smaller_sigma.png", vmin=-2.0, vmax=2.0
)

prior = BiLaplacianPrior(mesh, sigma=0.2, ell=0.02, mean=mean_zero, seed=1)
sample = prior.sample()
plot_function(sample, file="figures/priors/square_baseline_smaller_ell.png")

# %%
# non-zero mean
mean_str = "2*(x[0]+x[1])"
plot_function(str_to_function(mean_str, mesh), title="Mean function")

prior = BiLaplacianPrior(
    mesh, sigma=0.2, ell=0.1, mean=str_to_vector(mean_str, mesh), seed=1
)
sample = prior.sample()
plot_function(sample, file="figures/priors/square_non-zero_mean.png")

# %%
# pointwise variance
sigma_non_stationary = expression_to_vector(
    Expression("0.4-0.38*x[0]", degree=1), FunctionSpace(mesh, "CG", 1)
)
prior = BiLaplacianPrior(mesh, sigma=sigma_non_stationary, ell=0.1, seed=1)
sample = prior.sample()
plot_function(sample, file="figures/priors/square_pointwise_sigma.png")

###############
# Unit sphere #
###############
# %%
mesh = Mesh("data/sphere.xml")  # tetrahedrons
sphere_mesh = BoundaryMesh(mesh, "exterior", order=False)  # triangles
sphere_mesh_ordered = BoundaryMesh(mesh, "exterior", order=True)
V = sphere_mesh.coordinates()
F = sphere_mesh.cells()
x_axes, y_axes, _ = get_reference_coordinates(V, F)

prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=1.0, ell=0.1, seed=1)
sample = prior.sample()
plot_numpy_sample(sample, V, F, file="figures/priors/sphere_prior_scalar_field.eps")

# %%
plotter = initialize_vector_field_plotter(get_poly_data(V, F))
plotter.add_arrows(V, x_axes, mag=0.1, color="#009682")
plotter.add_arrows(V, y_axes, mag=0.1, color="#009682")
plotter.save_graphic(filename="figures/priors/sphere_vector_heat_method.eps")
plotter.show()

# %%
vector_field = angles_to_3d_vector(
    angles=sample_to_angles(sample), x_axes=x_axes, y_axes=y_axes
)

plotter = initialize_vector_field_plotter(get_poly_data(V, F))
plotter.add_arrows(V, vector_field, mag=0.16, color="#009682")
plotter.save_graphic(filename="figures/priors/sphere_prior_vector_field.eps")
plotter.show()


##########
# Atrium #
##########
# %%
V_dict, F_dict, uac_dict, fibers_dict, tags_dict = read_all_human_atrial_fiber_meshes()
keys = sorted(V_dict.keys())

# %%
plotter = Plotter(shape="3/4")

for i in keys:
    plotter.subplot(i - 1)
    plotter.add_text(f"Geometry {i}")
    plotter.add_mesh(get_poly_data(V_dict[i], F_dict[i]))
    plotter.camera.zoom(1.4)

plotter.save_graphic(filename="figures/atrial_geometries.eps")
plotter.show()

# %%
geometry = Geometry(3)
V_raw, F, uac, _, _ = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(geometry)
V = scale_mesh_to_unit_cube(V_raw)
params = PriorParameters.load(Path(f"data/parameters/params_{geometry.value}.npy"))

# %%
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=0.2, ell=0.1, seed=1)
sample = prior.sample()
plot_numpy_sample(
    sample, V=V, F=F, file="figures/priors/atrium_baseline_sample.eps", zoom=1.0
)

# %%
