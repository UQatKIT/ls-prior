# %%
from pathlib import Path

import matplotlib.pyplot as plt
from potpourri3d import read_mesh

from prior_fields.converter import (
    convert_mesh_files,
    create_triangle_mesh_from_coordinates,
)
from prior_fields.plots import (
    plot_function,
    plot_vector_field_on_surface,
    plot_vertex_values_on_surface,
)
from prior_fields.prior import BiLaplacianPrior, BiLaplacianPriorNumpyWrapper
from prior_fields.utils import (
    angles_to_3d_vector,
    get_reference_coordinates,
    transform_sample_to_alpha,
)

##################
# Read mesh data #
##################
# %%
mesh_file = Path("data/left_atrium.ply")
if not mesh_file.is_file():
    # https://github.com/fsahli/FiberNet/blob/main/data/LA_model.vtk
    convert_mesh_files("left_atrium", input_type=".vtk", output_type=".ply")

V, F = read_mesh(mesh_file.as_posix())

mesh = create_triangle_mesh_from_coordinates(V, F)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax = plt.gca()
ax.set_aspect("equal")

ax.plot_trisurf(  # type: ignore
    *[V[:, i] for i in range(3)], triangles=mesh.cells(), alpha=0.7
)
plt.show()


##################
# Set parameters #
##################
# %%
sigma = 0.2
ell = 10.0


#######################
# With dolfin backend #
#######################
# %%
mesh.order()
prior = BiLaplacianPrior(mesh, sigma=sigma, ell=ell)

# %%
sample = prior.sample()
plot_function(sample, title="BiLaplacianPrior sample", alpha=0.9)


######################
# With numpy backend #
######################
# %%
prior_numpy = BiLaplacianPriorNumpyWrapper(V, F, sigma=sigma, ell=ell)

# %%
sample = prior_numpy.sample()
plot_vertex_values_on_surface(sample, V)


#########################################
# Prior field as angles of vector field #
#########################################
# %%
x_axes, y_axes = get_reference_coordinates(V, F)

alphas = transform_sample_to_alpha(sample)
vector_field = angles_to_3d_vector(alphas=alphas, x_axes=x_axes, y_axes=y_axes)

plot_vector_field_on_surface(vector_field, V)

# %%
