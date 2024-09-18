# %%
import matplotlib.pyplot as plt
import numpy as np
from dolfin import BoundaryMesh, Mesh

from prior_fields.prior.converter import function_to_numpy
from prior_fields.prior.plots import plot_function
from prior_fields.prior.prior import BiLaplacianPrior
from prior_fields.tensor.plots import add_3d_vectors_to_plot
from prior_fields.tensor.transformer import (
    angles_to_3d_vector,
    sample_to_alpha,
)
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

x_axes, y_axes = get_reference_coordinates(V, F)
vector_field = angles_to_3d_vector(alphas=alphas, x_axes=x_axes, y_axes=y_axes)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d", computed_zorder=False)

ax.plot_trisurf(  # type: ignore
    *[V[:, i] for i in range(3)],  # type: ignore
    triangles=sphere_mesh_ordered.cells(),
    cmap="Blues",
    zorder=1,
)

idx = []
for j in range(V.shape[0]):
    # Check whether vertex j is on the viewer's side of the sphere
    dot_product = np.dot(V[j], np.array([1, -1, 1]))
    if dot_product > 0:
        idx.append(j)


add_3d_vectors_to_plot(V[idx], x_axes[idx], ax, "tab:green")
add_3d_vectors_to_plot(V[idx], y_axes[idx], ax, "tab:green")

add_3d_vectors_to_plot(V[idx], vector_field[idx], ax, "tab:red")

plt.show()

# %%
