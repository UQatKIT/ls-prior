"""
This file can be used to preprocess the bi-atrial shape models from
https://zenodo.org/records/5571925. To do so, download a .zip file f'cn617_g{i:03}.zip',
extract the files and move the .vtk files into f'data/cn617_g{i:03}/'.

The preprocessing steps are:
1. Read endocardial vertices and faces from the mesh file f'cn617_g{i:03}_LA_laplace.vtk'

2. Reorder the faces to work with the vector heat method from potpourri3d:
    An edge is not allowed to appear twice in the same order. For example, the two faces
    [1, 2, 3] and [1, 2, 4] violate this, while reordering the second face to [4, 2, 1]
    resolves the violation: The edge (1, 2) appears twice but in different directions.

3. Extract fiber information from f'cn617_g{i:03}_heart_fine.vtk':
    Fiber information is not contained in f'cn617_g{i:03}_LA_laplace.vtk'. Therefore, we
    extract it from the other file representing the same geometry. Unfortunately, the
    files use different discretisations of the same surface, such that we have no obvious
    mapping of fiber from '..._heart_fine.vtk' to the mesh from '..._LA_laplace.vtk'.
    We solve this, by searching the nearest neighbour of every vertex in LA_laplace.
    Since fiber information is available for the faces, not the vertices, we then use the
    mean fiber orientation of all faces adjacent to the nearest neighbour vertex.

4. Remove vertices that appear in more than one boundary loop:
    We are not interested in the boundary, so we don't care about loosing one boundary
    vertex.

5. Collect vertices, faces and fibers in a mesh and write them to a new .vtk file.
"""

# %%
import re

from meshio import Mesh, write

from prior_fields.tensor.data_cleaning import (
    remove_vertex_from_mesh,
    reorder_edges_of_faces,
)
from prior_fields.tensor.io import (
    get_fiber_orientation_at_vertices,
    read_endocardial_mesh_from_bilayer_model,
)
from prior_fields.tensor.vector_heat_method import get_reference_coordinates

##################
# Read mesh data #
##################
# %%
i = 1  # choose geometry (indices of files in https://zenodo.org/records/5571925)
V, F = read_endocardial_mesh_from_bilayer_model(i)

# %%
F = reorder_edges_of_faces(F)
fibers = get_fiber_orientation_at_vertices(i, V)

# %%
# Check that vector heat method is applicable
done = False
while not done:
    try:
        _, _ = get_reference_coordinates(V, F)
        done = True
    except RuntimeError as e:
        match = re.search(r"vertex (\d+) appears", repr(e))
        if match:
            idx = int(match.group(1))
            print(f"Removing vertex {idx}")
            V, F, fibers = remove_vertex_from_mesh(idx, V, F, fibers)
        else:
            print(e)
            break

# %%
out_mesh = Mesh(points=V, cells={"triangle": F}, point_data={"fiber": fibers})
write(f"data/LA_with_fibers_{i}.vtk", out_mesh)

# %%
