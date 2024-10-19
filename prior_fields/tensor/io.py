import re
from pathlib import Path

import meshio
import numpy as np

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3
from prior_fields.tensor.mapper import (
    get_coefficients,
    get_dict_with_adjacent_faces_for_each_vertex,
    map_categories_from_faces_to_vertices,
    map_fibers_to_tangent_space,
    map_vectors_from_faces_to_vertices,
)
from prior_fields.tensor.tangent_space_coordinates import get_uac_basis_vectors
from prior_fields.tensor.transformer import vector_coefficients_2d_to_angles


def get_mesh_and_point_data_from_lge_mri_based_data(
    path: Path,
) -> tuple[ArrayNx3, ArrayNx3, ArrayNx2, ArrayNx3, Array1d]:
    """Extract vertices, faces, uac and fibers from mesh file and map fibers to vertices.

    Parameters
    ----------
    path : Path
        Path to the files corresponding to a single geometry downloaded from
        https://zenodo.org/records/3764917.

    Returns
    -------
    (ArrayNx3, ArrayNx3, ArrayNx2, ArrayNx3)
        Vertices, faces, and UAC and fiber orientation at the vertices.
    """
    mesh = meshio.read(path / f"LA_Endo_{str(path)[-1]}.vtk")

    # vertices and faces
    V = mesh.points
    F = mesh.get_cells_type(mesh.cells[0].type)

    # universal atrial coordinates
    alpha = mesh.point_data["alpha"]
    beta = mesh.point_data["beta"]
    uac = np.column_stack([alpha, beta])

    # Construct mapping of vertex indices to vertex indices of its adjacent faces
    adjacent_faces = get_dict_with_adjacent_faces_for_each_vertex(F)

    # fibers
    fibers = map_vectors_from_faces_to_vertices(
        vecs=mesh.cell_data["fibers"][0], adjacent_faces=adjacent_faces
    )

    # tag for anatomical structure assignment
    tag = map_categories_from_faces_to_vertices(
        categories=extract_element_tags_from_file(str(path)[-1]),
        adjacent_faces=adjacent_faces,
    )

    return V, F, uac, fibers, tag


def extract_element_tags_from_file(geometry: str | int) -> Array1d:
    """Read anatomical structure tags at faces of geometry from .elem file.

    Parameters
    ----------
    geometry : str | int
        Index of geometry.

    Returns
    -------
    Array1d
        Array of tags at faces.
    """

    pattern = re.compile(rf"Labelled_{geometry}_.*\.elem$")
    elem_file = next(
        f
        for f in Path(f"data/LGE-MRI-based/{geometry}/").iterdir()
        if pattern.match(f.name)
    )

    with open(elem_file) as f:
        # First line contains the number of faces
        num_faces = int(f.readline().strip())

        element_tags = np.zeros(num_faces, dtype=int)
        for i, line in enumerate(f):
            # Last component in each line is the element tag
            element_tags[i] = int(line.split()[-1])

    return element_tags


def collect_data_from_lge_mri_meshes() -> tuple[ArrayNx2, ArrayNx3, Array1d]:
    # Read data (takes about 70 seconds)
    V_dict, F_dict, uac_dict, fibers_dict, tags_dict = _read_meshes_from_lge_mri_data()
    keys = sorted(V_dict.keys())

    directions_constant_beta_dict: dict[int, ArrayNx3] = dict()
    directions_constant_alpha_dict: dict[int, ArrayNx3] = dict()

    # Get UAC-based coordinates (takes about 80 seconds)
    for i in keys:
        print(f"Geometry {i}: Get UAC-based tangent space coordinates.")
        directions_constant_beta_dict[i], directions_constant_alpha_dict[i] = (
            get_uac_basis_vectors(V_dict[i], F_dict[i], uac_dict[i])
        )
        print()

    # Unite different geometries
    uac = np.vstack([uac_dict[i] for i in keys])
    fibers = np.vstack([fibers_dict[i] for i in keys])
    tags = np.hstack([tags_dict[i] for i in keys])
    directions_constant_beta = np.vstack(
        [directions_constant_beta_dict[i] for i in keys]
    )
    directions_constant_alpha = np.vstack(
        [directions_constant_alpha_dict[i] for i in keys]
    )

    # Map fibers to tangent space
    fibers_in_tangent_space = map_fibers_to_tangent_space(
        fibers, directions_constant_beta, directions_constant_alpha
    )

    # Get coefficients of fibers in tangent space coordinates
    fiber_coeffs_x, fiber_coeffs_y = get_coefficients(
        fibers_in_tangent_space, directions_constant_beta, directions_constant_alpha
    )

    # Get fiber angle within (-pi/2, pi/2] in UAC system
    fiber_angles = vector_coefficients_2d_to_angles(fiber_coeffs_x, fiber_coeffs_y)

    return uac, fiber_angles, tags


def _read_meshes_from_lge_mri_data() -> tuple[
    dict[int, ArrayNx3],
    dict[int, ArrayNx3],
    dict[int, ArrayNx2],
    dict[int, ArrayNx3],
    dict[int, Array1d],
]:
    """Read meshes extracted from real data (https://zenodo.org/records/3764917).

    Returns
    -------
    (
        dict[int, ArrayNx3],
        dict[int, ArrayNx3],
        dict[int, ArrayNx2],
        dict[int, ArrayNx3],
        dict[int, ArrayNx3],
    )
        Vertices and faces, and UAC, fiber orientation and tags for anatomical structure
        at the vertices for the 7 geometries and the atlas (geometry 6 with mean fiber
        orientation).
    """
    V: dict[int, ArrayNx3] = dict()
    F: dict[int, ArrayNx3] = dict()
    uac: dict[int, ArrayNx2] = dict()
    fibers: dict[int, ArrayNx3] = dict()
    tags: dict[int, Array1d] = dict()

    for p in Path("data/LGE-MRI-based").iterdir():
        idx = str(p)[-1]
        i = int(idx) if idx.isnumeric() else None

        if i is not None:
            V[i], F[i], uac[i], fibers[i], tags[i] = (
                get_mesh_and_point_data_from_lge_mri_based_data(p)
            )

    return V, F, uac, fibers, tags
