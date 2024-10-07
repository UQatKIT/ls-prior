import re
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3, ArrayNx4
from prior_fields.tensor.mapper import (
    map_categories_from_faces_to_vertices,
    map_vectors_from_faces_to_vertices,
)


def read_meshes_from_lge_mri_data() -> tuple[
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

    # fibers
    fibers = map_vectors_from_faces_to_vertices(vecs=mesh.cell_data["fibers"][0], F=F)

    # tag for anatomical structure assignment
    tag = map_categories_from_faces_to_vertices(
        categories=extract_element_tags_from_file(str(path)[-1]), F=F
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


def read_endocardial_mesh_from_bilayer_model(
    i: int,
) -> tuple[ArrayNx3, ArrayNx3, ArrayNx3]:
    """Read endocardial vertices and faces from bilayer shape model file.

    Parameters
    ----------
    i : int
        Index of shape model.

    Returns
    -------
    (ArrayNx3, ArrayNx3, ArrayNx3)
        Vertices and faces of triangle mesh of atrial endocardium,
        and the solution to Laplace's equation.
    """
    mesh_volume = meshio.read(
        f"data/shape_model/cn617_g{i:03}/cn617_g{i:03}_LA_laplace.vtk"
    )
    V_volume = mesh_volume.points
    F_volume = mesh_volume.get_cells_type(mesh_volume.cells[0].type)

    # coordinates on endo-/epocardial axis
    phie_phi = mesh_volume.point_data["phie_phi"]

    # solution to Laplace's equation
    psi_ab = mesh_volume.point_data["phie_ab"]
    psi_r = mesh_volume.point_data["phie_r"]
    psi_v = mesh_volume.point_data["phie_v"]
    psi = np.column_stack([psi_ab, psi_r, psi_v])

    return _extract_endocardium_from_volume_mesh(V_volume, F_volume, phie_phi, psi)


def _extract_endocardium_from_volume_mesh(
    V: ArrayNx3, F: ArrayNx4, phie_phi: Array1d, psi: ArrayNx3
) -> tuple[ArrayNx3, ArrayNx3, ArrayNx3]:
    """Extract endocardial vertices and faces from tetrahedal mesh of atrium.

    Note
    ----
    This method is tailored for shape models generated with AugmentA as uploaded in
    https://zenodo.org/records/5571925.

    Parameters
    ----------
    V : ArrayNx3
        Array of vertex coordinates.
    F : ArrayNx4
        Array of vertex indices that form the faces of the tessellation.
    phie_phi : Array1d
        Coordinates of vertices on the endo-/epicardial axis (endocardium = 0)
    psi : ArrayNx3
        Solution to Laplace's equation.

    Returns
    -------
    (ArrayNx3, ArrayNx3, ArrayNx3)
        Vertices and faces of triangle mesh of atrial endocardium,
        and the solution to Laplace's equation.
    """
    # Identify endocardial points
    idx_endo_points = phie_phi == 0  # boolean array
    idx_endo = np.flatnonzero(idx_endo_points)  # int array for index mapping
    V_endo = V[idx_endo_points]
    psi_endo = psi[idx_endo_points]

    # Extract tetrahedral cells with all vertices in endocardium
    idx_endo_cells = np.isin(F, idx_endo).sum(axis=1) == 3  # boolean array
    F_endo_tetra = F[idx_endo_cells]

    # Extract endocardial triangles from these tetrahedral cells
    F_endo_triangle = F_endo_tetra[np.isin(F_endo_tetra, idx_endo)].reshape(-1, 3)

    # Map volume mesh indices of V to surface mesh indices of V_endo
    F_endo = np.searchsorted(idx_endo, F_endo_triangle)

    # Identify and remove unreferenced vertices
    referenced_indices = np.unique(F_endo)
    V_endo = V_endo[referenced_indices]
    psi_endo = psi_endo[referenced_indices]

    # Reindex F_endo to ensure correct references after removing vertices
    reindex_map = -1 * np.ones(idx_endo.shape[0], dtype=int)
    reindex_map[referenced_indices] = np.arange(referenced_indices.shape[0])
    F_endo = reindex_map[F_endo]
    F_endo = np.unique(np.sort(F_endo, axis=1), axis=0)

    return V_endo, F_endo, psi_endo


###############################
# Read fiber orientation data #
###############################
def get_fiber_orientation_at_vertices(i: int, V_endo: ArrayNx3) -> ArrayNx3:
    """
    Calculate fiber orientation at vertices from fiber orientation in close faces.

    Parameters
    ----------
    i : int
        Index of shape model.
    V_endo : ArrayNx3
        Endocaridial vertex coordinates for which to infer fiber orientation.

    Returns
    -------
    ArrayNx3
        3d vectors of fiber orientations at endocardial vertices.
    """
    mesh = meshio.read(f"data/shape_model/cn617_g{i:03}/cn617_g{i:03}_heart_fine.vtk")
    V_atria = mesh.points
    F_atria = mesh.get_cells_type(mesh.cells[0].type)
    fibers_atria = mesh.cell_data["fiber"][0]

    return _transfer_fiber_data_to_endocardial_mesh(
        V_endo, V_atria, F_atria, fibers_atria
    )


def _transfer_fiber_data_to_endocardial_mesh(
    V_endo: ArrayNx3, V_atria: ArrayNx3, F_atria: ArrayNx4, fibers_atria: ArrayNx3
) -> ArrayNx3:
    fibers = np.zeros_like(V_endo)

    # For every vertex in V_endo, find the index of the closest vertex in V_atria
    tree = cKDTree(V_atria)
    closest_vertices = tree.query(V_endo)[1]

    for idx, v in enumerate(closest_vertices):
        # Find all faces that contain the closest vertex
        relevant_faces = np.where(F_atria == v)[0]

        # Compute the average fiber direction within the relevant faces
        f = fibers_atria[relevant_faces].mean(axis=0)
        fibers[idx] = f / np.linalg.norm(f)

    return fibers
