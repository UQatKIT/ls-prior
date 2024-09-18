import meshio

from prior_fields.prior.dtypes import ArrayNx3


def read_endocardial_mesh_with_fibers(i: int) -> tuple[ArrayNx3, ArrayNx3, ArrayNx3]:
    """Read vertices, faces and fibers of endocardial mesh.

    Parameters
    ----------
    i : int
        Index of mesh file (data/LA_with_fibers_{i}.vtk).

    Returns
    -------
    (ArrayNx3, ArrayNx3, ArrayNx3)
        Arrays with vertex coordinates, faces, and fiber directions at vertices.
    """
    mesh_data = meshio.read(f"data/LA_with_fibers_{i}.vtk")
    V = mesh_data.points
    F = mesh_data.get_cells_type("triangle")
    fibers = mesh_data.point_data["fiber"]

    return V, F, fibers
