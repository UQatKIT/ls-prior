import numpy as np
from potpourri3d import MeshVectorHeatSolver

from prior_fields.prior.dtypes import ArrayNx3


def get_reference_coordinates(V: ArrayNx3, F: ArrayNx3) -> tuple[ArrayNx3, ArrayNx3]:
    """Get reference coordinate systems at each vertex.

    The basis vectors (1, 0) and (0, 1) in the tangent space associated with vertex 1 are
    transported to every other vertex in V using the vector heat method.

    Parameters
    ----------
    V : ArrayNx3
        Array of vertex coordinates.
    F : ArrayNx3
        Array of vertex indices that form the faces of the tessellation.

    Returns
    -------
    (ArrayNx3, ArrayNx3)
        X- and y-axes of the reference coordinate systems embedded in 3d space.
    """
    solver = MeshVectorHeatSolver(V, F)
    basis_x, basis_y, _ = solver.get_tangent_frames()

    # Parallel transport x-axis vector along the surface
    x_axes_2d = solver.transport_tangent_vector(v_ind=1, vector=[1, 0])
    x_axes = (
        x_axes_2d[:, 0, np.newaxis] * basis_x + x_axes_2d[:, 1, np.newaxis] * basis_y
    )

    # Parallel transport y-axis vector along the surface
    y_axes_2d = solver.transport_tangent_vector(v_ind=1, vector=[0, 1])
    y_axes = (
        y_axes_2d[:, 0, np.newaxis] * basis_x + y_axes_2d[:, 1, np.newaxis] * basis_y
    )

    return x_axes, y_axes
