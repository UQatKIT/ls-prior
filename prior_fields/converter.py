from pathlib import Path
from typing import Literal

import meshio
import numpy as np
from dolfin import (
    Expression,
    Function,
    FunctionSpace,
    Matrix,
    Mesh,
    MeshEditor,
    PETScMatrix,
    Vector,
    as_backend_type,
    interpolate,
)
from petsc4py.PETSc import Mat  # type: ignore

from prior_fields.dtypes import Array1d, Array2d, ArrayNx2, ArrayNx3


####################################
# Convert numpy/dolfin/petsc types #
####################################
def vector_to_numpy(v: Vector) -> Array1d:
    return v.get_local()


def matrix_to_numpy(M: Matrix) -> Array2d:
    return M.array()


def expression_to_function(expr: Expression, Vh: FunctionSpace) -> Function:
    return interpolate(expr, Vh)


def function_to_vector(f: Function) -> Vector:
    return f.vector()


def expression_to_vector(expr: Expression, Vh: FunctionSpace) -> Vector:
    return function_to_vector(expression_to_function(expr, Vh))


def expression_to_numpy(expr: Expression, Vh: FunctionSpace) -> Array1d:
    return vector_to_numpy(expression_to_vector(expr, Vh))


def function_to_numpy(f: Function) -> Array1d:
    return vector_to_numpy(function_to_vector(f))


def numpy_to_function(a: Array1d, Vh: FunctionSpace) -> Function:
    f = Function(Vh)
    f.vector().set_local(a)
    return f


def numpy_to_vector(a: Array1d) -> Vector:
    v = Vector()
    v.init(len(a))
    v.set_local(a)
    return v


def str_to_vector(s: str, mesh: Mesh) -> Vector:
    expr = Expression(s, degree=1)
    Vh = FunctionSpace(mesh, "CG", 1)
    return expression_to_vector(expr, Vh)


def str_to_function(s: str, mesh: Mesh) -> Function:
    expr = Expression(s, degree=1)
    Vh = FunctionSpace(mesh, "CG", 1)
    return expression_to_function(expr, Vh)


def matrix_to_petsc(M: Matrix) -> Mat:
    return as_backend_type(M).mat()


def petsc_to_matrix(M: Mat) -> Matrix:
    return Matrix(PETScMatrix(M))


################################
# Convert mesh representations #
################################
def create_triangle_mesh_from_coordinates(V: ArrayNx2 | ArrayNx3, F: ArrayNx3) -> Mesh:
    mesh = Mesh()

    editor = MeshEditor()
    editor.open(mesh, "triangle", 2, V.shape[1])
    editor.init_vertices(V.shape[0])
    editor.init_cells(F.shape[0])

    for i, v in enumerate(V):
        editor.add_vertex(i, v)

    for i, c in enumerate(F):
        editor.add_cell(i, c)

    editor.close()

    return mesh


######################
# Convert mesh files #
######################
def convert_mesh_files(
    filename: str,
    input_type: Literal[".vtk", ".ply"],
    output_type: Literal[".vtk", ".ply", ".xdmf"],
    path: Path = Path("data/"),
) -> None:
    """Convert mesh file to different file type using meshio.

    Notes
    -----
    This can be used to obtain a mesh with an ordering as required in
    `potpourri3d.MeshVectorHeatSolver()`. Meshio takes care of an appropriate ordering
    when saving a mesh in the .ply format.

    Also `potpourri3d.read_mesh()` supports .ply files, but doesn't support .vtk and
    .xdmf files.

    Parameters
    ----------
    filename : str
        Name of mesh file (input and output)
    input_type : str
        Path extension defining the input file type.
    output_type : str
        Path extension defining the output file type.
    path : Path
        Location of input and output file. Defaults to `PosixPath('data')`.
    """
    mesh = meshio.read(path / (filename + input_type))

    for key in mesh.cell_data:
        mesh.cell_data[key][0] = mesh.cell_data[key][0].astype(np.float32)

    cell_type = mesh.cells[0].type
    cells = mesh.get_cells_type(cell_type)
    out_mesh = meshio.Mesh(
        points=mesh.points.astype(np.float32),
        cells={cell_type: cells},
        cell_data={key: data for key, data in mesh.cell_data.items()},
    )

    meshio.write(path / (filename + output_type), out_mesh)
