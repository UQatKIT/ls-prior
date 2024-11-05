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
    vertex_to_dof_map,
)
from petsc4py.PETSc import Mat  # type: ignore
from scipy.sparse import csr_array

from prior_fields.prior.dtypes import Array1d, Array2d, ArrayNx2, ArrayNx3


####################################
# Convert numpy/dolfin/petsc types #
####################################
def vector_to_numpy(
    v: Vector, Vh: FunctionSpace | None = None, get_vertex_values: bool = False
) -> Array1d:
    if get_vertex_values:
        if FunctionSpace is None:
            raise ValueError(
                "Need function space in order to map the values "
                "from the DOFs to the vertices."
            )
        return v.get_local()[vertex_to_dof_map(Vh)]
    else:
        return v.get_local()


def numpy_to_vector(
    a: Array1d, Vh: FunctionSpace | None = None, map_vertex_values_to_dof: bool = False
) -> Vector:
    if map_vertex_values_to_dof:
        if FunctionSpace is None:
            raise ValueError(
                "Need function space in order to map the values "
                "from the vertices to the DOFs."
            )
        a = a[np.argsort(vertex_to_dof_map(Vh))]
    v = Vector()
    v.init(len(a))
    v.set_local(a)
    return v


def matrix_to_numpy(M: Matrix) -> Array2d:
    return M.array()


def expression_to_function(expr: Expression, Vh: FunctionSpace) -> Function:
    return interpolate(expr, Vh)


def function_to_vector(f: Function) -> Vector:
    return f.vector()


def expression_to_vector(expr: Expression, Vh: FunctionSpace) -> Vector:
    return function_to_vector(expression_to_function(expr, Vh))


def function_to_numpy(f: Function, get_vertex_values: bool = False) -> Array1d:
    if get_vertex_values:
        mesh = f.ufl_function_space().mesh()
        return f.compute_vertex_values(mesh)
    else:
        return f.vector().get_local()


def numpy_to_function(
    a: Array1d, Vh: FunctionSpace, map_vertex_values_to_dof: bool = False
) -> Function:
    if map_vertex_values_to_dof:
        a = a[np.argsort(vertex_to_dof_map(Vh))]
    f = Function(Vh)
    f.vector().set_local(a)
    return f


def numpy_to_matrix_sparse(M: csr_array) -> Matrix:
    petMat = Mat()
    petMat.createAIJ(size=M.shape, csr=(M.indptr, M.indices, M.data))
    petMat.setUp()
    return petsc_to_matrix(petMat)


def str_to_vector(s: str, mesh: Mesh) -> Vector:
    """
    Convert string to dolfin vector ordered according to the DOFs of the function space.
    """
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


################
# Convert mesh #
################
def create_triangle_mesh_from_coordinates(V: ArrayNx2 | ArrayNx3, F: ArrayNx3) -> Mesh:
    """
    Create dolfin.Mesh with triangular faces from numpy vertices and faces.

    Parameters
    ----------
    V : ArrayNx2 | ArrayNx3
        Array with vertex coordinates.
    F : ArrayNx3
        (n, 3) array where each row contains the vertex indices of one face of the mesh.

    Returns
    -------
    dl.Mesh
    """
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


def scale_mesh_to_unit_cube(V: ArrayNx3) -> ArrayNx3:
    """
    Scale vertex coordinates such that the mesh lies within the unit cube.

    Parameters
    ----------
    V : ArrayNx3
        (n, 3) array where each row represents one vertex of a mesh.

    Returns
    -------
    ArrayNx3
        Vertex coordinates scale to [0,1]
    """
    return V / (V.max() - V.min())
