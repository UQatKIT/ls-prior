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
from scipy.sparse import csr_array

from prior_fields.prior.dtypes import Array1d, Array2d, ArrayNx2, ArrayNx3


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


def numpy_to_matrix_sparse(M: csr_array) -> Matrix:
    petMat = Mat()
    petMat.createAIJ(size=M.shape, csr=(M.indptr, M.indices, M.data))
    petMat.setUp()
    return petsc_to_matrix(petMat)


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
