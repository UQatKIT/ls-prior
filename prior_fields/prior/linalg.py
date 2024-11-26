from dolfin import Matrix, Vector

from prior_fields.prior.converter import (
    matrix_to_petsc,
    petsc_to_matrix,
)


def multiply_matrices(A: Matrix, B: Matrix) -> Matrix:
    """
    Compute the product of two matrices.

    Parameters
    ----------
    A : dl.Matrix
        First matrix in matrix-matrix product
    B : dl.Matrix
        Second matrix in matrix-matrix product

    Returns
    -------
    dl.Matrix
        :math:`AB`
    """
    return petsc_to_matrix(matrix_to_petsc(A).matMult(matrix_to_petsc(B)))


def len_vector(v: Vector) -> int:
    return len(v.get_local())
