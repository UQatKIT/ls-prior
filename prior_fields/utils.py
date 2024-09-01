from typing import overload

import numpy as np
from dolfin import Matrix, Vector
from numpy.random import Generator
from potpourri3d import MeshVectorHeatSolver

from prior_fields.converter import (
    matrix_to_numpy,
    matrix_to_petsc,
    numpy_to_vector,
    petsc_to_matrix,
    vector_to_numpy,
)
from prior_fields.dtypes import Array1d, ArrayNx3


def get_sigma_from_kappa_and_tau(kappa: float, tau: float) -> float:
    """
    Compute marginal standard deviation of a stationary BiLaplacianPrior parameterized
    with :math:`\\kappa` and :math:`\\tau`.

    Parameters
    ----------
    kappa : float
    tau : float

    Returns
    -------
    float
        Marginal standard deviation.
    """
    return 1 / (2 * np.sqrt(np.pi) * kappa * tau)


@overload
def get_kappa_from_ell(ell: float) -> float: ...


@overload
def get_kappa_from_ell(ell: Array1d) -> Array1d: ...


def get_kappa_from_ell(ell: float | Array1d) -> float | Array1d:
    """Get scaling parameter :math:`\\kappa` from correlation length :math:`\\ell`."""
    return 1 / ell


@overload
def get_tau_from_sigma_and_ell(sigma: float, ell: float) -> float: ...


@overload
def get_tau_from_sigma_and_ell(sigma: Array1d, ell: Array1d) -> Array1d: ...


def get_tau_from_sigma_and_ell(
    sigma: float | Array1d, ell: float | Array1d
) -> float | Array1d:
    """
    Get :math:`\\tau` from marginal variance :math:`\\sigma` and correlation length
    :math:`\\ell`.

    Notes
    -----
    This transformation is valid for the bi-Laplacian case only.
    """
    return ell / (2 * np.sqrt(np.pi) * sigma)


def transform_sample_to_alpha(x: Array1d) -> Array1d:
    """Sigmoid-like transformations of values in (-infty, infty) to (-pi, pi)."""
    return np.pi * (2 / (1 + np.exp(-x)) - 1)


def multiply_matrices(A: Matrix, B: Matrix) -> Matrix:
    """Compute the product of two matrices.

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


def random_normal_vector(dim: int, prng: Generator) -> Vector:
    """Create a vector of standard normally distributed noise.

    Parameters
    ----------
    dim : int
        Length of the random vector
    prng : np.random.Generator
        Pseudo random number generator

    Returns
    -------
    dl.Vector
        Sample from standard normal distribution
    """
    return numpy_to_vector(prng.standard_normal(size=dim))


def len_vector(v: Vector) -> int:
    return len(vector_to_numpy(v))


def ncol(M: Matrix) -> int:
    return matrix_to_numpy(M).shape[0]


def nrow(M: Matrix) -> int:
    return matrix_to_numpy(M).shape[1]


def get_reference_coordinates(V: ArrayNx3, F: ArrayNx3) -> tuple[ArrayNx3, ArrayNx3]:
    """Get reference coordinate systems at each vertex.

    The basis vectors (1, 0) and (0, 1) in the tangent space associated with vertex 1 are
    transported to every other vertex in V using the vector heat method.

    Parameters
    ----------
    V : ArrayNx3
        Array of vertex coordinates.
    F : ArrayNx3
        Array of vertex indices that form the facets of the tessellation.

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


def angles_to_3d_vector(alphas: Array1d, x_axes: ArrayNx3, y_axes: ArrayNx3) -> ArrayNx3:
    """
    Compute 3d vectors of directions from given angles and reference coordinate systems.

    Parameters
    ----------
    alphas : Array1d
        Array of angles between vectors and x-axes
    x_axes : ArrayNx3
        Vectors representing the x-axes of the reference coordinates systems.
    y_axes : ArrayNx3
        Vectors representing the y-axes of the reference coordinates systems.

    Returns
    -------
    ArrayNx3
        Directions corresponding to the given alphas
    """
    return (np.cos(alphas) * x_axes.T).T + (np.sin(alphas) * y_axes.T).T


def vectors_3d_to_angles(
    directions: ArrayNx3, x_axes: ArrayNx3, y_axes: ArrayNx3
) -> Array1d:
    """
    Compute angles in reference coordinate systems for given 3d vectors of directions.

    Parameters
    ----------
    directions : ArrayNx3
        3d vectors of directions in the coordinate systems.
    x_axes : ArrayNx3
        Vectors representing the x-axes of the reference coordinates systems.
    y_axes : ArrayNx3
        Vectors representing the y-axes of the reference coordinates systems.

    Returns
    -------
    Array1D
         Angles between :math:`-\\pi` and :math:`\\pi`
    """
    alphas_x = _angles_between_vectors(x_axes, directions)
    alphas_y = _angles_between_vectors(y_axes, directions)
    alphas = np.array(
        [ax if ay <= np.pi / 2 else -ax for ax, ay in zip(alphas_x, alphas_y)]
    )
    return alphas


def _angles_between_vectors(a, b):
    return np.arccos(
        np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    )
