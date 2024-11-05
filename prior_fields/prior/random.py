from dolfin import FunctionSpace, Vector
from numpy.random import Generator

from prior_fields.prior.converter import numpy_to_vector


def random_normal_vector(dim: int, prng: Generator, Vh: FunctionSpace) -> Vector:
    """
    Create a vector of standard normally distributed noise.

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
    return numpy_to_vector(prng.standard_normal(size=dim), Vh)
