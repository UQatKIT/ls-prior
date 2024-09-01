import numpy as np
import pytest

from prior_fields.utils import angles_to_3d_vector, vectors_3d_to_angles


@pytest.fixture
def x_axes():
    # initialize random x_axes
    x_axes = np.random.uniform(-1, 1, (100, 3))

    # normalize x_axes
    x_axes = (x_axes.T / np.linalg.norm(x_axes, axis=1)).T

    return x_axes


@pytest.fixture
def y_axes(x_axes):
    # initialize random y_axes
    y_axes = np.random.uniform(0, 1, (100, 3))

    # make y_axes orthonormal to x_axes
    y_axes -= (np.sum(x_axes * y_axes, axis=1) * x_axes.T).T

    # normalize y_axes
    y_axes = (y_axes.T / np.linalg.norm(y_axes, axis=1)).T

    return y_axes


def test_angles_to_3d_vector_and_back(x_axes, y_axes):
    alphas_inp = np.random.uniform(-np.pi, np.pi, 100)

    alphas_out = vectors_3d_to_angles(
        angles_to_3d_vector(alphas_inp, x_axes, y_axes), x_axes, y_axes
    )
    assert np.allclose(alphas_inp, alphas_out)


def test_vectors_3d_to_angles_outputs_between_minus_pi_and_pi(x_axes, y_axes):
    alphas_inp = np.random.uniform(-10, 10, 100)

    alphas_out = vectors_3d_to_angles(
        angles_to_3d_vector(alphas_inp, x_axes, y_axes), x_axes, y_axes
    )

    assert all(-np.pi <= alphas_out)
    assert all(alphas_out <= np.pi)
