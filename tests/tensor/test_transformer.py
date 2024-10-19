import numpy as np
import pytest

from prior_fields.tensor.transformer import (
    angles_to_3d_vector,
    angles_to_sample,
    normalize,
    sample_to_angles,
    vectors_3d_to_angles,
)


@pytest.fixture
def x_axes():
    # initialize random x_axes
    x_axes = np.random.uniform(-1, 1, (100, 3))

    # normalize x_axes
    x_axes = normalize(x_axes)

    return x_axes


@pytest.fixture
def y_axes(x_axes):
    # initialize random y_axes
    y_axes = np.random.uniform(0, 1, (100, 3))

    # make y_axes orthonormal to x_axes
    y_axes -= (np.sum(x_axes * y_axes, axis=1) * x_axes.T).T

    # normalize y_axes
    y_axes = normalize(y_axes)

    return y_axes


def test_angles_to_3d_vector_and_back(x_axes, y_axes):
    angles_inp = np.random.uniform(-np.pi, np.pi, 100)

    angles_out = vectors_3d_to_angles(
        angles_to_3d_vector(angles_inp, x_axes, y_axes), x_axes, y_axes
    )
    assert np.allclose(angles_inp, angles_out)


def test_vectors_3d_to_angles_outputs_between_minus_pi_and_pi(x_axes, y_axes):
    angles_inp = np.random.uniform(-10, 10, 100)

    angles_out = vectors_3d_to_angles(
        angles_to_3d_vector(angles_inp, x_axes, y_axes), x_axes, y_axes
    )

    assert all(-np.pi <= angles_out)
    assert all(angles_out <= np.pi)


def test_sigmoid_transformation_forward_reverse():
    x = np.random.standard_normal(1000)
    assert np.allclose(x, angles_to_sample(sample_to_angles(x)))


def test_sigmoid_transformation_reverse_forward():
    angles = np.random.uniform(-np.pi, np.pi, 1000)
    assert np.allclose(angles, sample_to_angles(angles_to_sample(angles)))
