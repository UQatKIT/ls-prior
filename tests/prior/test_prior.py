import numpy as np
import pytest
from dolfin import BoundaryMesh, Expression, FunctionSpace, Mesh, UnitSquareMesh
from potpourri3d import read_mesh

from prior_fields.prior.converter import (
    expression_to_vector,
    function_to_numpy,
    numpy_to_vector,
    str_to_vector,
    vector_to_numpy,
)
from prior_fields.prior.linalg import len_vector
from prior_fields.prior.prior import BiLaplacianPrior, BiLaplacianPriorNumpyWrapper


@pytest.fixture
def square_mesh():
    return UnitSquareMesh(32, 32)


@pytest.fixture
def sphere_mesh():
    mesh = Mesh("data/sphere.xml")  # tetrahedrons
    return BoundaryMesh(mesh, "exterior")


####################
# BiLaplacianPrior #
####################


@pytest.mark.parametrize(
    "mesh", ["square_mesh", "sphere_mesh"], ids=["square", "sphere"]
)
def test_bilaplacianpriors_with_same_seed_produce_same_samples(request, mesh):
    mesh = request.getfixturevalue(mesh)
    for _ in range(5):
        seed = np.random.randint(1, 100)
        prior1 = BiLaplacianPrior(mesh, 0.1, 0.1, seed=seed)
        prior2 = BiLaplacianPrior(mesh, 0.1, 0.1, seed=seed)
        for _ in range(20):
            sample1 = function_to_numpy(prior1.sample())
            sample2 = function_to_numpy(prior2.sample())
            np.testing.assert_array_equal(sample1, sample2)


@pytest.mark.parametrize(
    "mesh", ["square_mesh", "sphere_mesh"], ids=["square", "sphere"]
)
def test_bilaplacianprior_with_seed_produces_different_samples(request, mesh):
    mesh = request.getfixturevalue(mesh)
    for _ in range(5):
        seed = np.random.randint(1, 100)
        prior = BiLaplacianPrior(mesh, 0.1, 0.1, seed=seed)
        for _ in range(20):
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_almost_equal,
                function_to_numpy(prior.sample()),
                function_to_numpy(prior.sample()),
            )


@pytest.mark.parametrize(
    ["mesh", "mean"],
    (
        ["square_mesh", "0"],
        ["square_mesh", "0.2*x[0]+0.4*x[1]"],
        ["sphere_mesh", "0"],
        ["sphere_mesh", "0.2*x[0]+0.4*x[1]+0.1*x[2]"],
    ),
    ids=[
        "square with zero mean",
        "square with non-zero mean",
        "sphere with zero mean",
        "sphere with non-zero mean",
    ],
)
def test_bilaplacianprior_cost_is_zero_at_true_mean(request, mesh, mean):
    mesh = request.getfixturevalue(mesh)
    mean_expr = Expression(mean, degree=1)
    mean_vector = expression_to_vector(mean_expr, FunctionSpace(mesh, "CG", 1))
    prior = BiLaplacianPrior(mesh, 5.0, 1.0, mean=mean_vector)

    assert prior.cost(mean_vector) == 0


@pytest.mark.parametrize(
    ["mesh", "mean"],
    (
        ["square_mesh", "0"],
        ["square_mesh", "0.2*x[0]+0.4*x[1]"],
        ["sphere_mesh", "0"],
        ["sphere_mesh", "0.2*x[0]+0.4*x[1]+0.1*x[2]"],
    ),
    ids=[
        "square with zero mean",
        "square with non-zero mean",
        "sphere with zero mean",
        "sphere with non-zero mean",
    ],
)
def test_bilaplacianprior_cost_decreases_towards_mean(request, mesh, mean):
    mesh = request.getfixturevalue(mesh)

    mean_expr = Expression(mean, degree=1)
    mean_vector = expression_to_vector(mean_expr, FunctionSpace(mesh, "CG", 1))
    mean_array = vector_to_numpy(mean_vector)

    mean_approx = numpy_to_vector(mean_array + np.ones_like(mean_array) / 1000)
    mean_approx_worse = numpy_to_vector(mean_array + np.ones_like(mean_array) / 900)

    prior = BiLaplacianPrior(mesh, 5.0, 1.0, mean=mean_vector)

    assert prior.cost(mean_approx) <= 1e-3
    assert prior.cost(mean_approx) < prior.cost(mean_approx_worse)


@pytest.mark.parametrize(
    ["mesh", "mean"],
    (
        ["square_mesh", "0"],
        ["square_mesh", "0.2*x[0]+0.4*x[1]"],
        ["sphere_mesh", "0"],
        ["sphere_mesh", "0.2*x[0]+0.4*x[1]+0.1*x[2]"],
    ),
    ids=[
        "square with zero mean",
        "square with non-zero mean",
        "sphere with zero mean",
        "sphere with non-zero mean",
    ],
)
def test_bilaplacianprior_grad_is_zero_at_true_mean(request, mesh, mean):
    mesh = request.getfixturevalue(mesh)
    mean_expr = Expression(mean, degree=1)
    mean_vector = expression_to_vector(mean_expr, FunctionSpace(mesh, "CG", 1))
    prior = BiLaplacianPrior(mesh, 5.0, 1.0, mean=mean_vector)
    grad = prior.grad(mean_vector)

    assert len_vector(mean_vector) == len_vector(grad)
    assert np.linalg.norm(grad) == 0


@pytest.mark.parametrize(
    ["mesh", "mean"],
    (
        ["square_mesh", "0"],
        ["square_mesh", "0.2*x[0]+0.4*x[1]"],
        ["sphere_mesh", "0"],
        ["sphere_mesh", "0.2*x[0]+0.4*x[1]+0.1*x[2]"],
    ),
    ids=[
        "square with zero mean",
        "square with non-zero mean",
        "sphere with zero mean",
        "sphere with non-zero mean",
    ],
)
def test_bilaplacianprior_negative_grad_points_towards_true_mean(request, mesh, mean):
    mesh = request.getfixturevalue(mesh)
    Vh = FunctionSpace(mesh, "CG", 1)

    mean_expr = Expression(mean, degree=1)
    mean_vector = expression_to_vector(mean_expr, FunctionSpace(mesh, "CG", 1))

    mean_approx = numpy_to_vector(
        np.random.normal(0, 1, vector_to_numpy(mean_vector).shape), Vh
    )

    prior = BiLaplacianPrior(mesh, 5.0, 1.0, mean=mean_vector)
    grad = prior.grad(mean_approx)

    mean_approx_minus_grad = numpy_to_vector(
        vector_to_numpy(mean_approx) - 1e-5 * vector_to_numpy(grad)
    )

    assert prior.cost(mean_approx_minus_grad) < prior.cost(mean_approx)


@pytest.mark.parametrize(
    "mesh", ["square_mesh", "sphere_mesh"], ids=["square", "sphere"]
)
@pytest.mark.parametrize(
    ["sigma", "ell"],
    [[1, 1], [1.0, 1], [1, 1.0]],
    ids=["both", "sigma", "ell"],
)
def test_bilaplacianprior_raises_for_integer_params(request, mesh, sigma, ell):
    mesh = request.getfixturevalue(mesh)
    with pytest.raises(
        TypeError,
        match=r"Got (?:sigma|ell) of type <class 'int'>, expected float or dl.Vector.",
    ):
        BiLaplacianPrior(mesh, sigma, ell)


################################
# BiLaplacianPriorNumpyWrapper #
################################


def test_bilaplacianpriors_from_wrapper_with_same_seed_produce_same_samples():
    V, F = read_mesh("data/left_atrium.ply")
    for _ in range(5):
        seed = np.random.randint(1, 100)
        prior1 = BiLaplacianPriorNumpyWrapper(V, F, 0.1, 0.1, seed=seed)
        prior2 = BiLaplacianPriorNumpyWrapper(V, F, 0.1, 0.1, seed=seed)
        for _ in range(20):
            sample1 = prior1.sample()
            sample2 = prior2.sample()
            np.testing.assert_array_equal(sample1, sample2)


def test_bilaplacianprior_from_wrapper_with_seed_produces_different_samples():
    V, F = read_mesh("data/left_atrium.ply")
    for _ in range(5):
        seed = np.random.randint(1, 100)
        prior = BiLaplacianPriorNumpyWrapper(V, F, 0.1, 0.1, seed=seed)
        for _ in range(20):
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_almost_equal,
                prior.sample(),
                prior.sample(),
            )


@pytest.mark.parametrize(
    ["mesh", "mean"],
    (
        ["square_mesh", "0"],
        ["square_mesh", "0.2*x[0]+0.4*x[1]"],
        ["sphere_mesh", "0"],
        ["sphere_mesh", "0.2*x[0]+0.4*x[1]+0.1*x[2]"],
    ),
    ids=[
        "square with zero mean",
        "square with non-zero mean",
        "sphere with zero mean",
        "sphere with non-zero mean",
    ],
)
def test_bilaplacianprior_from_wrapper_cost_is_zero_at_true_mean(request, mesh, mean):
    mesh = request.getfixturevalue(mesh)
    Vh = FunctionSpace(mesh, "CG", 1)
    V = mesh.coordinates()
    F = mesh.cells()

    # Convert mean expression to array ordered as the vertices in `V`
    mean_array = vector_to_numpy(
        str_to_vector(mean, mesh), Vh, use_vertex_to_dof_map=True
    )
    prior = BiLaplacianPriorNumpyWrapper(V, F, 5.0, 1.0, mean=mean_array)

    assert prior.cost(mean_array) == 0


@pytest.mark.parametrize(
    ["mesh", "mean"],
    (
        ["square_mesh", "0"],
        ["square_mesh", "0.2*x[0]+0.4*x[1]"],
        ["sphere_mesh", "0"],
        ["sphere_mesh", "0.2*x[0]+0.4*x[1]+0.1*x[2]"],
    ),
    ids=[
        "square with zero mean",
        "square with non-zero mean",
        "sphere with zero mean",
        "sphere with non-zero mean",
    ],
)
def test_bilaplacianprior_from_wrapper_cost_decreases_towards_mean(request, mesh, mean):
    mesh = request.getfixturevalue(mesh)
    Vh = FunctionSpace(mesh, "CG", 1)
    V = mesh.coordinates()
    F = mesh.cells()

    # Convert mean expression to array ordered as the vertices in `V`
    mean_array = vector_to_numpy(
        str_to_vector(mean, mesh), Vh, use_vertex_to_dof_map=True
    )

    mean_approx = mean_array + np.ones_like(mean_array) / 1000
    mean_approx_worse = mean_array + np.ones_like(mean_array) / 900

    prior = BiLaplacianPriorNumpyWrapper(V, F, 10.0, 0.1, mean=mean_array)

    assert prior.cost(mean_approx) <= 1e-3
    assert prior.cost(mean_approx) < prior.cost(mean_approx_worse)


@pytest.mark.parametrize(
    ["mesh", "mean"],
    (
        ["square_mesh", "0"],
        ["square_mesh", "0.2*x[0]+0.4*x[1]"],
        ["sphere_mesh", "0"],
        ["sphere_mesh", "0.2*x[0]+0.4*x[1]+0.1*x[2]"],
    ),
    ids=[
        "square with zero mean",
        "square with non-zero mean",
        "sphere with zero mean",
        "sphere with non-zero mean",
    ],
)
def test_bilaplacianprior_from_wrapper_grad_is_zero_at_true_mean(request, mesh, mean):
    mesh = request.getfixturevalue(mesh)
    Vh = FunctionSpace(mesh, "CG", 1)
    V = mesh.coordinates()
    F = mesh.cells()

    # Convert mean expression to array ordered as the vertices in `V`
    mean_array = vector_to_numpy(
        str_to_vector(mean, mesh), Vh, use_vertex_to_dof_map=True
    )
    prior = BiLaplacianPriorNumpyWrapper(V, F, 5.0, 1.0, mean=mean_array)
    grad = prior.grad(mean_array)

    assert mean_array.shape == grad.shape
    assert np.linalg.norm(grad) == 0


@pytest.mark.parametrize(
    ["mesh", "mean"],
    (
        ["square_mesh", "0"],
        ["square_mesh", "0.2*x[0]+0.4*x[1]"],
        ["sphere_mesh", "0"],
        ["sphere_mesh", "0.2*x[0]+0.4*x[1]+0.1*x[2]"],
    ),
    ids=[
        "square with zero mean",
        "square with non-zero mean",
        "sphere with zero mean",
        "sphere with non-zero mean",
    ],
)
def test_bilaplacianprior_from_wrapper_negative_grad_points_towards_true_mean(
    request, mesh, mean
):
    mesh = request.getfixturevalue(mesh)
    Vh = FunctionSpace(mesh, "CG", 1)
    V = mesh.coordinates()
    F = mesh.cells()

    # Convert mean expression to array ordered as the vertices in `V`
    mean_array = vector_to_numpy(
        str_to_vector(mean, mesh), Vh, use_vertex_to_dof_map=True
    )
    mean_approx = np.random.normal(0, 1, mean_array.shape)

    prior = BiLaplacianPriorNumpyWrapper(V, F, 5.0, 1.0, mean=mean_array)
    grad = prior.grad(mean_approx)

    mean_approx_minus_grad = mean_approx - 1e-5 * grad

    assert prior.cost(mean_approx_minus_grad) < prior.cost(mean_approx)


@pytest.mark.parametrize(
    "mesh", ["square_mesh", "sphere_mesh"], ids=["square", "sphere"]
)
@pytest.mark.parametrize(
    ["sigma", "ell"],
    [[1, 1], [1.0, 1], [1, 1.0]],
    ids=["both", "sigma", "ell"],
)
def test_bilaplacianprior_from_wrapper_raises_for_integer_params(
    request, mesh, sigma, ell
):
    mesh = request.getfixturevalue(mesh)
    V = mesh.coordinates()
    F = mesh.cells()
    with pytest.raises(
        TypeError,
        match=r"Got (?:sigma|ell) of type <class 'int'>, expected float or Array1d.",
    ):
        BiLaplacianPriorNumpyWrapper(V, F, sigma, ell)
