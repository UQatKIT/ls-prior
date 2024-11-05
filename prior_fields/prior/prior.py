import numpy as np
from dolfin import (
    Function,
    FunctionSpace,
    Matrix,
    Mesh,
    PETScKrylovSolver,
    PETScPreconditioner,
    TestFunction,
    TrialFunction,
    UserExpression,
    Vector,
    assemble,
    assemble_local,
    cells,
)
from dolfin.function.argument import Argument
from scipy.linalg import cholesky
from scipy.sparse import block_diag, coo_array
from ufl import Form, ds, dx, grad, inner

from prior_fields.prior.converter import (
    create_triangle_mesh_from_coordinates,
    numpy_to_function,
    numpy_to_matrix_sparse,
    numpy_to_vector,
    vector_to_numpy,
)
from prior_fields.prior.dtypes import Array1d, ArrayNx3
from prior_fields.prior.parameterization import (
    get_kappa_from_ell,
    get_tau_from_sigma_and_ell,
)
from prior_fields.prior.random import random_normal_vector


class BiLaplacianPrior:
    """
    Finite element representation of a 2D Gaussian prior field with Matern covariance
    with :math:`\\nu = \\alpha - d/2 = 2 - 2/2 = 1`, i.e.,
    :math:`\\sigma^2 (\\kappa ||s - t||) K_1(\\kappa ||s - t||)`,
    where :math:`\\sigma^2 = (4 \\pi \\kappa^2 \\tau^2)^{-1}` is the marginal variance
    and :math:`\\ell = 1/\\kappa` represents the correlation length.

    The covariance matrix of the coefficient vector
    in the finite element representation is :math:`C = A^{-1} M A^{-1}`
    where :math:`A = \\tau (\\kappa^2 M + L)` is the discretized version
    of the differential operator :math:`\\tau (\\kappa^2 - \\Delta)`.

    M, L, and A are symmetric,
    because we use the Galerkin approach (test functions = trial functions).

    In the anisotropic case, i.e. if theta is an anisotropic tensor, the SPDE is given by
    :math:`\\tau (\\kappa^2 - \\nabla \\cdot (\\Theta \\nabla)) u = \\mathcal{W}`.

    In the non-stationary case, i.e. if kappa and/or tau depend on x, the SPDE is given
    by :math:`\\tau(x) (\\kappa(x)^2 - \\Delta) u(x) = \\mathcal{W}(x)`.

    Example
    -------
    ```python
    mesh = dl.UnitSquareMesh(64, 64)
    prior = BiLaplacianPrior(mesh, sigma=1.0, ell=0.3, seed=1)
    sample = prior.sample()
    plot_function(sample)
    ```

    Attributes
    ----------
    Vh : dl.FunctionSpace
        Finite element space
    kappa : float | dl.Function
        Scaling parameter :math:`\\kappa(x) > 0`,
        where :math:`\\ell = 1/\\kappa` is the correlation length of the prior field.
        Correlations are near 0.1 at distance :math:`\\delta = \\ell \\sqrt{8\\nu}`,
        where :math:`\\nu > 0` is the smoothness parameter.
    tau : float | dl.Function
        Controls marginal variance
    beta : dl.Constant
        Coefficient in Robin boundary condition (empirically optimal)
    mean : dl.Vector
        Prior mean. Defaults to zero.
    theta : dl.UserExpression
        Anisotropic tensor
    M : dl.Matrix
        Mass matrix with entries :math:`M_{ij} = \\int \\psi_i(s) \\phi_j(s) ds`
    Msolver : dl.PETScKrylovSolver
        Solver to efficiently solve :math:`Mx = y` using `Msolver.solve(x, y)`
    A : dl.Matrix
        Finite element matrix :math:`\\tau (\\kappa^2 M + L)`
        arising from discretization of :math:`\\tau (\\kappa^2 - \\Delta)`.
        where :math:`L_{ij} = \\int \\nabla \\psi_i(s) \\cdot \\nabla \\phi_j(s) ds`.
    Asolver : dl.PETScKrylovSolver
        Solver to efficiently solve :math:`Ax = y` using `Asolver.solve(x, y)`
    sqrtM : dl.Matrix
        Sparse decomposition :math:`M = sqrtM sqrtM^\\top`.
    prng : np.random.default_rng
        Pseudo random random generator used for sampling.
    """

    def __init__(
        self,
        mesh: Mesh,
        sigma: float | Vector,
        ell: float | Vector,
        mean: Vector | None = None,
        theta: UserExpression | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Construct a bi-Laplacian prior.

        Parameters
        ----------
        mesh : dl.Mesh
            Mesh on which finite element space is defined.
        sigma : float | dl.Vector
            Marginal standard deviation.
            Constant in stationary case. Use dl.Vector to model non-stationarity.
        ell : float | dl.Vector
            Correlation length.
            Constant in stationary case. Use dl.Vector to model non-stationarity.
        mean : dl.Vector | None  (optional)
            Prior mean. Defaults to None (zero-mean). For a prior with mean function
            m(x), use `str_to_vector("m(x)", mesh)`, which is equivalent to
            `interpolate(Expression("m(x)", degree=1), Vh).vector()`.
        theta : dl.UserExpression | None  (optional)
            Semi-positive definite tensor to model anisotropy.
            Defaults to None, i.e. isotropy.
        seed : int | None
            Random seed for sampling.
        """
        self._validate_inputs(sigma, ell)

        self.Vh = FunctionSpace(mesh, "CG", 1)
        trial, test = self._init_trial_test_functions(self.Vh)

        self.kappa, self.tau, self.beta = self._init_parameters(sigma, ell)

        self.theta = theta

        varfM, varfA = self._init_variational_forms(trial, test)

        self.M, self.Msolver = self._init_operator(varfM, preconditioner="jacobi")
        self.A, self.Asolver = self._init_operator(varfA, preconditioner="petsc_amg")

        self._init_sqrtm(varfM)

        self.mean = mean
        if self.mean is None:
            self.mean = Vector(self.A.mpi_comm())
            self.A.init_vector(self.mean, 0)

        self.prng = np.random.default_rng(seed=seed)

    def sample(self) -> Function:
        """
        Draw sample from bi-Laplacian prior.

        Returns
        ----------
        dl.Function
            bi-Laplacian sample
        """
        noise = random_normal_vector(dim=self.sqrtM.size(1), prng=self.prng)  # ~ N(0, I)
        return self._transform_standard_normal_noise_with_mean_and_covariance(noise)

    def cost(self, m: Vector) -> float:
        """
        Compute cost functional
        :math:`0.5 * || m - mean ||_Q^2 = 0.5 * (m - mean)'Q(m - mean)`,
        where :math:`Q = C^{-1}` is the precision matrix
        of the finite element representation of the prior field.

        Parameters
        ----------
        m : dl.Vector
            Vertex values for which to evaluate the cost functional.

        Returns
        -------
        float
            :math:`0.5 * (m - mean)' A M^{-1} A (m - mean)`
        """
        d = m.copy()
        d.axpy(-1.0, self.mean)  # d = m - mean

        Qd = self._multiply_with_precision(d)

        return 0.5 * Qd.inner(d)

    def grad(self, m: Vector) -> Vector:
        """
        Compute gradient of discrete cost functional :math:`A M^{-1} A (m - mean)`.

        Parameters
        ----------
        m : dl.Vector
            Vertex values for which to evaluate the gradient.

        Returns
        -------
        dl.Vector
            :math:`A M^{-1} A (m - mean)`
        """
        d = m.copy()
        d.axpy(-1.0, self.mean)  # d = m - mean

        Qd = self._multiply_with_precision(d)

        return Qd

    def compute_hessian_vector_product(self, d: Vector) -> Vector:
        """
        Multiply hessian to given direction vector.

        Parameters
        ----------
        d : dl.Vector
            Vertex values of directions.

        Returns
        -------
        dl.Vector
            :math:`A M^{-1} A d`
        """
        return self._multiply_with_precision(d)

    def _validate_inputs(self, sigma, ell):
        if not (isinstance(sigma, float) or hasattr(sigma, "get_local")):
            raise TypeError(
                f"Got sigma of type {type(sigma)}, expected float or dl.Vector."
            )
        if not (isinstance(ell, float) or hasattr(ell, "get_local")):
            raise TypeError(f"Got ell of type {type(ell)}, expected float or dl.Vector.")

    def _init_parameters(self, sigma: float | Vector, ell: float | Vector):
        if not isinstance(sigma, float):
            sigma = vector_to_numpy(sigma)
        if not isinstance(ell, float):
            ell = vector_to_numpy(ell)

        kappa = get_kappa_from_ell(ell)
        tau = get_tau_from_sigma_and_ell(sigma, ell)
        beta = np.mean(kappa) * np.mean(tau) / 1.42

        if not isinstance(kappa, float):
            kappa = numpy_to_function(kappa, self.Vh)

        if not isinstance(tau, float):
            tau = numpy_to_function(tau, self.Vh)

        return kappa, tau, beta

    @staticmethod
    def _init_solver(
        solver, operator: Matrix, max_iter: int = 1000, rel_tol: float = 1e-12
    ):
        solver.set_operator(operator)
        solver.parameters["maximum_iterations"] = max_iter
        solver.parameters["relative_tolerance"] = rel_tol
        solver.parameters["error_on_nonconvergence"] = True
        solver.parameters["nonzero_initial_guess"] = False

    @staticmethod
    def _init_trial_test_functions(Vh: FunctionSpace) -> tuple[Argument, Argument]:
        return TrialFunction(Vh), TestFunction(Vh)

    def _init_variational_forms(
        self, trial: Argument, test: Argument
    ) -> tuple[Form, Form]:
        """Initialize variational forms of the operators M & A."""
        varfM = self.tau * self.kappa**2 * inner(trial, test) * dx

        if self.theta:
            varfL = self.tau * inner(self.theta * grad(trial), grad(test)) * dx
        else:
            varfL = self.tau * inner(grad(trial), grad(test)) * dx

        varf_robin = self.tau * inner(trial, test) * ds

        varfA = varfM + varfL + self.beta * varf_robin

        return varfM, varfA

    def _init_operator(
        self, varf: Form, preconditioner: PETScPreconditioner | str
    ) -> tuple[Matrix, PETScKrylovSolver]:
        """Assemble variational form of an operator, initialize corresponding solver."""
        mat = assemble(varf)
        solver = PETScKrylovSolver("cg", preconditioner)
        self._init_solver(solver, mat)
        return mat, solver

    def _init_sqrtm(self, varfM: Form):
        """Sparse decomposition :math:`M = sqrtM sqrtM^\\top`."""
        H_e_list = []
        idx = []

        for i, cell in enumerate(cells(self.Vh.mesh())):
            M_e = assemble_local(varfM, cell)
            H_e = cholesky(M_e)
            H_e_list.append(H_e)
            idx.append(self.Vh.dofmap().cell_dofs(i))

        H_block_diag = block_diag(H_e_list)
        n_cells = self.Vh.mesh().num_cells()
        L_transposed = coo_array(
            (
                np.ones(3 * n_cells, dtype=np.int32),
                (np.hstack(idx), np.arange(3 * n_cells, dtype=np.int32)),
            ),
            dtype=np.int32,
        )

        self.sqrtM = numpy_to_matrix_sparse(L_transposed @ H_block_diag)

    def _transform_standard_normal_noise_with_mean_and_covariance(
        self, noise: Vector
    ) -> Function:
        """
        Transform standard normal noise into coefficient vector with precision matrix
        :math:`(M^{-1}A)^2`  and mean `m`.

        Parameters
        ----------
        noise : dl.Vector
            Vector of length Qh.dim() with standard normal noise
        """
        s = Vector(self.A.mpi_comm())
        self.A.init_vector(s, 0)
        rhs = self.sqrtM * noise
        self.Asolver.solve(s, rhs)

        s.axpy(1.0, self.mean)

        f = Function(self.Vh)
        f.vector().axpy(1.0, s)

        return f

    def _multiply_with_precision(self, d: Vector) -> Vector:
        """
        Multiply vector with precision matrix :math:`Q = C^{-1} = A M^{-1} A`.

        Parameters
        ----------
        d : dl.Vector
            Vector to which the precision matrix is multiplied

        Returns
        -------
        dl.Vector
            :math:`Qd = C^{-1}d = A M^{-1} A d`
        """
        Ad = Vector(self.A.mpi_comm())
        self.A.init_vector(Ad, 0)
        self.A.mult(d, Ad)

        MinvAd = Vector(self.A.mpi_comm())
        self.A.init_vector(MinvAd, 1)
        self.Msolver.solve(MinvAd, Ad)

        Qd = Vector(self.A.mpi_comm())
        self.A.init_vector(Qd, 0)
        self.A.mult(MinvAd, Qd)

        return Qd


class BiLaplacianPriorNumpyWrapper:
    """
    Wrapper to support numpy inputs and outputs for the BiLaplacianPrior.

    Example
    -------
    ```python
    V, F = read_mesh("data/left_atrium.ply")  # ordered mesh file
    prior_numpy = BiLaplacianPriorNumpyWrapper(V, F, sigma=0.2, ell=10.0)
    sample = prior_numpy.sample()
    ```

    Note
    ----
    If the parameters `sigma`, `ell`, and `mean` are arrays, they are assumed to be
    ordered according to the vertex ordering in `V`. The class takes care of the
    reordering according to the DOFs of the `FunctionSpace` in the `BiLaplacianPrior`.

    Attributes
    ----------
    V : ArrayNx3
        Vertices of the mesh on which the finite element space is defined
    F : ArrayNx3
        Indices of vertices in V that are connected into faces.
    sigma : float | Array1d
        Marginal standard deviation.
        Constant in stationary case. Use Array1d to model non-stationarity.
    ell : float | Array1d
        Correlation length.
        Constant in stationary case. Use Array1d to model non-stationarity.
    mean : Array1d
        Prior mean at each vertex in V. Defaults to zero-vector.
    _prior : BiLaplacianPrior
        Instance of BiLaplacianPrior class with dolfin backend.
    """

    def __init__(
        self,
        V: ArrayNx3,
        F: ArrayNx3,
        sigma: float | Array1d,
        ell: float | Array1d,
        mean: Array1d | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Construct a bi-Laplacian prior from numpy inputs.

        Parameters
        ----------
        V : ArrayNx3
            Vertices of the mesh on which the finite element space is defined
        F : ArrayNx3
            Indices of vertices in V that are connected into faces.
        sigma : float | Array1d
            Marginal standard deviation.
            Constant in stationary case. Use Array1d to model non-stationarity.
        ell : float | Array1d
            Correlation length.
            Constant in stationary case. Use Array1d to model non-stationarity.
        mean : Array1d | None  (optional)
            Prior mean at each vertex in V. Defaults to None (zero-mean).
        seed : int | None
            Random seed for sampling.
        """
        self._validate_inputs(sigma, ell)

        self.V = V
        self.F = F

        self.sigma = sigma
        self.ell = ell

        self.mean = mean
        if self.mean is None:
            self.mean = np.zeros(V.shape[0])

        mesh = create_triangle_mesh_from_coordinates(self.V, self.F)
        Vh = FunctionSpace(mesh, "CG", 1)

        self._prior = BiLaplacianPrior(
            mesh=mesh,
            sigma=(
                sigma
                if isinstance(sigma, float)
                else numpy_to_vector(sigma, Vh, use_vertex_to_dof_map=True)
            ),
            ell=(
                ell
                if isinstance(ell, float)
                else numpy_to_vector(ell, Vh, use_vertex_to_dof_map=True)
            ),
            mean=numpy_to_vector(self.mean, Vh, use_vertex_to_dof_map=True),
            seed=seed,
        )

    def sample(self) -> Array1d:
        """
        Draw sample from bi-Laplacian prior.

        Returns
        -------
        Array1d
            1d-array with sample values at each vertex in V.
        """
        return self._prior.sample().compute_vertex_values(self._prior.Vh.mesh())

    def cost(self, m: Array1d) -> float:
        """
        Compute cost functional of the prior field for given values at each vertex.

        Parameters
        ----------
        m : Array1d
            Vertex values for which to evaluate the cost functional.

        Returns
        -------
        float
            :math:`0.5 * (m - mean)' A M^{-1} A (m - mean)`
        """
        return self._prior.cost(
            numpy_to_vector(m, self._prior.Vh, use_vertex_to_dof_map=True)
        )

    def grad(self, m: Array1d) -> Array1d:
        """
        Compute gradient of the cost functional for given values at each vertex.

        Parameters
        ----------
        m : Array1d
            Vertex values for which to evaluate the gradient.

        Returns
        -------
        Array1d
            :math:`A M^{-1} A (m - mean)`
        """
        return vector_to_numpy(
            self._prior.grad(
                numpy_to_vector(m, self._prior.Vh, use_vertex_to_dof_map=True)
            ),
            self._prior.Vh,
            use_vertex_to_dof_map=True,
        )

    def compute_hessian_vector_product(self, d: Array1d) -> Array1d:
        """
        Multiply hessian to given direction vector.

        Parameters
        ----------
        d : Array1d
            Vertex values of directions.

        Returns
        -------
        Array1d
            :math:`A M^{-1} A d`
        """
        return self._prior.compute_hessian_vector_product(
            numpy_to_vector(d, self._prior.Vh, use_vertex_to_dof_map=True)
        )

    def _validate_inputs(self, sigma, ell):
        if not (
            isinstance(sigma, float)
            or (isinstance(sigma, np.ndarray) and sigma.ndim == 1)
        ):
            raise TypeError(
                f"Got sigma of type {type(sigma)}, expected float or Array1d."
            )
        if not (
            isinstance(ell, float) or (isinstance(ell, np.ndarray) and ell.ndim == 1)
        ):
            raise TypeError(f"Got ell of type {type(ell)}, expected float or Array1d.")


class AnisotropicTensor2d(UserExpression):
    """
    User expression to model anisotropy in :math:`\\mathbb{R}^2`.

    Represents an anisotropic tensor of the form
    :math:`\\Theta =
    \\begin{bmatrix}
        \\theta_0 \\sin^2\\alpha & (\\theta_0-\\theta_1) \\sin{\\alpha} \\cos{\\alpha} \\
        (\\theta_0-\\theta_1) \\sin{\\alpha} \\cos{\\alpha} & \\theta_1 \\cos^2\\alpha.
    \\end{bmatrix}`.

    Attributes
    ----------
    alpha : float
    theta0 : float
    theta1: float
    """

    def __init__(self, alpha: float, theta0: float, theta1: float, degree: int = 1):
        self.alpha = alpha
        self.theta0 = theta0
        self.theta1 = theta1
        super().__init__(degree=degree)

    def eval(self, values, x):
        sa = np.sin(self.alpha)
        ca = np.cos(self.alpha)
        c00 = self.theta0 * sa * sa
        c01 = (self.theta0 - self.theta1) * sa * ca
        c11 = self.theta1 * ca * ca

        values[0] = c00
        values[1] = c01
        values[2] = c01
        values[3] = c11

    def value_shape(self):
        return (2, 2)
