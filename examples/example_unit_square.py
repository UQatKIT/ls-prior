# %%
import numpy as np
from dolfin import Expression, FunctionSpace, UnitSquareMesh

from prior_fields.converter import expression_to_vector, str_to_function, str_to_vector
from prior_fields.plots import plot_function
from prior_fields.prior import AnisotropicTensor2d, BiLaplacianPrior

# %%
# Minimal working example:
# zero-mean, stationary, isotropic BiLaplacianPrior on UnitSquareMesh with 5x5 vertices
mesh = UnitSquareMesh(4, 4)
prior = BiLaplacianPrior(mesh, sigma=0.1, ell=0.1)
sample = prior.sample()
plot_function(sample, show_mesh=True)

# %%
# Baseline parameters
sigma = 0.05
ell = 0.2
mesh = UnitSquareMesh(64, 64)
mean_zero = str_to_vector("0", mesh)

# %%
# Baseline: zero-mean, stationary, isotropic BiLaplacianPrior
prior = BiLaplacianPrior(mesh, sigma=sigma, ell=ell, mean=mean_zero)
sample = prior.sample()
plot_function(sample)

# %%
# non-zero mean
mean_str = "0.2*x[0]+0.4*x[1]"
plot_function(str_to_function(mean_str, mesh), title="Mean function")

prior = BiLaplacianPrior(mesh, sigma=sigma, ell=ell, mean=str_to_vector(mean_str, mesh))
sample = prior.sample()
plot_function(sample, title="BiLaplacianPrior sample with non-zero mean")

# %%
# non-stationary
ell_non_stationary = expression_to_vector(
    Expression("0.13-0.1*x[0]-0.02*x[1]", degree=1), FunctionSpace(mesh, "CG", 1)
)

prior = BiLaplacianPrior(mesh, sigma=sigma, ell=ell_non_stationary)
sample = prior.sample()
plot_function(sample, title="Non-stationary BiLaplacianPrior sample")

# %%
# anisotropic
alpha = 1 / 4 * np.pi
theta0 = 1.0
theta1 = 0.4
theta = AnisotropicTensor2d(alpha, theta0, theta1)

prior = BiLaplacianPrior(mesh, sigma=sigma, ell=ell, theta=theta)
sample = prior.sample()
plot_function(sample, title="Anisotropic BiLaplacianPrior sample")

# %%
