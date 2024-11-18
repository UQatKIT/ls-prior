# %%
import numpy as np
from dolfin import FunctionSpace, UnitSquareMesh

from prior_fields.prior.converter import function_to_numpy, numpy_to_function
from prior_fields.prior.plots import plot_function
from prior_fields.prior.prior import BiLaplacianPrior

mesh = UnitSquareMesh(64, 64)
Vh = FunctionSpace(mesh, "CG", 1)

prior = BiLaplacianPrior(mesh, sigma=0.2, ell=0.2, seed=1)

samples_list = []
for _ in range(10000):
    samples_list.append(function_to_numpy(prior.sample()))

samples = np.array(samples_list)

# %%
mean = samples.mean(axis=0)
covariance = np.cov(samples, rowvar=False)

mean_function = numpy_to_function(mean, Vh)
plot_function(mean_function, title="Empirical mean")

pointwise_variance = np.diag(covariance)
variance_function = numpy_to_function(pointwise_variance, Vh)
plot_function(variance_function, title="Empirical pointwise variance")
