"""
This file shows how to combine the results from `03_example_atrium.py` with conduction
velocity data to construct a conduction velocity tensor that can be used in the Bayesian
inverse problem of cardiac electrophysiology.
"""

# %%
from pathlib import Path

from prior_fields.parameterization.parameters import Geometry, PriorParameters
from prior_fields.parameterization.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)
from prior_fields.parameterization.tangent_space import get_vhm_based_coordinates
from prior_fields.parameterization.transformer import (
    sample_to_angles,
    shift_angles_by_mean,
)
from prior_fields.prior.converter import scale_mesh_to_unit_cube
from prior_fields.prior.prior import BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.conduction_velocity import angles_to_cv_tensor

geometry = Geometry(1)

#############
# Read data #
#############
# %%
# Read vertices, faces, UACs and anatomical tags
V, F, uac, _, tags = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(geometry)
V = scale_mesh_to_unit_cube(V)

# Compute VHM-based reference coordinates
basis_x, basis_y, _ = get_vhm_based_coordinates(V, F)

# Load data-informed parameters computed in
# `prior_fields/tensor/scripts/02a_compute_prior_parameters.py`
params = PriorParameters.load(Path(f"data/parameters/params_{geometry.value}.npy"))

######################
# Bi-Laplacian Prior #
######################
# %%
# Initialize prior
prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=params.sigma, ell=params.ell)

# Transform prior sample to angle field
angles = shift_angles_by_mean(
    sample_to_angles(prior.sample()), sample_to_angles(params.mean)
)

##############################
# Conduction velocity tensor #
##############################
# %%
# Add CV information to angle field to assemble the corresponding CV tensor for the BIP
cv_tensor = angles_to_cv_tensor(prior.sample(), tags, basis_x, basis_y)
cv_tensor

# %%
