# %%
from pathlib import Path

from prior_fields.prior.converter import scale_mesh_to_unit_cube
from prior_fields.prior.prior import BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.conduction_velocity import sample_to_cv_tensor
from prior_fields.tensor.parameterization import Geometry, PriorParameters
from prior_fields.tensor.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)
from prior_fields.tensor.tangent_space import get_vhm_based_coordinates

geometry = Geometry(1)

# %%
V, F, uac, fibers, tags = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(
    geometry
)
V = scale_mesh_to_unit_cube(V)
basis_x, basis_y, _ = get_vhm_based_coordinates(V, F)
params = PriorParameters.load(Path(f"data/parameters/params_{geometry.value}.npy"))

# %%
prior = BiLaplacianPriorNumpyWrapper(
    V, F, sigma=params.sigma, ell=params.ell, mean=params.mean
)

# %%
cv_tensor = sample_to_cv_tensor(prior.sample(), tags, basis_x, basis_y)
cv_tensor

# %%
