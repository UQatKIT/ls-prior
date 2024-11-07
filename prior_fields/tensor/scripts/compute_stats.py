# %%
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from prior_fields.prior.converter import create_triangle_mesh_from_coordinates
from prior_fields.prior.dtypes import ArrayNx3
from prior_fields.prior.prior import BiLaplacianPrior, BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.fiber_grid import get_fiber_parameters_from_uac_data
from prior_fields.tensor.parameters import Geometry, PriorParameters
from prior_fields.tensor.reader import (
    _map_fibers_and_tags_to_vertices,
    read_all_human_atrial_fiber_meshes,
    read_raw_atrial_mesh,
)
from prior_fields.tensor.tangent_space import get_uac_basis_vectors


def profile(f: Callable, **kwargs):
    start = datetime.now()
    _ = f(**kwargs)
    end = datetime.now()
    duration = end - start
    return duration.seconds + duration.microseconds * 1e-6


V_dict, F_dict, uac_dict, fibers_dict, tags_dict = read_all_human_atrial_fiber_meshes()
keys = sorted(V_dict.keys())

# %%
################################
# UAC-based coordinate systems #
################################
directions_constant_beta_dict: dict[int, ArrayNx3] = dict()
directions_constant_alpha_dict: dict[int, ArrayNx3] = dict()

durations_uac_coordinates: list[float] = []
for i in keys:
    durations_uac_coordinates.append(
        profile(
            get_uac_basis_vectors,
            **dict(V=V_dict[i], F=F_dict[i], uac=uac_dict[i]),
        )
    )
    print()

print("Stats for constructing the UAC-based coordinate systems for the 7 geometries:")
print(
    f"mean={np.mean(durations_uac_coordinates):.2f}s, "
    f"std={np.std(durations_uac_coordinates):.2f}s"
)


# %%
###################################
# Map fibers and tags to vertices #
###################################
durations_map_to_vertices: list[float] = []
for i in keys:
    V, F, uac, fibers_on_faces, tags_of_faces = read_raw_atrial_mesh(Geometry(i))

    durations_map_to_vertices.append(
        profile(
            _map_fibers_and_tags_to_vertices,
            **dict(F=F, fibers_on_faces=fibers_on_faces, tags_of_faces=tags_of_faces),
        )
    )
    print()

print("Stats for mapping data to vertices for the 7 geometries:")
print(
    f"mean={np.mean(durations_map_to_vertices):.2f}s, "
    f"std={np.std(durations_map_to_vertices):.2f}s"
)

# %%
###################################
# Compute parameters for geometry #
###################################
durations_compute_fiber_paramters: list[float] = []
for i in keys:
    durations_compute_fiber_paramters.append(
        profile(
            get_fiber_parameters_from_uac_data,
            **dict(V=V_dict[i], F=F_dict[i], uac=uac_dict[i], k=100),
        )
    )
    print()

print("Stats for computing fiber parameters for the 7 geometries:")
print(
    f"mean={np.mean(durations_compute_fiber_paramters):.2f}s, "
    f"std={np.std(durations_compute_fiber_paramters):.2f}s"
)

# %%
##################################
# Initialize bi-Laplacian priors #
##################################
duration_prior_init = profile(
    BiLaplacianPriorNumpyWrapper,
    **dict(V=V_dict[3], F=F_dict[3], sigma=0.2, ell=0.1, seed=1),
)

print(
    "Duration for initializing a baseline prior (wrapper) "
    f"on geometry 3: {duration_prior_init:.2f}"
)

mesh3 = create_triangle_mesh_from_coordinates(V_dict[3], F_dict[3])
duration_prior_init = profile(
    BiLaplacianPrior,
    **dict(mesh=mesh3, sigma=0.2, ell=0.1, seed=1),
)

print(
    "Duration for initializing a baseline prior (no wrapper) "
    f"on geometry 3: {duration_prior_init:.2f}"
)

params = PriorParameters.load(Path("data/parameters/params_3.npy"))
duration_prior_init = profile(
    BiLaplacianPriorNumpyWrapper,
    **dict(
        V=V_dict[3],
        F=F_dict[3],
        sigma=params.sigma,
        ell=params.ell,
        mean=params.mean,
        seed=1,
    ),
)

print(
    "Duration for initializing a data-informed prior "
    f"on geometry 3: {duration_prior_init:.2f}"
)

# %%
###################################
# Sample from bi-Laplacian priors #
###################################
prior = BiLaplacianPriorNumpyWrapper(
    V=V_dict[3], F=F_dict[3], sigma=0.2, ell=0.1, seed=1
)

durations_prior_sample = [profile(prior.sample) for _ in range(100)]
print(
    "Duration for sampling from the baseline prior:\n"
    f"First sample: {durations_prior_sample[0]:.2f}\n"
    f"Further samples: {np.mean(durations_prior_sample[1:]):.2f} "
    f"({np.std(durations_prior_sample[1:]):.2f})"
)

prior = BiLaplacianPriorNumpyWrapper(
    V=V_dict[3],
    F=F_dict[3],
    sigma=params.sigma,
    ell=params.ell,
    mean=params.mean,
    seed=1,
)

durations_prior_sample = [profile(prior.sample) for _ in range(100)]
print(
    "Duration for sampling from the data-informed prior:\n"
    f"First sample: {durations_prior_sample[0]:.2f}\n"
    f"Further samples: {np.mean(durations_prior_sample[1:]):.2f} "
    f"({np.std(durations_prior_sample[1:]):.2f})"
)

# %%
