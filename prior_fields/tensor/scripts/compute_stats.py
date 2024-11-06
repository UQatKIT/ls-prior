# %%
from collections.abc import Callable
from datetime import datetime

import numpy as np

from prior_fields.prior.dtypes import ArrayNx3
from prior_fields.tensor.fiber_grid import get_fiber_parameters_from_uac_data
from prior_fields.tensor.parameters import Geometry
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


# %%
################################
# UAC-based coordinate systems #
################################
V_dict, F_dict, uac_dict, fibers_dict, tags_dict = read_all_human_atrial_fiber_meshes()
keys = sorted(V_dict.keys())

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
