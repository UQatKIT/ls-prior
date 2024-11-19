# %%
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from pyacvd import Clustering

from prior_fields.parameterization.parameters import (
    Geometry,
    PriorParameters,
    get_fiber_parameters_from_uac_data,
)
from prior_fields.parameterization.reader import (
    _map_fibers_and_tags_to_vertices,
    read_all_human_atrial_fiber_meshes,
    read_raw_atrial_mesh,
)
from prior_fields.parameterization.tangent_space import get_uac_based_coordinates
from prior_fields.prior.converter import create_triangle_mesh_from_coordinates
from prior_fields.prior.dtypes import ArrayNx3
from prior_fields.prior.plots import get_poly_data
from prior_fields.prior.prior import BiLaplacianPrior, BiLaplacianPriorNumpyWrapper

font = {"family": "times", "size": 20}
rc("font", **font)


# %%
def profile(f: Callable, **kwargs):
    start = datetime.now()
    output = f(**kwargs)
    end = datetime.now()
    duration = end - start
    return duration.seconds + duration.microseconds * 1e-6, output


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
            get_uac_based_coordinates,
            **dict(V=V_dict[i], F=F_dict[i], uac=uac_dict[i]),
        )[0]
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
        )[0]
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
durations_compute_fiber_parameters: list[float] = []
for i in keys:
    durations_compute_fiber_parameters.append(
        profile(
            get_fiber_parameters_from_uac_data,
            **dict(V=V_dict[i], F=F_dict[i], uac=uac_dict[i], k=100),
        )[0]
    )
    print()

print("Stats for computing fiber parameters for the 7 geometries:")
print(
    f"mean={np.mean(durations_compute_fiber_parameters):.2f}s, "
    f"std={np.std(durations_compute_fiber_parameters):.2f}s"
)

# %%
##################################
# Initialize bi-Laplacian priors #
##################################
duration_prior_init = profile(
    BiLaplacianPriorNumpyWrapper,
    **dict(V=V_dict[3], F=F_dict[3], sigma=0.2, ell=0.1, seed=1),
)[0]

print(
    "Duration for initializing a baseline prior (wrapper) "
    f"on geometry 3: {duration_prior_init:.2f}"
)

mesh3 = create_triangle_mesh_from_coordinates(V_dict[3], F_dict[3])
duration_prior_init = profile(
    BiLaplacianPrior,
    **dict(mesh=mesh3, sigma=0.2, ell=0.1, seed=1),
)[0]

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
)[0]

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

durations_prior_sample = [profile(prior.sample)[0] for _ in range(100)]
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

durations_prior_sample = [profile(prior.sample)[0] for _ in range(100)]
print(
    "Duration for sampling from the data-informed prior:\n"
    f"First sample: {durations_prior_sample[0]:.2f}\n"
    f"Further samples: {np.mean(durations_prior_sample[1:]):.2f} "
    f"({np.std(durations_prior_sample[1:]):.2f})"
)

# %%
#################################################
# Prior initialization with different mesh size #
#################################################
cluster_sizes = (np.array([[2], [4], [8]]) * (10 ** np.arange(2, 5))).flatten("F")
initialization_times: list[float] = []
sampling_times: list[list[float]] = []
for n_clusters in cluster_sizes:
    poly_data3 = get_poly_data(V_dict[3], F_dict[3])
    c = Clustering(poly_data3)
    c.cluster(n_clusters)
    poly_data3_subsamled = c.create_mesh()

    V3_subsampled = poly_data3_subsamled.points
    F3_subsampled = poly_data3_subsamled.faces.reshape(-1, 4)[
        :, 1:  # remove first column (is always 3 for triangle mesh)
    ]

    duration_prior_init, prior = profile(
        BiLaplacianPriorNumpyWrapper,
        **dict(V=V3_subsampled, F=F3_subsampled, sigma=0.2, ell=0.1, seed=1),
    )
    initialization_times.append(duration_prior_init)
    print(
        f"Duration for initializing a prior on a mesh with {n_clusters} vertices: "
        f"{duration_prior_init:.2f}"
    )

    durations_prior_sample = [profile(prior.sample)[0] for _ in range(1000)]
    sampling_times.append(durations_prior_sample)
    print(
        f"Duration for sampling from on a mesh with {n_clusters} vertices:\n"
        f"First sample: {durations_prior_sample[0]:.2f}\n"
        f"Further samples: {np.mean(durations_prior_sample[1:]):.2f} "
        f"({np.std(durations_prior_sample[1:]):.2f})"
    )

# %%
further_sample_means = [np.mean(t[1:]) for t in sampling_times]
further_sample_lower_q = [np.quantile(t[1:], 0.005) for t in sampling_times]
further_sample_upper_q = [np.quantile(t[1:], 0.995) for t in sampling_times]
line_x = np.array([min(cluster_sizes), max(cluster_sizes)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # type: ignore

ax1.plot(
    cluster_sizes,
    initialization_times,
    label="initialization",
    color="#009682",
    marker="x",
)
ax1.plot(line_x, line_x / 2e3, color="black", linestyle=":", label="linear growth")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Number of Vertices")
ax1.set_ylabel("Initialization Time")
ax1.legend()

ax2.plot(
    cluster_sizes,
    [t[0] for t in sampling_times],
    label="first sample",
    color="#009682",
    marker="x",
)
ax2.plot(
    cluster_sizes,
    further_sample_means,
    label="further sample",
    color="#4664aa",
    marker="x",
)
ax2.fill_between(
    cluster_sizes,
    further_sample_lower_q,
    further_sample_upper_q,
    color="#4664aa",
    alpha=0.2,
)
ax2.plot(line_x, line_x / 5e5, color="black", linestyle=":", label="linear growth")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Number of Vertices")
ax2.set_ylabel("Sampling Times")
ax2.legend()

plt.tight_layout()
plt.savefig("figures/other/computation_times_for_different_mesh_sizes.png")
plt.show()

# %%
