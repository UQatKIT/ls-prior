from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.spatial import KDTree
from scipy.stats import circmean, mode

from prior_fields.parameterization.fiber_grid import DataUAC, FiberGrid
from prior_fields.parameterization.tangent_space import (
    get_angles_in_tangent_space,
    get_uac_based_coordinates,
    get_vhm_based_coordinates,
)
from prior_fields.parameterization.transformer import (
    angles_to_3d_vector,
    angles_to_sample,
)
from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3


class Geometry(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    ATLAS = "A"
    UNKNOWN = "unknown"


@dataclass
class PriorParameters:
    geometry: Geometry
    mean: Array1d
    sigma: Array1d
    ell: Array1d

    @classmethod
    def load(cls, file: Path) -> PriorParameters:
        """
        Read 'PriorParameters' from .npy file.

        Parameters
        ----------
        file : Path | str, optional
            Path to binary file.

        Returns
        -------
        PriorParameters
        """
        logger.info(f"Load parameters from {file}.")
        data = np.load(file)

        # Extract geometry from file name
        pattern = r"params_([1234567A])\.npy"
        matched = re.search(pattern, str(file))
        geometry: Geometry
        if matched is None:
            geometry = Geometry.UNKNOWN
        else:
            g = matched.group(1)
            if g == "A":
                geometry = Geometry.ATLAS
            elif g.isnumeric():
                geometry = Geometry(int(g))

        return PriorParameters(
            geometry=geometry, mean=data[0], sigma=data[1], ell=data[2]
        )

    def save(self, path: Path = Path("data/parameters/")) -> None:
        """
        Write parameters for `BiLaplacianPriorNumpyWrapper` to binary file.

        Parameters
        ----------
        path : Path, optional
            Path to which the parameters are saved, defaults to 'data/parameters/'.
        """
        Path.mkdir(path, exist_ok=True)
        file = path / f"params_{self.geometry.value}.npy"
        logger.info(f"Saving collected data to {file}.")
        np.save(file, np.vstack([self.mean, self.sigma, self.ell]))


def get_fiber_parameters_from_uac_data(
    V: ArrayNx3,
    F: ArrayNx3,
    uac: ArrayNx2,
    k: int = 20,
    file: Path | str = "data/uacs_fibers_tags.npy",
) -> tuple[Array1d, Array1d, Array1d]:
    """
    Compute fiber parameters for given mesh based on k nearest UAC-neighbors over all
    source geometries.

    Parameters
    ----------
    V : ArrayNx3
        Vertex coordinates
    F : ArrayNx3
        Faces
    uac : ArrayNx2
        UACs of the vertices
    k : int > 1, optional
        Number of nearest neighbors considered within each source geometry,
        defaults to 20.
    file : Path | str, optional
        Path to the fiber data collected in UACs,
        defaults to 'data/uacs_fibers_tags.npy'.

    Returns
    -------
    (Array1d, Array1d, Array1d)
        Mean and pointwise standard deviation in the prior's range (not angles), tags for
        anatomical structure of each vertex.
    """
    data_uac = DataUAC.load(file)
    geometries = np.unique(data_uac.geometry)

    angles_uac_list: list[Array1d] = []
    for g in geometries:
        idx_g = data_uac.geometry == g

        # Find k nearest neighbors in g to each UAC in the target geometry
        tree = KDTree(data_uac.uac[idx_g])
        _, idx_neighbors = tree.query(uac, k=k, p=2)

        # Get angles of kNN based on UAC
        angles_uac_list.append(
            circmean(
                data_uac.fiber_angles[idx_neighbors],
                axis=1,
                low=-np.pi / 2,
                high=np.pi / 2,
                nan_policy="omit",
            )
        )

    # Collect mean angles from different geometries for each vertex of the target mesh
    angles_uac = np.vstack(angles_uac_list).T.reshape(-1)

    # Transform angles to 3d fiber vectors on vertices of target geometry
    alpha_axes, beta_axes = get_uac_based_coordinates(V, F, uac)
    alpha_axes_repeated = _repeat_array(alpha_axes, len(geometries))
    beta_axes_repeated = _repeat_array(beta_axes, len(geometries))
    fibers = angles_to_3d_vector(
        angles=angles_uac, x_axes=alpha_axes_repeated, y_axes=beta_axes_repeated
    )

    # Compute orthonormal bases of tangent spaces and fiber angles within these bases
    x_axes, y_axes, _ = get_vhm_based_coordinates(V, F)
    x_axes_repeated = _repeat_array(x_axes, len(geometries))
    y_axes_repeated = _repeat_array(y_axes, len(geometries))
    angles_vhm = get_angles_in_tangent_space(fibers, x_axes_repeated, y_axes_repeated)

    # Transform angles to values in (-inf, inf) for BiLaplacianPrior parameterization
    transformed_observations = angles_to_sample(angles_vhm).reshape(-1, len(geometries))

    # Compute parameters
    prior_mean = np.nanmean(transformed_observations, axis=1)
    prior_sigma = np.nanstd(transformed_observations, axis=1)
    mode_tag = mode(data_uac.anatomical_tags[idx_neighbors], axis=1).mode

    # Handle missing values
    prior_mean[np.isnan(prior_mean)] = np.nanmean(prior_mean)
    prior_sigma[np.isnan(prior_sigma)] = np.nanmean(prior_sigma)

    # Replace zeros in sigma
    prior_sigma = np.clip(prior_sigma, a_min=1e-6, a_max=None)

    return prior_mean, prior_sigma, mode_tag


def _repeat_array(array: ArrayNx3, n: int) -> ArrayNx3:
    return np.vstack(
        [array[:, 0].repeat(n), array[:, 1].repeat(n), array[:, 2].repeat(n)]
    ).T


def get_fiber_parameters_from_uac_grid(
    uac: ArrayNx2,
    file: Path | str = "data/fiber_grid_max_depth8_point_threshold120.npy",
) -> tuple[Array1d, Array1d]:
    """
    Map mean and variance of fiber angles from `FiberGrid` to vertices based on the given
    UACs.

    Parameters
    ----------
    uac : ArrayNx2
        Universal atrial coordinates of vertices.
    file : Path | str, optional
        Path to binary file with fiber grid,
        defaults to 'data/fiber_grid_max_depth8_point_threshold120.npy'.

    Returns
    -------
    (Array1d, Array1d)
        Arrays with mean and standard deviation of fiber angles.
    """
    fiber_grid = FiberGrid.load(file)

    fiber_mean = np.zeros(uac.shape[0])
    fiber_var = np.zeros(uac.shape[0])
    unmatched_vertices = []

    for i in range(fiber_mean.shape[0]):
        j = np.where(
            (uac[i, 0] >= fiber_grid.grid_x[:, 0])
            & (uac[i, 0] < fiber_grid.grid_x[:, 1])
            & (uac[i, 1] >= fiber_grid.grid_y[:, 0])
            & (uac[i, 1] < fiber_grid.grid_y[:, 1])
        )[0]
        try:
            fiber_mean[i] = fiber_grid.fiber_angle_circmean[j[0]]
            fiber_var[i] = fiber_grid.fiber_angle_circvar[j[0]]
        except IndexError:
            fiber_var[i] = np.nan
            unmatched_vertices.append(i)

    if len(unmatched_vertices) > 0:
        logger.warning(
            "Couldn't find grid cell for "
            f"{100 * len(unmatched_vertices) / uac.shape[0]:.2f}%"
            " of the vertices."
        )

    return fiber_mean, fiber_var
