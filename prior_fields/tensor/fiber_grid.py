from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import circmean, circvar, mode

from prior_fields.prior.dtypes import Array1d, ArrayNx2
from prior_fields.tensor.transformer import angles_to_2d_vector_coefficients

circ_kwargs = dict(low=-np.pi / 2, high=np.pi / 2, nan_policy="omit")


def compute_uac_fiber_grid(
    max_depth: int, point_threshold: int, path: Path = Path("data/")
) -> None:
    """
    Compute and save `FiberGrid` based on all 7 human atrial geometries.

    Parameters
    ----------
    max_depth : int
        Maximum number of splits per cell (`FiberGridComputer.max_depth`).
    point_threshold : int
        Minimum number of points to split a cell (`FiberGridComputer.point_threshold`).
    path : Path, optional
        Path to which uac based data and fiber grid is saved, defaults to 'data/'.
    """
    uac_data_file = path / "uacs_fibers_tags.npy"
    logger.info(f"Load UAC-level data from {uac_data_file}")
    data = DataUAC.load(uac_data_file)

    logger.info(f"Compute fiber grid with {max_depth=} and {point_threshold=}...")
    fiber_grid = FiberGridComputer(
        uac=data.uac,
        fiber_angles=data.fiber_angles,
        anatomical_structure_tags=data.anatomical_tags,
        max_depth=max_depth,
        point_threshold=point_threshold,
    ).get_fiber_grid()

    fiber_grid_file = (
        path / f"fiber_grid_max_depth{max_depth}_point_threshold{point_threshold}.npy"
    )
    logger.info(f"Saving fiber grid to {fiber_grid_file}")
    fiber_grid.save(fiber_grid_file)


class DataUAC:
    """
    Collection of fiber and tag data for given UACs.

    Attributes
    ----------
    geometry : Array1d
        Geometry to which the vertex belongs
    uac : ArrayNx2
        Universal atrial coordinates.
    fiber_angles : Array1d
        Angle of fibers mappend to the UAC unit square.
    anatomical_tag : Array1d
        Tag for anatomical structure assignment of the data point.
    """

    def __init__(
        self,
        geometry: Array1d,
        uac: ArrayNx2,
        fiber_angles: Array1d,
        anatomical_tags: Array1d,
    ) -> None:
        self.geometry = geometry
        self.uac = uac
        self.fiber_angles = fiber_angles
        self.anatomical_tags = anatomical_tags

    @classmethod
    def load(cls, file: Path | str = "data/uacs_fibers_tags.npy") -> DataUAC:
        """
        Read 'DataUAC' from .npy file.

        Parameters
        ----------
        file : Path | str, optional
            Path to binary file, defaults to 'data/uacs_fibers_tags.npy'.

        Returns
        -------
        DataUAC
        """
        data = np.load(file)

        return DataUAC(
            geometry=data[:, 0],
            uac=data[:, 1:3],
            fiber_angles=data[:, 3],
            anatomical_tags=data[:, 4],
        )

    def save(self, file: Path | str = "data/uacs_fibers_tags.npy") -> None:
        """
        Write fiber grid to binary file.

        Parameters
        ----------
        file : Path | str, optional
            Path including file name to which the fiber grid is saved,
            defaults to 'data/uacs_fibers_tags.npy'.
        """
        logger.info(f"Saving collected data to {file}.")
        np.save(
            file,
            np.vstack(
                [self.geometry, self.uac.T, self.fiber_angles, self.anatomical_tags]
            ).T,
        )


class FiberGrid:
    """
    Adpative grid of the UAC unit square with cell size depending on the data density.
    Each cell has the following attributes:
    - Circular mean and variance of the fiber angle
    - Mode of anatomical structure tag

    Attributes
    ----------
    grid_x : Array1d
        Array of lower and upper boundaries of the cells in x-direction.
    grid_y : Array1d
        Array of lower and upper boundaries of the cells in y-direction.
    fiber_angle_circmean : Array1d
        Circular mean of fiber angles in (-pi/2, pi/2] for each cell.
    fiber_angle_circvar : Array1d
        Circular variance of fiber angles for each cell.
    anatomical_tag_mode : Array1d
        Mode of anatomical structure tag for each cell.
    """

    def __init__(
        self,
        grid_x: ArrayNx2,
        grid_y: ArrayNx2,
        fiber_angle_circmean: Array1d,
        fiber_angle_circvar: Array1d,
        anatomical_tag_mode: Array1d,
    ) -> None:
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.fiber_angle_circmean = fiber_angle_circmean
        self.fiber_angle_circvar = fiber_angle_circvar
        self.anatomical_tag_mode = anatomical_tag_mode

    @classmethod
    def load(
        cls, file: Path | str = "data/fiber_grid_max_depth8_point_threshold120.npy"
    ) -> FiberGrid:
        """
        Read 'FiberGrid' from .npy file.

        Parameters
        ----------
        file : Path | str, optional
            Path to binary file,
            defaults to 'data/fiber_grid_max_depth8_point_threshold120.npy'.

        Returns
        -------
        FiberGrid
        """
        grid = np.load(file)

        return FiberGrid(
            grid_x=grid[:, 0:2],
            grid_y=grid[:, 2:4],
            fiber_angle_circmean=grid[:, 4],
            fiber_angle_circvar=grid[:, 5],
            anatomical_tag_mode=grid[:, 6],
        )

    def plot(self, color: Literal["tag", "mean", "var"]) -> None:
        """
        Plot the adaptive grid with mean fiber vector in each cell. The cells are colored
        according to the fiber properties or anatomical region.

        Parameters
        ----------
        color : 'tag' | 'mean' | 'var'
            Property used to color the grid cells.
        """
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_aspect("equal")

        grid_centers_x = [(x[0] + x[1]) / 2 for x in self.grid_x]
        grid_centers_y = [(y[0] + y[1]) / 2 for y in self.grid_y]

        c = ax.scatter(
            grid_centers_x,
            grid_centers_y,
            c=(
                self.anatomical_tag_mode
                if color == "tag"
                else (
                    self.fiber_angle_circmean
                    if color == "mean"
                    else self.fiber_angle_circvar if color == "var" else None
                )
            ),
            s=[
                12e4 * (x[1] - x[0]) * (y[1] - y[0])
                for x, y in zip(self.grid_x, self.grid_y)
            ],
            cmap="twilight" if color == "mean" else "viridis",
            marker="s",
            alpha=0.3,
        )

        fiber_coeff_x, fiber_coeff_y = angles_to_2d_vector_coefficients(
            self.fiber_angle_circmean
        )

        ax.quiver(
            grid_centers_x,
            grid_centers_y,
            fiber_coeff_x,
            fiber_coeff_y,
            angles="xy",
            scale_units="xy",
            scale=60,
            width=0.001,
        )

        plt.colorbar(c)

        if color == "tag":
            plt.title(
                "Mean of fibers in UAC over 7 geometries with anatomical structures"
            )
            plt.tight_layout()
            plt.savefig("figures/uac_fibers_with_tag.svg")
        elif color == "mean":
            plt.title("Circular mean of fiber angle")
            plt.tight_layout()
            plt.savefig("figures/uac_fibers_with_circmean.svg")
        elif color == "var":
            plt.title("Circular variance of fiber angle")
            plt.tight_layout()
            plt.savefig("figures/uac_fibers_with_circvar.svg")

        plt.show()

    def save(
        self, file: Path | str = "data/fiber_grid_max_depth8_point_threshold120.npy"
    ) -> None:
        """
        Write fiber grid to binary file.

        Parameters
        ----------
        file : Path | str, optional
            Path including file name to which the fiber grid is saved,
            defaults to 'data/fiber_grid_max_depth8_point_threshold120.npy'.
        """
        np.save(
            file,
            np.hstack(
                [
                    self.grid_x,
                    self.grid_y,
                    np.vstack(
                        [
                            self.fiber_angle_circmean,
                            self.fiber_angle_circvar,
                            self.anatomical_tag_mode,
                        ]
                    ).T,
                ]
            ),
        )


class FiberGridComputer:
    """
    Computes a fiber grid and the fiber properties/anatomical tags for each cell from the
    UAC coordinates and fiber/tag data for the vertices of left atrial meshes.
    """

    def __init__(
        self,
        uac: ArrayNx2,
        fiber_angles: Array1d,
        anatomical_structure_tags: Array1d,
        max_depth: int = 8,
        point_threshold: int = 120,
    ) -> None:
        """
        Parameters
        ----------
        uac : ArrayNx2
            UACs at vertices.
        fiber_angles : Array1d
            Angle within (-pi/2, pi/2] representing the fiber orientation. 0 represents a
            fiber in the direction of no change in beta which is parallel to the alpha-
            axis in the UAC system.
        anatomical_structure_tags : Array1d
            Tag for anatomical structure assignment.
        max_depth : int, optional
            Maximum number of splits per cell, defaults to 8.
        point_threshold : int, optional
            Minimum number of points to split a cell, defaults to 120.
        """
        self.max_depth = max_depth
        self.point_threshold = point_threshold

        self.grid_x: list[list[float]] = []
        self.grid_y: list[list[float]] = []

        self.fiber_angle_circmean: list[float] = []
        self.fiber_angle_circvar: list[float] = []
        self.anatomical_tag_mode: list[int] = []

        self.uac = uac
        self.fiber_angles = fiber_angles
        self.anatomical_structure_tags = anatomical_structure_tags

        # Start with full unit square
        # Extend upper boundaries (we use open intervals on the right)
        self._subdivide(0, 1 + 1e-6, 0, 1 + 1e-6)

    def get_fiber_grid(self) -> FiberGrid:
        """Use grid and parameters to initialize a `FiberGrid` instance."""
        return FiberGrid(
            grid_x=np.array(self.grid_x),
            grid_y=np.array(self.grid_y),
            fiber_angle_circmean=np.array(self.fiber_angle_circmean),
            fiber_angle_circvar=np.array(self.fiber_angle_circvar),
            anatomical_tag_mode=np.array(self.anatomical_tag_mode),
        )

    def _subdivide(
        self, x_min: float, x_max: float, y_min: float, y_max: float, depth: int = 0
    ) -> None:
        """
        Recursively, split cell (x_min, x_max) x (y_min, y_max) in four quadrants until
        the number of points in each cell is smaller than the `point_threshold` or until
        `max_depth` is reached.

        On each final cell, compute fiber properties and anatomical tag for each cell.

        The resulting grid is assembled in `self.grid_x` and `self.grid_y`.
        """
        if depth > self.max_depth:
            return

        # Select data points within the current cell
        mask = (
            (self.uac[:, 0] >= x_min)
            & (self.uac[:, 0] < x_max)
            & (self.uac[:, 1] >= y_min)
            & (self.uac[:, 1] < y_max)
        )
        data_count = mask.sum()

        if data_count < self.point_threshold or depth == self.max_depth:
            # Don't further split the current cell
            self.grid_x.append([x_min, x_max])
            self.grid_y.append([y_min, y_max])

            n_min = ceil(self.point_threshold / 10)
            if data_count < n_min:
                # Find nearest neighbors of cell to compute parameters
                midpoint = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                tree = KDTree(self.uac)
                _, idx = tree.query(midpoint, k=n_min, p=np.infty)
                mask[idx] = True

            # Compute properties in the cell
            self.fiber_angle_circmean.append(
                circmean(self.fiber_angles[mask], **circ_kwargs)
            )
            self.fiber_angle_circvar.append(
                circvar(self.fiber_angles[mask], **circ_kwargs)
            )
            self.anatomical_tag_mode.append(
                mode(self.anatomical_structure_tags[mask]).mode
            )
        else:
            # Subdivide into four quadrants
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2

            self._subdivide(x_min, x_mid, y_min, y_mid, depth + 1)  # Bottom-left
            self._subdivide(x_mid, x_max, y_min, y_mid, depth + 1)  # Bottom-right
            self._subdivide(x_min, x_mid, y_mid, y_max, depth + 1)  # Top-left
            self._subdivide(x_mid, x_max, y_mid, y_max, depth + 1)  # Top-right
