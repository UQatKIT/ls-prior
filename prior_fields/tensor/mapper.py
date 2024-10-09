import re
from typing import Literal, overload

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import circmean, circstd, mode

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3


def get_dict_with_adjacent_faces_for_each_vertex(F: ArrayNx3) -> dict[int, list[int]]:
    adjacent_faces: dict[int, list[int]] = {i: [] for i in range(F.max() + 1)}

    for face_index, face_vertices in enumerate(F):
        for vertex_id in face_vertices:
            adjacent_faces[vertex_id].append(face_index)

    return adjacent_faces


def map_vectors_from_faces_to_vertices(
    vecs: ArrayNx3, adjacent_faces: dict[int, list[int]]
) -> ArrayNx3:
    """Map vectors defined on face-level to vertices.

    For each vertex, the resulting vector is the component-wise mean over the vectors of
    all adjacent faces.

    Parameters
    ----------
    vecs : ArrayNx3
        Vectors defined on face-level.
    adjacent_faces : dict[int, list[int]]
        Dictionary where the keys are the vertex indices and the values are lists of
        vertex indices of all adjacent faces.

    Returns
    -------
    ArrayNx3
        Vectors mapped to vertex-level.
    """
    return np.array([vecs[i].mean(axis=0) for i in adjacent_faces.values()])


def map_categories_from_faces_to_vertices(
    categories: Array1d, adjacent_faces: dict[int, list[int]]
) -> Array1d:
    """Map categories defined on face-level to vertices.

    For each vertex, the resulting tag is the mode over the tags of the adjacent faces.

    Parameters
    ----------
    categories : Array1d
        Categories on face-level.
    adjacent_faces : dict[int, list[int]]
        Dictionary where the keys are the vertex indices and the values are lists of
        vertex indices of all adjacent faces.

    Returns
    -------
    Array1d
        Categories mapped to vertex-level.
    """
    return np.array([mode(categories[i]).mode for i in adjacent_faces.values()])


def map_fibers_to_tangent_space(fibers: ArrayNx3, x: ArrayNx3, y: ArrayNx3) -> ArrayNx3:
    """Map fibers to tangent spaces spanned by x and y.

    Parameters
    ----------
    fibers : ArrayNx3
        (n, 3) array where each row is a fiber vector.
    x : ArrayNx3
        (n, 3) array where each row is a vector in the tangent space.
    y : ArrayNx3
        (n, 3) array where each row is another vector in the tangent space.

    Returns
    -------
    ArrayNx3
        Fibers mapped to the tangent spaces.
    """
    fibers_x, fibers_y = get_coefficients(fibers, x, y)

    return fibers_x[:, np.newaxis] * x + fibers_y[:, np.newaxis] * y


def get_coefficients(
    fibers: ArrayNx3, x: ArrayNx3, y: ArrayNx3
) -> tuple[ArrayNx3, ArrayNx3]:
    """Get coefficients to write fibers in tangent space basis.

    Parameters
    ----------
    fibers : ArrayNx3
        (n, 3) array where each row is a fiber vector.
    x : ArrayNx3
        (n, 3) array where each row is a vector in the tangent space.
    y : ArrayNx3
        (n, 3) array where each row is another vector in the tangent space.

    Returns
    -------
    (ArrayNx3, ArrayNx3)

    """
    return np.einsum("ij,ij->i", fibers, x), np.einsum("ij,ij->i", fibers, y)


class FiberGrid:
    """
    Adpative grid of the UAC unit square with cell size depending on the data density.
    Each cell has the following attributes:
    - Mean coefficients of fibers in UAC basis
    - Circular mean and standard deviation of the fiber angle
    - Mode of anatomical structure tag

    Attributes
    ----------
    max_depth : int
        Maximum number of splits per cell.
    point_threshold : int
        Minimum number of points per cell.
    grid_x : list[list[float]]
        List of lower and upper boundaries of the cells in x-direction.
    grid_y : list[list[float]]
        List of lower and upper boundaries of the cells in y-direction.
    fiber_coeff_x_mean : list[float]
        Mean first fiber coefficients for each cell.
    fiber_coeff_y_mean : list[float]
        Mean second fiber coefficients for each cell.
    fiber_angle_circmean : list[float]
        Circular mean of fiber angles in (-pi, pi] for each cell.
    fiber_angle_circstd : list[float]
        Circular standard deviation of fiber angles for each cell.
    anatomical_tag_mode : list[int]
        Mode of anatomical structure tag for each cell.
    """

    @overload
    def __init__(
        self,
        uac: ArrayNx2,
        fiber_coeffs_x: Array1d,
        fiber_coeffs_y: Array1d,
        fiber_angles: Array1d,
        anatomical_structure_tags: Array1d,
        grid_x: None,
        grid_y: None,
        fiber_coeff_x_mean: None,
        fiber_coeff_y_mean: None,
        fiber_angle_circmean: None,
        fiber_angle_circstd: None,
        anatomical_tag_mode: None,
        max_depth: int = 5,
        point_threshold: int = 100,
    ) -> None: ...

    @overload
    def __init__(
        self,
        uac: None,
        fiber_coeffs_x: None,
        fiber_coeffs_y: None,
        fiber_angles: None,
        anatomical_structure_tags: None,
        grid_x: list[list[float]],
        grid_y: list[list[float]],
        fiber_coeff_x_mean: list[float],
        fiber_coeff_y_mean: list[float],
        fiber_angle_circmean: list[float],
        fiber_angle_circstd: list[float],
        anatomical_tag_mode: list[int],
        max_depth: int,
        point_threshold: int,
    ) -> None: ...

    def __init__(
        self,
        uac: ArrayNx2 | None = None,
        fiber_coeffs_x: Array1d | None = None,
        fiber_coeffs_y: Array1d | None = None,
        fiber_angles: Array1d | None = None,
        anatomical_structure_tags: Array1d | None = None,
        grid_x: list[list[float]] | None = None,
        grid_y: list[list[float]] | None = None,
        fiber_coeff_x_mean: list[float] | None = None,
        fiber_coeff_y_mean: list[float] | None = None,
        fiber_angle_circmean: list[float] | None = None,
        fiber_angle_circstd: list[float] | None = None,
        anatomical_tag_mode: list[int] | None = None,
        max_depth: int = 5,
        point_threshold: int = 100,
    ) -> None:
        """
        Parameters
        ----------
        uac : ArrayNx2
            UACs at vertices.
        fiber_coeffs_x : Array1d
            Fiber coefficient for first UAC-based tangent space coordinate.
        fiber_coeffs_y : Array1d
            Fiber coefficient for second UAC-based tangent space coordinate.
        fiber_angles : Array1d
            Angle within (-pi, pi] representing the fiber orientation.0 represents a
            fiber in the direction of no change in beta which is parallel to the alpha-
            axis in the UAC system.
        anatomical_structure_tags : Array1d
            Tag for anatomical structure assignment.
        max_depth : int, optional
            Maximum number of splits per cell, defaults to 5.
        point_threshold : int, optional
            Minimum number of points per cell, defaults to 100.
        """
        self.max_depth = max_depth
        self.point_threshold = point_threshold

        if grid_x is None:
            self.grid_x: list[list[float]] = []
            self.grid_y: list[list[float]] = []

            self.fiber_coeff_x_mean: list[float] = []
            self.fiber_coeff_y_mean: list[float] = []
            self.fiber_angle_circmean: list[float] = []
            self.fiber_angle_circstd: list[float] = []
            self.anatomical_tag_mode: list[int] = []

            self._uac = uac
            self._fiber_coeffs_x = fiber_coeffs_x
            self._fiber_coeffs_y = fiber_coeffs_y
            self._fiber_angles = fiber_angles
            self._anatomical_structure_tags = anatomical_structure_tags

            self._subdivide(0, 1, 0, 1)  # Start with full unit square

        elif uac is None:
            self.grid_x = grid_x
            self.grid_y = grid_y
            self.fiber_coeff_x_mean = fiber_coeff_x_mean
            self.fiber_coeff_y_mean = fiber_coeff_y_mean
            self.fiber_angle_circmean = fiber_angle_circmean
            self.fiber_angle_circstd = fiber_angle_circstd
            self.anatomical_tag_mode = anatomical_tag_mode

    @classmethod
    def from_binary_file(cls, path: str):

        grid = np.load(path)
        grid_x: list[list[float]] = grid[:, 0:2].tolist()
        grid_y: list[list[float]] = grid[:, 2:4].tolist()
        fiber_coeff_x_mean: list[float] = grid[:, 4].tolist()
        fiber_coeff_y_mean: list[float] = grid[:, 5].tolist()
        fiber_angle_circmean: list[float] = grid[:, 6].tolist()
        fiber_angle_circstd: list[float] = grid[:, 7].tolist()
        anatomical_tag_mode: list[int] = grid[:, 8].tolist()

        match_max_depth = re.search(r"max_depth(\d+)_", path)
        max_depth = int(match_max_depth.group(1)) if match_max_depth is not None else 5
        match_point_threshold = re.search(r"point_threshold(\d+).npy", path)
        point_threshold = (
            int(match_point_threshold.group(1))
            if match_point_threshold is not None
            else 100
        )

        return FiberGrid(
            uac=None,
            fiber_coeffs_x=None,
            fiber_coeffs_y=None,
            fiber_angles=None,
            anatomical_structure_tags=None,
            grid_x=grid_x,
            grid_y=grid_y,
            fiber_coeff_x_mean=fiber_coeff_x_mean,
            fiber_coeff_y_mean=fiber_coeff_y_mean,
            fiber_angle_circmean=fiber_angle_circmean,
            fiber_angle_circstd=fiber_angle_circstd,
            anatomical_tag_mode=anatomical_tag_mode,
            max_depth=max_depth,
            point_threshold=point_threshold,
        )

    def plot(self, color: Literal["tag", "mean", "std"]) -> None:
        """
        Plot the adaptive grid with mean fiber vector in each cell. The cells are colored
        according to the fiber porperties or anatomical region.

        Parameters
        ----------
        color : 'tag' | 'mean' | 'std'
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
                    else self.fiber_angle_circstd if color == "std" else None
                )
            ),
            s=[
                12e4 * (x[1] - x[0]) * (y[1] - y[0])
                for x, y in zip(self.grid_x, self.grid_y)
            ],
            marker="s",
            alpha=0.3,
        )

        ax.quiver(
            grid_centers_x,
            grid_centers_y,
            self.fiber_coeff_x_mean,
            self.fiber_coeff_y_mean,
            angles="xy",
            scale_units="xy",
            scale=50,
            width=0.001,
        )

        plt.colorbar(c)

        if color == "tag":
            plt.title(
                "Mean of fibers in UAC over 7 geometries with anatomical structures"
            )
        elif color == "mean":
            plt.title("Circular mean of fiber angle")
        elif color == "std":
            plt.title("Circular standard deviation of fiber angle")

        plt.show()

    def save(self, filename_prefix: str = "fiber_grid"):
        """Write grid to binary file."""
        np.save(
            f"data/LGE-MRI-based/{filename_prefix}"
            f"_max_depth{self.max_depth}"
            f"_point_threshold{self.point_threshold}.npy",
            np.hstack(
                [
                    self.grid_x,
                    self.grid_y,
                    np.vstack(
                        [
                            self.fiber_coeff_x_mean,
                            self.fiber_coeff_y_mean,
                            self.fiber_angle_circmean,
                            self.fiber_angle_circstd,
                            self.anatomical_tag_mode,
                        ]
                    ).T,
                ]
            ),
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
            (self._uac[:, 0] >= x_min)
            & (self._uac[:, 0] < x_max)
            & (self._uac[:, 1] >= y_min)
            & (self._uac[:, 1] < y_max)
        )
        data_count = mask.sum()

        if data_count < self.point_threshold or depth == self.max_depth:
            # Don't further split the current cell
            self.grid_x.append([x_min, x_max])
            self.grid_y.append([y_min, y_max])

            # Compute properties in the cell
            self.fiber_coeff_x_mean.append(self._fiber_coeffs_x[mask].mean())
            self.fiber_coeff_y_mean.append(self._fiber_coeffs_y[mask].mean())
            self.fiber_angle_circmean.append(
                circmean(self._fiber_angles[mask], high=np.pi, low=-np.pi)
            )
            self.fiber_angle_circstd.append(
                circstd(self._fiber_angles[mask], high=np.pi, low=-np.pi)
            )
            self.anatomical_tag_mode.append(
                mode(self._anatomical_structure_tags[mask]).mode
            )
        else:
            # Subdivide into four quadrants
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2

            self._subdivide(x_min, x_mid, y_min, y_mid, depth + 1)  # Bottom-left
            self._subdivide(x_mid, x_max, y_min, y_mid, depth + 1)  # Bottom-right
            self._subdivide(x_min, x_mid, y_mid, y_max, depth + 1)  # Top-left
            self._subdivide(x_mid, x_max, y_mid, y_max, depth + 1)  # Top-right
