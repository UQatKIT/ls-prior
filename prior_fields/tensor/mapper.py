import re
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import circmean, circstd, mode

from prior_fields.prior.dtypes import Array1d, ArrayNx2, ArrayNx3
from prior_fields.tensor.transformer import normalize


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
    fibers_x, fibers_y = get_coefficients(normalize(fibers), x, y)

    fibers_mapped = fibers_x[:, np.newaxis] * x + fibers_y[:, np.newaxis] * y

    return normalize(fibers_mapped)


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
    grid_x : Array1d
        Array of lower and upper boundaries of the cells in x-direction.
    grid_y : Array1d
        Array of lower and upper boundaries of the cells in y-direction.
    fiber_coeff_x_mean : Array1d
        Mean first fiber coefficients for each cell.
    fiber_coeff_y_mean : Array1d
        Mean second fiber coefficients for each cell.
    fiber_angle_circmean : Array1d
        Circular mean of fiber angles in (-pi, pi] for each cell.
    fiber_angle_circstd : Array1d
        Circular standard deviation of fiber angles for each cell.
    anatomical_tag_mode : Array1d
        Mode of anatomical structure tag for each cell.
    max_depth : int
        Maximum number of splits per cell, defaults to 5.
    point_threshold : int
        Minimum number of points per cell, defaults to 100.
    """

    def __init__(
        self,
        grid_x: ArrayNx2,
        grid_y: ArrayNx2,
        fiber_coeff_x_mean: Array1d,
        fiber_coeff_y_mean: Array1d,
        fiber_angle_circmean: Array1d,
        fiber_angle_circstd: Array1d,
        anatomical_tag_mode: Array1d,
        max_depth: int = 5,
        point_threshold: int = 100,
    ) -> None:
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.fiber_coeff_x_mean = fiber_coeff_x_mean
        self.fiber_coeff_y_mean = fiber_coeff_y_mean
        self.fiber_angle_circmean = fiber_angle_circmean
        self.fiber_angle_circstd = fiber_angle_circstd
        self.anatomical_tag_mode = anatomical_tag_mode
        self.max_depth = max_depth
        self.point_threshold = point_threshold

    @classmethod
    def read_from_binary_file(cls, path: str):
        grid = np.load(path)

        match_max_depth = re.search(r"max_depth(\d+)_", path)
        max_depth = int(match_max_depth.group(1)) if match_max_depth is not None else 5
        match_point_threshold = re.search(r"point_threshold(\d+).npy", path)
        point_threshold = (
            int(match_point_threshold.group(1))
            if match_point_threshold is not None
            else 100
        )

        return FiberGrid(
            grid_x=grid[:, 0:2],
            grid_y=grid[:, 2:4],
            fiber_coeff_x_mean=grid[:, 4],
            fiber_coeff_y_mean=grid[:, 5],
            fiber_angle_circmean=grid[:, 6],
            fiber_angle_circstd=grid[:, 7],
            anatomical_tag_mode=grid[:, 8],
            max_depth=max_depth,
            point_threshold=point_threshold,
        )

    def plot(self, color: Literal["tag", "mean", "std"]) -> None:
        """
        Plot the adaptive grid with mean fiber vector in each cell. The cells are colored
        according to the fiber properties or anatomical region.

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
            plt.tight_layout()
            plt.savefig("figures/uac_fibers_with_tag.svg")
        elif color == "mean":
            plt.title("Circular mean of fiber angle")
            plt.tight_layout()
            plt.savefig("figures/uac_fibers_with_circmean.svg")
        elif color == "std":
            plt.title("Circular standard deviation of fiber angle")
            plt.tight_layout()
            plt.savefig("figures/uac_fibers_with_circstd.svg")

        plt.show()

    def save(self, filename_prefix: str = "fiber_grid") -> None:
        """Write fiber grid to binary file.

        Parameters
        ----------
        filename_prefix : str, optional
            Prefix of the filename, defaults to 'fiber_grid'.
            The suffix is fixed to contain max_depth and point_threshold which are used to
            initialize a FiberGrid from the file.
        """
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


class FiberGridComputer:
    """
    Computes a fiber grid and the fiber properties/anatomical tags for each cell from the
    UAC coordinates and fiber/tag data for the vertices of left atrial meshes.
    """

    def __init__(
        self,
        uac: ArrayNx2,
        fiber_coeffs_x: Array1d,
        fiber_coeffs_y: Array1d,
        fiber_angles: Array1d,
        anatomical_structure_tags: Array1d,
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

    def get_fiber_grid(self) -> FiberGrid:
        return FiberGrid(
            grid_x=np.array(self.grid_x),
            grid_y=np.array(self.grid_y),
            fiber_coeff_x_mean=np.array(self.fiber_coeff_x_mean),
            fiber_coeff_y_mean=np.array(self.fiber_coeff_y_mean),
            fiber_angle_circmean=np.array(self.fiber_angle_circmean),
            fiber_angle_circstd=np.array(self.fiber_angle_circstd),
            anatomical_tag_mode=np.array(self.anatomical_tag_mode),
            max_depth=self.max_depth,
            point_threshold=self.point_threshold,
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
                circmean(self._fiber_angles[mask], low=-np.pi / 2, high=np.pi / 2)
            )
            self.fiber_angle_circstd.append(
                circstd(self._fiber_angles[mask], low=-np.pi / 2, high=np.pi / 2)
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
