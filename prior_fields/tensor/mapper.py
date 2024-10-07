from typing import Literal

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
    Split UAC unit square into adaptive grid with width depending on the data density and
    for each component compute
    - mean coefficients of fibers in UAC basis,
    - circular mean and standard deviation of the fiber angle, and
    - mode of anatomical structure tag.

    Attributes
    ----------
    uac : ArrayNx2
        UACs at vertices.
    fiber_coeffs_x : Array1d
        Fiber coefficient for first UAC-based tangent space coordinate.
    fiber_coeffs_y : Array1d
        Fiber coefficient for second UAC-based tangent space coordinate.
    fiber_angles : Array1d
        Angle within (-pi, pi] representing the fiber orientation. 0 represents a fiber
        in the direction of no change in beta.
    anatomical_structure_tags : Array1d
        Tag for anatomical structure assignment.
    max_depth : int
        Maximum number of splits per cell.
    point_threshold : int
        Minimum number of points per cell.
    grid_x : list[list[float]]
        List of lower and upper boundaries of the cells in x-direction.
    grid_y : list[list[float]]
        List of lower and upper boundaries of the cells in y-direction.
    fiber_coeff_mean_x : list[float]
        Mean first fiber coefficients for each cell.
    fiber_coeff_mean_y : list[float]
        Mean second fiber coefficients for each cell.
    phi_circmean : list[float]
        Circular mean of fiber angles for each cell.
    phi_circstd : list[float]
        Circular standard deviation of fiber angles for each cell.
    tag_mode : list[int]
        Mode of anatomical structure tag for each cell.
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
    ):
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
            Angle within (-pi, pi] representing the fiber orientation. 0 represents a fiber
            in the direction of no change in beta.
        anatomical_structure_tags : Array1d
            Tag for anatomical structure assignment.
        max_depth : int, optional
            Maximum number of splits per cell, defaults to 5.
        point_threshold : int, optional
            Minimum number of points per cell, defaults to 100.
        """
        self.uac = uac
        self.fiber_coeffs_x = fiber_coeffs_x
        self.fiber_coeffs_y = fiber_coeffs_y
        self.fiber_angles = fiber_angles
        self.anatomical_structure_tags = anatomical_structure_tags
        self.max_depth = max_depth
        self.point_threshold = point_threshold

        self.grid_x: list[list[float]] = []
        self.grid_y: list[list[float]] = []

        self.fiber_coeff_mean_x: list[float] = []
        self.fiber_coeff_mean_y: list[float] = []
        self.phi_circmean: list[float] = []
        self.phi_circstd: list[float] = []
        self.tag_mode: list[int] = []

    def _subdivide(self, x_min, x_max, y_min, y_max, depth=0):
        if depth > self.max_depth:
            return

        # Select data points within this region
        mask = (
            (self.uac[:, 0] >= x_min)
            & (self.uac[:, 0] < x_max)
            & (self.uac[:, 1] >= y_min)
            & (self.uac[:, 1] < y_max)
        )
        data_count = mask.sum()

        if data_count < self.point_threshold or depth == self.max_depth:
            self.grid_x.append([x_min, x_max])
            self.grid_y.append([y_min, y_max])

            # Compute properties in this region
            self.fiber_coeff_mean_x.append(self.fiber_coeffs_x[mask].mean())
            self.fiber_coeff_mean_y.append(self.fiber_coeffs_y[mask].mean())
            self.phi_circmean.append(
                circmean(self.fiber_angles[mask], high=np.pi, low=-np.pi)
            )
            self.phi_circstd.append(
                circstd(self.fiber_angles[mask], high=np.pi, low=-np.pi)
            )
            self.tag_mode.append(mode(self.anatomical_structure_tags[mask]).mode)
        else:
            # Subdivide into four quadrants
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2

            self._subdivide(x_min, x_mid, y_min, y_mid, depth + 1)  # Bottom-left
            self._subdivide(x_mid, x_max, y_min, y_mid, depth + 1)  # Bottom-right
            self._subdivide(x_min, x_mid, y_mid, y_max, depth + 1)  # Top-left
            self._subdivide(x_mid, x_max, y_mid, y_max, depth + 1)  # Top-right

    def compute(self):
        # Start with the full unit square (0, 1) x (0, 1)
        self._subdivide(0, 1, 0, 1)

    def plot_results(self, c: Literal["tag", "mean", "std"]):
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_aspect("equal")

        grid_centers_x = [(x[0] + x[1]) / 2 for x in self.grid_x]
        grid_centers_y = [(y[0] + y[1]) / 2 for y in self.grid_y]

        cb = ax.scatter(
            grid_centers_x,
            grid_centers_y,
            c=(
                self.tag_mode
                if c == "tag"
                else (
                    self.phi_circmean
                    if c == "mean"
                    else self.phi_circstd
                    if c == "std"
                    else None
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
            self.fiber_coeff_mean_x,
            self.fiber_coeff_mean_y,
            angles="xy",
            scale_units="xy",
            scale=50,
            width=0.001,
        )

        plt.colorbar(cb)

        if c == "tag":
            plt.title(
                "Mean of fibers in UAC over 7 geometries with anatomical structures"
            )
        elif c == "mean":
            plt.title("Circular mean of fiber angle")
        elif c == "std":
            plt.title("Circular standard deviation of fiber angle")

        plt.show()
