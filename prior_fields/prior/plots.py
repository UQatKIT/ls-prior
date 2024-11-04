from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dolfin import Function, plot
from pyvista import Plotter, PolyData

from prior_fields.prior.dtypes import Array1d, ArrayNx3


def get_poly_data(V: ArrayNx3, F: ArrayNx3) -> PolyData:
    return PolyData(V, np.hstack((np.full((F.shape[0], 1), 3), F)))


def plot_function(
    f: Function,
    show_mesh: bool = False,
    title: str = "",
    file: Path | str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Plot function defined on a finite element space.

    Parameters
    ----------
    f : dl.Function
        Function on a finite element space.
    show_mesh : bool  (optional)
        Plot mesh on top of the function.
    title : str  (optional)
        Plot title.
    file : Path | str | None (optional)
        If specified, plot is saved to this destination.
    """
    gdim = f.function_space().mesh().geometry().dim()

    if gdim == 2:
        c = plot(f, vmin=vmin, vmax=vmax)
        if show_mesh:
            plot(f.function_space().mesh())
        plt.colorbar(c)
        plt.title(title)

        if file:
            plt.tight_layout()
            plt.savefig(file)

        plt.show()

    elif gdim == 3:
        plotter = Plotter()
        plotter.add_text(title)

        plotter.add_mesh(
            get_poly_data(
                f.function_space().mesh().coordinates(),
                f.function_space().mesh().cells(),
            ),
            scalars=f.compute_vertex_values(f.function_space().mesh()),
            show_edges=show_mesh,
        )
        plotter.add_axes(x_color="black", y_color="black", z_color="black")
        plotter.camera.zoom(1.25)

        if file:
            plotter.save_graphic(filename=file)

        plotter.show(window_size=(500, 500))


def plot_sample_from_numpy_wrapper(
    s: Array1d,
    V: ArrayNx3,
    F: ArrayNx3,
    show_mesh: bool = False,
    title: str = "",
    file: Path | str | None = None,
) -> None:
    plotter = Plotter()
    plotter.add_text(title)

    plotter.add_mesh(get_poly_data(V, F), scalars=s, show_edges=show_mesh)
    plotter.add_axes(x_color="black", y_color="black", z_color="black")
    plotter.camera.zoom(1.25)

    if file:
        plotter.save_graphic(filename=file)

    plotter.show(window_size=(500, 500))
