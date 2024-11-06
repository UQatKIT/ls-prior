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
    labsize: int = 20,
    titlesize: int = 20,
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
        cbar = plt.colorbar(c)

        plt.gca().tick_params(labelsize=labsize)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(labsize)

        plt.title(title, fontsize=titlesize)

        if file:
            plt.tight_layout()
            plt.savefig(file)

        plt.show()

    elif gdim == 3:
        plotter = Plotter(window_size=(500, 500))
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

        plotter.show()


def plot_numpy_sample(
    s: Array1d,
    V: ArrayNx3,
    F: ArrayNx3,
    show_mesh: bool = False,
    title: str = "",
    file: Path | str | None = None,
    zoom: float = 1.25,
    clim: list[float] | None = None,
    scalar_bar_title: str | None = None,
) -> None:
    plotter = Plotter(window_size=(500, 500))
    plotter.add_text(title)

    plotter.add_mesh(
        get_poly_data(V, F),
        scalars=s,
        show_edges=show_mesh,
        clim=clim,
        below_color="purple" if clim and (clim[0] > s.min()) else None,
        above_color="orange" if clim and (clim[1] < s.max()) else None,
        scalar_bar_args=dict(
            title=scalar_bar_title, n_labels=2, position_x=0.3, position_y=0.06
        ),
    )
    plotter.add_axes(x_color="black", y_color="black", z_color="black")
    plotter.camera.zoom(zoom)

    if file:
        plotter.save_graphic(filename=file)

    plotter.show()
