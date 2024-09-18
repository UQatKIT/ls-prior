import matplotlib.pyplot as plt
import numpy as np
from dolfin import Function, plot
from vedo.dolfin import plot as vedo_plot

from prior_fields.prior.dtypes import Array1d, ArrayNx3


def plot_function(f: Function, show_mesh: bool = False, title: str = "", **kwargs):
    """Plot function defined on a finite element space.

    Parameters
    ----------
    f : dl.Function
        Function on a finite element space.
    show_mesh : bool  (optional)
        Plot mesh on top of the function.
    title : str  (optional)
        Plot title.
    """
    gdim = f.function_space().mesh().geometry().dim()

    if gdim == 2:
        c = plot(f)
        if show_mesh:
            plot(f.function_space().mesh())
        plt.colorbar(c)
        plt.title(title)
        plt.show()

    elif gdim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.set_title(title)
        im = vedo_plot(
            f,
            lw=show_mesh,
            style=1,
            axes={"xygrid": True, "yzgrid": True, "zxgrid": False},
            zoom=1,
            size=(475, 400),
            viewup="z",
            **kwargs,
        )
        ax.imshow(np.asarray(im))


def plot_vertex_values_on_surface(values: Array1d, pos: ArrayNx3) -> None:
    """Plot values at given positions in 3d space.

    Parameters
    ----------
    values : Array1d
        Values used to color the points of the scatter plot.
    pos : ArrayNx3
        (x, y, z) coordinates corresponding to the values.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax = plt.gca()
    ax.set_aspect("equal")

    c = ax.scatter(*[pos[:, i] for i in range(3)], c=values, s=1)  # type: ignore

    plt.colorbar(c)
    plt.show()
