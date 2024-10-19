import matplotlib.pyplot as plt
import numpy as np
from dolfin import Function, plot
from pyvista import Plotter, PolyData

from prior_fields.prior.dtypes import ArrayNx3


def get_poly_data(V: ArrayNx3, F: ArrayNx3) -> PolyData:
    return PolyData(V, np.hstack((np.full((F.shape[0], 1), 3), F)))


def plot_function(f: Function, show_mesh: bool = False, title: str = ""):
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
        plotter = Plotter()
        plotter.add_text(title)

        V = f.function_space().mesh().coordinates()
        F = f.function_space().mesh().cells()

        plotter.add_mesh(
            get_poly_data(V, F),
            scalars=f.compute_vertex_values(f.function_space().mesh()),
            show_edges=show_mesh,
        )

        plotter.show(window_size=(500, 500))
