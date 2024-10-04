import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from prior_fields.prior.dtypes import ArrayNx3


def add_3d_vectors_to_plot(
    pos: ArrayNx3,
    vecs: ArrayNx3,
    ax: axes3d.Axes3D,
    color="tab:blue",
    zorder=2,
    length=0.1,
    lw=1,
    **kwargs,
):
    """Add 3d vector field to plot axis.

    Parameters
    ----------
    pos : ArrayNx3
        (x, y, z) coordinates of vector positions.
    vecs : ArrayNx3
        (x, y, z) coordinates of vector directions.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
    color
    zorder
    length
    lw
    """
    ax.quiver(
        *[pos[:, i] for i in range(3)],
        *[vecs[:, i] for i in range(3)],
        length=length,
        lw=lw,
        color=color,
        zorder=zorder,
        **kwargs,
    )


def plot_vector_field(vecs: ArrayNx3, pos: ArrayNx3, length: float = 1.0) -> None:
    """Plot vector field in 3d space.

    Parameters
    ----------
    vecs : ArrayNx3
        (x, y, z) coordinates of vector directions.
    pos : ArrayNx3
        (x, y, z) coordinates of vector positions.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax = plt.gca()
    ax.set_aspect("equal")

    add_3d_vectors_to_plot(pos, vecs, ax, length=length, lw=0.5)

    plt.show()
