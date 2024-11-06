from pyvista import Light, Plotter


def initialize_vector_field_plotter(
    poly_data,
    title: str = "",
    zoom: float = 1.25,
    add_axes: bool = True,
    window_size: tuple[int, int] = (500, 500),
) -> Plotter:
    """Initialize pyvista.Plotter() with defaults for vector plots."""
    plotter = Plotter(lighting=None, window_size=window_size)
    plotter.add_text(title)
    plotter.add_mesh(poly_data, color="white")
    if add_axes:
        plotter.add_axes(x_color="black", y_color="black", z_color="black")
    plotter.add_light(
        Light(
            position=(0.3, 1.0, 1.0),
            focal_point=(0, 0, 0),
            color=[1.0, 0.95, 0.95, 1.0],
            intensity=1,
        )
    )
    plotter.camera.zoom(zoom)

    return plotter
