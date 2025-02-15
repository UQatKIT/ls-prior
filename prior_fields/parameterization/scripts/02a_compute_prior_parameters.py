from pathlib import Path

import numpy as np
import typer

from prior_fields.parameterization.parameters import (
    Geometry,
    PriorParameters,
    get_fiber_parameters_from_uac_data,
)
from prior_fields.parameterization.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)
from prior_fields.prior.converter import scale_mesh_to_unit_cube

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(geometry: int, k: int = 20, path: Path = Path().cwd() / "data/parameters/"):
    V_raw, F, uac, _, _ = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(
        Geometry(geometry)
    )
    V = scale_mesh_to_unit_cube(V_raw)
    mean, sigma, _ = get_fiber_parameters_from_uac_data(V, F, uac, k=k)
    ell = 0.05 * np.ones_like(sigma)

    PriorParameters(Geometry(geometry), mean, sigma, ell).save(path)


if __name__ == "__main__":
    app()
