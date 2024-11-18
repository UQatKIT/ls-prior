from pathlib import Path

import typer

from prior_fields.tensor.fiber_grid import compute_uac_fiber_grid

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    max_depth: int,
    point_threshold: int,
    path: Path = Path().cwd() / "data",
):
    compute_uac_fiber_grid(max_depth, point_threshold, path)


if __name__ == "__main__":
    app()
