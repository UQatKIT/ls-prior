import typer

from prior_fields.tensor.reader import collect_data_from_human_atrial_fiber_meshes


def main(file: str = "data/uacs_fibers_tags.npy"):
    data = collect_data_from_human_atrial_fiber_meshes()
    data.save(file)


if __name__ == "__main__":
    typer.run(main)
