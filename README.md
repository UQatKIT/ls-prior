# prior-fields-fenics

This Python library includes three major purposes:

1. Implementation of a bi-Laplacian prior (special case of a Whittle-Matérn field) on triangular meshes of 2D (curved) surfaces based on the FEniCS project.
2. Data-informed parameterization of this prior including transformations to use the prior for vector-valued random fields, particularly applied to atrial fiber orientations.
3. Combination of a fiber field with conduction velocity information into a conduction velocity tensor.

The background of this project is an infinite-dimensional formulation of the Bayesian inverse problem (BIP) of cardiac electrophysiology. The inverse problem is based on a hyperbolic eikonal-type model, and aims to reconstruct the conduction velocity (CV) tensor from discrete measurements of the local activation time (LAT). The contribution of this project is to provide the prior distribution for the BIP. We do this in a data-informed manner.


### Example: Fiber Field Prior Samples vs. Mean Fiber Field
![Fiber Field Prior vs. Mean Fibers Field](./figures/vector_fields/samples_vs_mean_visualization.gif)


## Installation

To run the project locally, clone this repository, `cd` into the project folder and run

```console
pixi install
```
This installs the environment from the [lock file](./pixi.lock). We currently support `linux-64`, `osx-arm64`, and `osx-64`. For further details on how to install and use pixi, see also the [pixi documentation](https://pixi.sh/latest/).

## Usage
Besides examples on how to use use the bi-Laplacian prior implementation, we provide the scripts for reconstructing our parameter fields.

In the [`examples/`](./examples/) folder you can find mainly four scripts, which introduce the use of the prior class starting from the unit square ([`01_example_unit_square.py`](./examples/01_example_unit_square.py)), moving on to curved surfaces in the unit sphere example ([`02_example_unit_sphere.py`](./examples/02_example_unit_sphere.py)), and finally turning to a geometry of the left atrium ([`03_example_atrium.py`](./examples/03_example_atrium.py)) and the assembly of the corresponding CV tensor ([`04_example_conduction_velocity_tensor.py`](./examples/04_example_conduction_velocity_tensor.py)).

In [`examples/03_example_atrium.py`](./examples/03_example_atrium.py), we also incorporate data-informed parameters (mean, pointwise standard deviation). These are constructed based on the human atrial fiber data included in [`data/LGE-MRI-based/`](./data/LGE-MRI-based/). In order to compute these parameters from scratch, see the scripts in [`prior_fields/tensor/scripts/`](./prior_fields/tensor/scripts/).

In particular,
```console
pixi run -- python prior_fields/tensor/scripts/01_collect_data_on_uac_level.py 
```
collects the fiber information from all seven geometries in the universal atrial coordinate (UAC) unit square and saves them to [`data/uacs_fibers_tags.npy`](./data/uacs_fibers_tags.npy). They are subsequently used when running
```console
pixi run -- python prior_fields/tensor/scripts/02a_compute_prior_parameters.py 3
```
to compute mean and standard deviation for geometry 3 and save them to [`data/parameters/params_3.npy`](./data/parameters/params_3.npy). You can replace the `3` by any geometry tag (1, ..., 7) to construct parameter fields for all of the geometries. Alternatively, you can use this script with your own atrial geometry, as long as the corresponding data set includes the UACs and is saved in the same format at the same path as the human atrial data sets used in this project. For more information on the data requirements, see [`data/README.md`](./data/README.md).

An alternative to this parameterization, which I shortly discuss in my thesis, can be found in [`02b_compute_uac_fiber_grid.py`](./prior_fields/tensor/scripts/02b_compute_uac_fiber_grid.py). We there compute the parameters independent of the target geometry for adaptive grid cells over the UAC unit square.

In [`03_compute_stats.py`](./prior_fields/tensor/scripts/03_compute_stats.py), we evaluate the computation times of the different components. We highlight the efficiency of the online computations (sampling, ...) related to the prior implementation, compared to the rather computationally expensive offline part (parameter computation, prior initialization, ...).

The plots included in the thesis are created in [`04a_prior_plots_for_report.py`](./prior_fields/tensor/scripts/04a_prior_plots_for_report.py) and [`04b_other_plots_for_report.py`](./prior_fields/tensor/scripts/04b_other_plots_for_report.py).

For information on our data sources, see [`data/README.md`](./data/README.md).

## Project status
This project was developed as part of a master's thesis. Therefore, it is not regularly maintained anymore. However, if you have any concerns or remarks, feel free to reach out by creating an issue.
