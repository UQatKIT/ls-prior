import numpy as np
import pytest
from dolfin import FunctionSpace, UnitSquareMesh

from prior_fields.prior.converter import (
    function_to_numpy,
    numpy_to_function,
    numpy_to_vector,
    scale_mesh_to_unit_cube,
    vector_to_numpy,
)
from prior_fields.prior.prior import BiLaplacianPriorNumpyWrapper
from prior_fields.tensor.parameters import Geometry
from prior_fields.tensor.reader import (
    read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices,
)


@pytest.mark.parametrize("use_vertex_to_dof_map", [True, False])
def test_numpy_to_function_reverts_compute_vertex_values_on_unit_square(
    use_vertex_to_dof_map,
):
    mesh = UnitSquareMesh(64, 64)
    Vh = FunctionSpace(mesh, "CG", 1)

    s_numpy = np.random.standard_normal(mesh.num_vertices())
    s_function = numpy_to_function(
        s_numpy, Vh, use_vertex_to_dof_map=use_vertex_to_dof_map
    )

    np.testing.assert_array_equal(
        function_to_numpy(s_function, use_vertex_to_dof_map=use_vertex_to_dof_map),
        s_numpy,
    )


@pytest.fixture
def atlas_mesh():
    V_raw, F, _, _, _ = read_atrial_mesh_with_fibers_and_tags_mapped_to_vertices(
        Geometry(2)
    )
    V = scale_mesh_to_unit_cube(V_raw)
    return V, F


@pytest.mark.parametrize("use_vertex_to_dof_map", [True, False])
def test_numpy_to_function_reverts_compute_vertex_values_on_atrium(
    atlas_mesh, use_vertex_to_dof_map
):
    V, F = atlas_mesh
    prior = BiLaplacianPriorNumpyWrapper(V, F, sigma=0.2, ell=0.1, seed=1)
    Vh = prior._prior.Vh
    mesh = Vh.mesh()

    s_numpy = np.random.standard_normal(mesh.num_vertices())
    s_function = numpy_to_function(
        s_numpy, Vh, use_vertex_to_dof_map=use_vertex_to_dof_map
    )

    np.testing.assert_array_equal(
        function_to_numpy(s_function, use_vertex_to_dof_map=use_vertex_to_dof_map),
        s_numpy,
    )


@pytest.mark.parametrize("use_vertex_to_dof_map", [True, False])
def test_numpy_to_vector_reverts_vector_to_numpy(use_vertex_to_dof_map):
    mesh = UnitSquareMesh(64, 64)
    Vh = FunctionSpace(mesh, "CG", 1)

    s_numpy = np.random.standard_normal(mesh.num_vertices())
    s_vector = numpy_to_vector(s_numpy, Vh, use_vertex_to_dof_map=use_vertex_to_dof_map)

    np.testing.assert_array_equal(
        vector_to_numpy(s_vector, Vh, use_vertex_to_dof_map=use_vertex_to_dof_map),
        s_numpy,
    )
