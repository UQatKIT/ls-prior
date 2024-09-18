import numpy as np

from prior_fields.tensor.data_cleaning import remove_vertex_from_mesh


def test_remove_vertex_from_mesh():
    idx = 0
    V_inp = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]])
    fibers_inp = V_inp.copy()
    F_inp = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 1], [1, 3, 4]])

    V_exp = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]])
    fibers_exp = V_exp.copy()
    F_exp = np.array([[2, 3, 0], [0, 1, 2]])

    V_out, F_out, fibers_out = remove_vertex_from_mesh(idx, V_inp, F_inp, fibers_inp)

    np.testing.assert_array_equal(V_out, V_exp)
    np.testing.assert_array_equal(F_out, F_exp)
    np.testing.assert_array_equal(fibers_out, fibers_exp)
