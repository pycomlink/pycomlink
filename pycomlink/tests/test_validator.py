import unittest
import numpy as np
import pycomlink as pycml


class TestIntersectWeights(unittest.TestCase):
    def test_with_simple_grid(self):
        x_grid, y_grid = np.meshgrid(np.arange(10), np.arange(10))

        x1, y1 = 0, 0
        x2, y2 = 0, 9

        intersec_weights = pycml.validation.validator.calc_intersect_weights(
            x1_line=x1,
            y1_line=y1,
            x2_line=x2,
            y2_line=y2,
            x_grid=x_grid,
            y_grid=y_grid)

        assert intersec_weights.sum() == 1.0

        np.testing.assert_array_almost_equal(
            intersec_weights,
            np.array(
                [[0.05555556, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0.11111111, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0.11111111, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0.11111111, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0.11111111, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0.11111111, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0.11111111, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0.11111111, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0.11111111, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0.05555556, 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))

        x1, y1 = 0, 0
        x2, y2 = 9, 9

        intersec_weights = pycml.validation.validator.calc_intersect_weights(
            x1_line=x1,
            y1_line=y1,
            x2_line=x2,
            y2_line=y2,
            x_grid=x_grid,
            y_grid=y_grid)

        assert intersec_weights.sum() == 1.0

        np.testing.assert_array_almost_equal(
            intersec_weights,
            np.array(
                [[0.05555556, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0.11111111, 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0.11111111, 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0.11111111, 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.11111111, 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.11111111, 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0.11111111, 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0.11111111, 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0.11111111, 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.05555556]]))
