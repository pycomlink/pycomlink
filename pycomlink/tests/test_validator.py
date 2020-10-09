import unittest
import pytest
import numpy as np
from collections import namedtuple
import pycomlink as pycml


class TestIntersectWeights(unittest.TestCase):
    def test_with_simple_grid(self):
        x_grid, y_grid = np.meshgrid(np.arange(10), np.arange(10))

        x1, y1 = 0, 0
        x2, y2 = 0, 9

        intersec_weights = pycml.validation.validator.calc_intersect_weights(
            x1_line=x1, y1_line=y1, x2_line=x2, y2_line=y2, x_grid=x_grid, y_grid=y_grid
        )

        assert intersec_weights.sum() == 1.0

        np.testing.assert_array_almost_equal(
            intersec_weights,
            np.array(
                [
                    [0.05555556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.05555556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        x1, y1 = 0, 0
        x2, y2 = 9, 9

        intersec_weights = pycml.validation.validator.calc_intersect_weights(
            x1_line=x1, y1_line=y1, x2_line=x2, y2_line=y2, x_grid=x_grid, y_grid=y_grid
        )

        assert intersec_weights.sum() == 1.0

        np.testing.assert_array_almost_equal(
            intersec_weights,
            np.array(
                [
                    [0.05555556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05555556],
                ]
            ),
        )

    def test_with_simple_grid_location_lower_left(self):
        x_grid, y_grid = np.meshgrid(np.arange(10), np.arange(10))

        x1, y1 = 0.5, 0
        x2, y2 = 0.5, 9

        intersec_weights = pycml.validation.validator.calc_intersect_weights(
            x1_line=x1,
            y1_line=y1,
            x2_line=x2,
            y2_line=y2,
            x_grid=x_grid,
            y_grid=y_grid,
            grid_point_location="lower_left",
        )

        assert intersec_weights.sum() == 1.0

        np.testing.assert_array_almost_equal(
            intersec_weights,
            np.array(
                [
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        x1, y1 = 0.5, 0.5
        x2, y2 = 9.5, 9.5

        intersec_weights = pycml.validation.validator.calc_intersect_weights(
            x1_line=x1,
            y1_line=y1,
            x2_line=x2,
            y2_line=y2,
            x_grid=x_grid,
            y_grid=y_grid,
            grid_point_location="lower_left",
        )

        assert intersec_weights.sum() == 1.0

        np.testing.assert_array_almost_equal(
            intersec_weights,
            np.array(
                [
                    [0.05555556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05555556],
                ]
            ),
        )


class TestCalcGridCorners(unittest.TestCase):
    def test_location_at_grid_center(self):
        x_grid, y_grid = np.meshgrid(np.arange(10, 20, 1), np.arange(50, 70, 1))
        grid = np.stack([x_grid, y_grid], axis=2)

        result = pycml.validation.validator._calc_grid_corners_for_center_location(
            grid=grid
        )

        GridCorners = namedtuple(
            "GridCorners", ["ur_grid", "ul_grid", "lr_grid", "ll_grid"]
        )
        expected = GridCorners(
            ur_grid=np.stack([x_grid + 0.5, y_grid + 0.5], axis=2),
            ul_grid=np.stack([x_grid - 0.5, y_grid + 0.5], axis=2),
            lr_grid=np.stack([x_grid + 0.5, y_grid - 0.5], axis=2),
            ll_grid=np.stack([x_grid - 0.5, y_grid - 0.5], axis=2),
        )

        np.testing.assert_almost_equal(result.ur_grid, expected.ur_grid)
        np.testing.assert_almost_equal(result.ul_grid, expected.ul_grid)
        np.testing.assert_almost_equal(result.lr_grid, expected.lr_grid)
        np.testing.assert_almost_equal(result.ll_grid, expected.ll_grid)

    def test_location_at_lower_left(self):
        x_grid, y_grid = np.meshgrid(np.arange(10, 20, 1), np.arange(50, 70, 1))
        grid = np.stack([x_grid, y_grid], axis=2)

        result = pycml.validation.validator._calc_grid_corners_for_lower_left_location(
            grid=grid
        )

        GridCorners = namedtuple(
            "GridCorners", ["ur_grid", "ul_grid", "lr_grid", "ll_grid"]
        )
        expected = GridCorners(
            ur_grid=np.stack([x_grid + 1.0, y_grid + 1.0], axis=2),
            ul_grid=np.stack([x_grid, y_grid + 1.0], axis=2),
            lr_grid=np.stack([x_grid + 1.0, y_grid], axis=2),
            ll_grid=np.stack([x_grid, y_grid], axis=2),
        )

        np.testing.assert_almost_equal(result.ur_grid, expected.ur_grid)
        np.testing.assert_almost_equal(result.ul_grid, expected.ul_grid)
        np.testing.assert_almost_equal(result.lr_grid, expected.lr_grid)
        np.testing.assert_almost_equal(result.ll_grid, expected.ll_grid)

    def test_location_at_lower_left_descending_x_error(self):
        x_grid, y_grid = np.meshgrid(np.arange(20, 10, -1), np.arange(50, 70, 1))
        grid = np.stack([x_grid, y_grid], axis=2)

        with pytest.raises(ValueError, match="x values must be ascending along axis 1"):
            pycml.validation.validator._calc_grid_corners_for_lower_left_location(
                grid=grid
            )

    def test_location_at_lower_left_descending_y_error(self):
        x_grid, y_grid = np.meshgrid(np.arange(10, 20, 1), np.arange(70, 50, -1))
        grid = np.stack([x_grid, y_grid], axis=2)

        with pytest.raises(ValueError, match="y values must be ascending along axis 0"):
            pycml.validation.validator._calc_grid_corners_for_lower_left_location(
                grid=grid
            )
