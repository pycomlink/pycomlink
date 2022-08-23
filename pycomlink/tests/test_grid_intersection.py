import unittest
import pytest
import numpy as np
from collections import namedtuple
import sparse
import xarray as xr
import pycomlink as pycml


class TestSparseIntersectWeights(unittest.TestCase):
    def test_creation_of_xarray_dataarray(self):

        x_grid, y_grid = np.meshgrid(np.arange(10), np.arange(12))

        x1_list = [0, 0]
        y1_list = [0, 0]
        x2_list = [0, 9]
        y2_list = [9, 9]
        cml_id_list = ["abc1", "cde2"]

        da_intersect_weights = pycml.spatial.grid_intersection.calc_sparse_intersect_weights_for_several_cmls(
            x1_line=x1_list,
            y1_line=y1_list,
            x2_line=x2_list,
            y2_line=y2_list,
            cml_id=cml_id_list,
            x_grid=x_grid,
            y_grid=y_grid,
        )

        for x1, y1, x2, y2, cml_id in zip(
            x1_list, y1_list, x2_list, y2_list, cml_id_list
        ):
            expected = pycml.spatial.grid_intersection.calc_intersect_weights(
                x1_line=x1,
                y1_line=y1,
                x2_line=x2,
                y2_line=y2,
                x_grid=x_grid,
                y_grid=y_grid,
            )
            np.testing.assert_array_almost_equal(
                expected, da_intersect_weights.sel(cml_id=cml_id).to_numpy()
            )


class TestIntersectWeights(unittest.TestCase):
    def test_with_simple_grid(self):
        x_grid, y_grid = np.meshgrid(np.arange(10), np.arange(10))

        x1, y1 = 0, 0
        x2, y2 = 0, 9

        intersec_weights = pycml.spatial.grid_intersection.calc_intersect_weights(
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

        intersec_weights = pycml.spatial.grid_intersection.calc_intersect_weights(
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

        intersec_weights = pycml.spatial.grid_intersection.calc_intersect_weights(
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

        intersec_weights = pycml.spatial.grid_intersection.calc_intersect_weights(
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

        result = pycml.spatial.grid_intersection._calc_grid_corners_for_center_location(
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

        result = (
            pycml.spatial.grid_intersection._calc_grid_corners_for_lower_left_location(
                grid=grid
            )
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
            pycml.spatial.grid_intersection._calc_grid_corners_for_lower_left_location(
                grid=grid
            )

    def test_location_at_lower_left_descending_y_error(self):
        x_grid, y_grid = np.meshgrid(np.arange(10, 20, 1), np.arange(70, 50, -1))
        grid = np.stack([x_grid, y_grid], axis=2)

        with pytest.raises(ValueError, match="y values must be ascending along axis 0"):
            pycml.spatial.grid_intersection._calc_grid_corners_for_lower_left_location(
                grid=grid
            )


def get_grid_intersect_ts_test_data():
    grid_data = np.tile(
        np.expand_dims(np.arange(10, dtype="float"), axis=[1, 2]), (1, 4, 4)
    )
    grid_data[0, 0, 1] = np.nan
    # fmt: off
    intersect_weights = np.array(
        [
            [[0.25, 0, 0, 0],
             [0.25, 0, 0, 0],
             [0.25, 0, 0, 0],
             [0.25, 0, 0, 0]],
            [[0, 0.25, 0.25, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
        ]
    )
    # fmt: on
    expected = np.array(
        [
            [0.0, np.nan],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
            [6.0, 3.0],
            [7.0, 3.5],
            [8.0, 4.0],
            [9.0, 4.5],
        ]
    )
    return grid_data, intersect_weights, expected


class TestGetGridTimeseries(unittest.TestCase):
    def test_numpy_grid_numpy_weights(self):
        grid_data, intersect_weights, expected = get_grid_intersect_ts_test_data()

        result = pycml.spatial.grid_intersection.get_grid_time_series_at_intersections(
            grid_data=grid_data,
            intersect_weights=intersect_weights,
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_dataarray_grid_numpy_weights(self):
        grid_data, intersect_weights, expected = get_grid_intersect_ts_test_data()
        time = np.arange(np.datetime64("2017-01-01"), np.datetime64("2017-01-11"))
        da_grid_data = xr.DataArray(
            data=grid_data,
            dims=("time", "y", "x"),
            coords={"time": time},
        )

        result = pycml.spatial.grid_intersection.get_grid_time_series_at_intersections(
            grid_data=da_grid_data,
            intersect_weights=intersect_weights,
        )
        np.testing.assert_array_almost_equal(result.data, expected)

        np.testing.assert_array_equal(result.time.values, time)
        assert result.dims == ("time", "cml_id")

    def test_numpy_grid_dataarray_weights(self):
        grid_data, intersect_weights, expected = get_grid_intersect_ts_test_data()
        cml_ids = ["cml_1", "cml_2"]
        da_intersect_weights = xr.DataArray(
            data=intersect_weights,
            dims=("cml_id", "y", "x"),
            coords={"cml_id": cml_ids},
        )

        result = pycml.spatial.grid_intersection.get_grid_time_series_at_intersections(
            grid_data=grid_data,
            intersect_weights=da_intersect_weights,
        )
        np.testing.assert_array_almost_equal(result.data, expected)

        np.testing.assert_array_equal(result.cml_id.values, cml_ids)
        assert result.dims == ("time", "cml_id")


    def test_dataarray_grid_dataarray_weights(self):
        grid_data, intersect_weights, expected = get_grid_intersect_ts_test_data()

        time = np.arange(np.datetime64("2017-01-01"), np.datetime64("2017-01-11"))
        da_grid_data = xr.DataArray(
            data=grid_data,
            dims=("time", "y", "x"),
            coords={"time": time},
        )

        cml_ids = ["cml_1", "cml_2"]
        da_intersect_weights = xr.DataArray(
            data=intersect_weights,
            dims=("cml_id", "y", "x"),
            coords={"cml_id": cml_ids},
        )

        result = pycml.spatial.grid_intersection.get_grid_time_series_at_intersections(
            grid_data=da_grid_data,
            intersect_weights=da_intersect_weights,
        )
        np.testing.assert_array_almost_equal(result.data, expected)

        np.testing.assert_array_equal(result.cml_id.values, cml_ids)
        np.testing.assert_array_equal(result.time.values, time)

        assert result.dims == ("time", "cml_id")

