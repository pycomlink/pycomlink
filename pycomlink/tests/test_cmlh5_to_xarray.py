import unittest
from pycomlink.io.cmlh5_to_xarray import read_cmlh5_file_to_xarray
import numpy as np
import pkg_resources


def get_test_data_path():
    return pkg_resources.resource_filename("pycomlink", "tests/test_data")


testdata_fn = str(get_test_data_path() + "/75_cmls_processed.h5")


class Testcmlh5tonetcdf(unittest.TestCase):
    def test_cmlh5_to_xarray_fun(self):

        # read to xarray
        cml_xr_file = read_cmlh5_file_to_xarray(testdata_fn)

        # check selected parts of time series
        np.testing.assert_array_almost_equal(
            np.array([13.0, 14.0, 14.0, 15.0, 15.0, 15.0]),
            cml_xr_file[19].isel(channel_id=0).tsl.values[1300:1306],
        )
        np.testing.assert_almost_equal(
            np.array([-47.6, -47.9, -47.9, -47.6, -47.9, -47.9]),
            cml_xr_file[11].isel(channel_id=1).rsl.values[1100:1106],
        )
        np.testing.assert_almost_equal(
            np.array([-47.3, -47.3, -47.0, -47.3, -47.0, -47.3]),
            cml_xr_file[52].isel(channel_id=1).rsl.values[1100:1106],
        )

        # Test some metadata
        np.testing.assert_array_almost_equal(
            24.913,
            cml_xr_file[3].sel(channel_id="channel_1").frequency.values,
        )
        np.testing.assert_array_almost_equal(
            18.085,
            cml_xr_file[42].sel(channel_id="channel_1").frequency.values,
        )
        np.testing.assert_array_almost_equal(
            19.095,
            cml_xr_file[42].sel(channel_id="channel_2").frequency.values,
        )

        # test correction conversion of timestamps
        np.testing.assert_array_equal(
            np.array(
                [
                    "2017-06-28T18:46:10.226388992",
                    "2017-06-28T18:47:10.222681088",
                    "2017-06-28T18:48:10.229869056",
                    "2017-06-28T18:49:10.166190080",
                    "2017-06-28T18:51:10.229346048",
                    "2017-06-28T18:52:10.227350016",
                ],
                dtype="datetime64[ns]",
            ),
            cml_xr_file[13].isel(channel_id=1).time.values[1100:1106],
        )
        np.testing.assert_array_equal(
            np.array(
                [
                    "2017-06-28T19:30:08.294352896",
                    "2017-06-28T19:31:08.274571008",
                    "2017-06-28T19:32:08.295169024",
                    "2017-06-28T19:34:08.278542080",
                    "2017-06-28T19:35:08.276589056",
                    "2017-06-28T19:36:08.277197056",
                ],
                dtype="datetime64[ns]",
            ),
            cml_xr_file[52].isel(channel_id=1).time.values[1100:1106],
        )

        np.testing.assert_array_equal(
            np.array(
                [51.0909, 51.1271, 50.8026, 50.738299999999995]
            ),
            np.array(
                [
                    cml_xr_file[12].site_a_longitude.values,
                    cml_xr_file[12].site_b_longitude.values,
                    cml_xr_file[12].site_a_latitude.values,
                    cml_xr_file[12].site_b_latitude.values,
                ],
            )
        )

