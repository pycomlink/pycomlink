import unittest
from pycomlink.io.cmlh5_to_netcdf import read_cmlh5_file_to_xarray
import numpy as np
import pkg_resources


def get_test_data_path():
    return pkg_resources.resource_filename("pycomlink", "tests/test_data")


testdata_fn = str(get_test_data_path() + "/75_cmls_processed.h5")


class Testcmlh5tonetcdf(unittest.TestCase):
    def test_cmlh5_to_netcdf_fun(self):

        # convert to netcdf
        cml_xr_file = read_cmlh5_file_to_xarray(testdata_fn)

        # compare h5 data to netcdf data
        np.testing.assert_array_almost_equal(
            np.array([13., 14., 14., 15., 15., 15.]),
            cml_xr_file[19].isel(channel_id=0).tsl.values[1300:1306],
        )
        np.testing.assert_array_almost_equal(
            24.913,
            cml_xr_file[3].sel(channel_id="channel_1").frequency.values,
        )
