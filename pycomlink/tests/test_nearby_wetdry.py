import unittest
import numpy as np
import xarray as xr
import pandas as pd

import pycomlink.processing.wet_dry.nearby_wetdry as nb_wd
import pycomlink.spatial.helper as spatial


class Test_nearby_wetdry_approach(unittest.TestCase):
    def test_distance_calculation(self):
        # create custom CML xarray dataset with four CMLs
        # three close together one far away
        ds_cml = xr.Dataset(
            data_vars=dict(tsl=(
                ["cml_id", "time"],
                np.reshape(np.repeat(10, 4 * 2160), (4, -1)))),
            coords=dict(
                time=pd.date_range(
                    "2020-01-01 00:00",
                    "2020-01-02 11:59",
                    freq="1min"),
                cml_id=["id0", "id1", "id2", "id3"],
                site_a_latitude=(["cml_id"], np.array([0, 0.01, 0.02, 0.5])),
                site_a_longitude=(["cml_id"], np.array([0, 0.01, 0.02, 0.5])),
                site_b_latitude=(["cml_id"], np.array([0.005, 0.02, 0.00, 0.512])),
                site_b_longitude=(["cml_id"], np.array([0.01, 0.02, 0.03, 0.51])),
            ),
        )

        # test distance calculation
        ds_dist = nb_wd.calc_distance_between_cml_endpoints(
            cml_ids=ds_cml.cml_id.values,
            site_a_latitude=ds_cml.site_a_latitude,
            site_a_longitude=ds_cml.site_a_longitude,
            site_b_latitude=ds_cml.site_b_latitude,
            site_b_longitude=ds_cml.site_b_longitude,
        )

        np.testing.assert_array_almost_equal(
            ds_dist.a_to_all_a.values,
            np.array(
                [[0., 1.57154642, 3.14309282, 78.57682262],
                 [1.57154642, 0., 1.5715464, 77.0052762],
                 [3.14309282, 1.5715464, 0., 75.4337298],
                 [78.57682262, 77.0052762, 75.4337298, 0.]]
            )
        )


    def test_wetdry_classification(self):
        pmin = np.concatenate(
            [
                np.repeat(-40, 1460),
                np.linspace(-40, -60, 15),
                np.linspace(-60, -40, 25),
                np.repeat(-40, 60),
                np.linspace(-40, -50, 300),
                np.linspace(-50, -40, 60),
                np.repeat(-40, 60),
                np.linspace(-40, -70, 20),
                np.linspace(-70, -40, 20),
                np.repeat(-40, 30),
            ]
        )
        pmin = np.reshape(np.repeat(pmin, 5), (-1, 5)).T

        a_lat = [0, 0.01, 0.02, 0.5, -0.075]
        a_lon = [0, 0.01, 0.02, 0.5, -0.075]
        b_lat = [0.005, 0.02, 0.00, 0.512, 0.075]
        b_lon = [0.01, 0.02, 0.03, 0.51, 0.075]

        ds_cml = xr.Dataset(
            data_vars=dict(pmin=(["cml_id", "time"], pmin)),
            coords=dict(
                time=pd.date_range(
                    "2020-01-01 00:00",
                    freq="1min",
                    periods=2050),
                cml_id=["id0", "id1", "id2", "id3", "id4"],
                site_a_latitude=(["cml_id"], a_lat),
                site_a_longitude=(["cml_id"], a_lon),
                site_b_latitude=(["cml_id"], b_lat),
                site_b_longitude=(["cml_id"], b_lon),
                length=(
                    ["cml_id"],
                    spatial.haversine(a_lon, a_lat, b_lon, b_lat, ))
            ),
        )

        # get distance datatset
        ds_dist = nb_wd.calc_distance_between_cml_endpoints(
            cml_ids=ds_cml.cml_id.values,
            site_a_latitude=ds_cml.site_a_latitude,
            site_a_longitude=ds_cml.site_a_longitude,
            site_b_latitude=ds_cml.site_b_latitude,
            site_b_longitude=ds_cml.site_b_longitude,
        )

        (
            wet,
            F,
        ) = nb_wd.nearby_wetdry(
            pmin=ds_cml.pmin,
            ds_dist=ds_dist,
            radius=15,
            thresh_median_P=-2.0,
            thresh_median_PL=-0.3,
            min_links=3,
        )


        test_result_array = np.array(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
                 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 1.])

        np.testing.assert_array_almost_equal(
            wet.isel(cml_id=0).values[-100:-1],
            test_result_array
        )

        # test CML which is longer than r
        np.testing.assert_array_almost_equal(
            wet.sel(cml_id="id4").sum(),
            328
        )

        # test correct F
        np.testing.assert_array_almost_equal(
            F.sum(),
            7086.41961
        )