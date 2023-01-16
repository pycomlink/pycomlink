import unittest
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

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

    def test_instant_to_minmax_data(self):
        rsl = np.concatenate(
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
                np.repeat(-40, 140),
            ]
        )
        rsl = np.reshape(np.repeat(rsl, 4), (-1, 4)).T

        tsl = np.reshape(np.repeat(10, 4 * 2160), (4, -1))

        a_lat = np.array([0, 0.01, 0.02, 0.5])
        a_lon = np.array([0, 0.01, 0.02, 0.5])
        b_lat = np.array([0.005, 0.02, 0.00, 0.512])
        b_lon = np.array([0.01, 0.02, 0.03, 0.51])

        ds_cml = xr.Dataset(
            data_vars=dict(tsl=(["cml_id", "time"], tsl),
                           rsl=(["cml_id", "time"], rsl)),
            coords=dict(
                time=pd.date_range(
                    "2020-01-01 00:00",
                    "2020-01-02 11:59",
                    freq="1min"),
                cml_id=["id0", "id1", "id2", "id3"],
                site_a_latitude=(["cml_id"], a_lat),
                site_a_longitude=(["cml_id"], a_lon),
                site_b_latitude=(["cml_id"], b_lat),
                site_b_longitude=(["cml_id"], b_lon),
                length=(
                ["cml_id"], spatial.haversine(a_lon, a_lat, b_lon, b_lat, ))
            ),
        )

        ds_cml_minmax = nb_wd.instanteanous_to_minmax_data(
            ds_cml,
            interval=15,
            timeperiod=24,
            min_hours=6)

        np.testing.assert_array_almost_equal(
            np.sum(ds_cml_minmax.sel(cml_id="id0").pmin.values),
            -7440.26763596)

    def test_wetdry_classification(self):
        rsl = np.concatenate(
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
                np.repeat(-40, 140),
            ]
        )
        rsl = np.reshape(np.repeat(rsl, 4), (-1, 4)).T

        tsl = np.reshape(np.repeat(10, 4 * 2160), (4, -1))

        a_lat = np.array([0, 0.01, 0.02, 0.5])
        a_lon = np.array([0, 0.01, 0.02, 0.5])
        b_lat = np.array([0.005, 0.02, 0.00, 0.512])
        b_lon = np.array([0.01, 0.02, 0.03, 0.51])

        ds_cml = xr.Dataset(
            data_vars=dict(tsl=(["cml_id", "time"], tsl),
                           rsl=(["cml_id", "time"], rsl)),
            coords=dict(
                time=pd.date_range(
                    "2020-01-01 00:00",
                    "2020-01-02 11:59",
                    freq="1min"),
                cml_id=["id0", "id1", "id2", "id3"],
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

        # create min max data
        ds_cml_minmax = nb_wd.instanteanous_to_minmax_data(
            ds_cml,
            interval=15,
            timeperiod=24,
            min_hours=6)

        ds_cml_minmax["wet"]=nb_wd.nearby_wetdry(
            ds_cml_dataset=ds_cml_minmax,
            ds_dist=ds_dist,
            r=15,
            thresh_median_P=-2.0,
            thresh_median_PL=-0.3,
            min_links=2,
        )

        np.testing.assert_array_almost_equal(
            ds_cml_minmax.wet.sel(cml_id="id1").values,
            np.array(
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            )

        )