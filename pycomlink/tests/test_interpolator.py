import unittest
import numpy as np
import pandas as pd
import pycomlink as pycml
from pycomlink.tests.utils import load_processed_cml_list


class TestIdwKdtreeInterpolator(unittest.TestCase):
    def test_without_nans(self):
        pass

    def test_with_nans(self):
        pass


class TestOrdiniaryKrigingInterpolator(unittest.TestCase):
    def test_without_nans(self):
        pass

    def test_with_nans(self):
        pass


class TestComlinkGridInterpolator(unittest.TestCase):
    def test_default_idw(self):
        cml_list = load_processed_cml_list()

        interpolator = pycml.spatial.interpolator.ComlinkGridInterpolator(
            cml_list=cml_list,
            variable='R',
            resolution=0.05,
            interpolator=pycml.spatial.interpolator.IdwKdtreeInterpolator())

        ds = interpolator.loop_over_time()

        assert ds.R.isel(time=0).values.shape == (16, 23)

        np.testing.assert_array_almost_equal(
            ds.R.isel(time=10).values[1:4, 1:4],
            np.array([[0., 0., 0.],
                      [0.02679148, 0., 0.],
                      [0.17902197, 0.10871572, 0.]]))

        np.testing.assert_array_almost_equal(
            ds.R.isel(time=20).values[1:4, 1:4],
            np.array([[0.01852385, 0.08876891, 0.14781145],
                      [0.05900791, 0.15852366, 0.14716676],
                      [0.16692699, 0.16040398, 0.11649888]]))

        zgrid_i = interpolator.interpolate_for_i(3)
        np.testing.assert_array_almost_equal(zgrid_i[1:4, 1:4], np.array(
            [[0.04484675, 0.12339762, 0.1185833],
             [0.07857356, 0.26151277, 0.13641106],
             [0.2014109, 0.19460449, 0.08598332]]))

    def test_default_idw_aggregated_to_new_index(self):
        t_str_list = [
            '2017-06-27 21:50:00',
            '2017-06-27 22:50:00',
            '2017-06-27 23:50:00',
            '2017-06-28 00:50:00',
            '2017-06-28 01:50:00',
            '2017-06-28 02:50:00',
            '2017-06-28 03:50:00',
            '2017-06-28 04:50:00',
            '2017-06-28 05:50:00',
            '2017-06-28 06:50:00',
            '2017-06-28 07:50:00',
            '2017-06-28 08:50:00',
            '2017-06-28 09:50:00']

        cml_list = load_processed_cml_list()

        interpolator = pycml.spatial.interpolator.ComlinkGridInterpolator(
            cml_list=cml_list,
            variable='R',
            resample_to_new_index=pd.to_datetime(t_str_list),
            resolution=0.05,
            interpolator=pycml.spatial.interpolator.IdwKdtreeInterpolator())

        ds = interpolator.loop_over_time()

        assert ds.R.isel(time=0).values.shape == (16, 23)

        assert ds.time[0].values == np.datetime64('2017-06-28T00:50:00')
        assert ds.time[-1].values == np.datetime64('2017-06-28T09:50:00')

        np.testing.assert_array_almost_equal(
            ds.R.isel(time=9).values[1:4, 1:4],
            np.array([[0.01552989,  0.09481763,  0.15710547],
                      [0.035198,  0.06016459,  0.19047063],
                      [0.06757572,  0.09647522,  0.08332671]]))

        np.testing.assert_array_almost_equal(
            ds.R.isel(time=6).values[-3:, -3:],
            np.array([[2.68456511,  2.3083342 ,  1.99627359],
                      [3.30853954,  2.87754027,  2.48889028],
                      [3.83424582,  3.28489834,  3.00832896]]))

    def test_default_kriging(self):
        cml_list = load_processed_cml_list()

        # TODO: Use 'C' backend when available via pip installed pykirge
        interpolator = pycml.spatial.interpolator.ComlinkGridInterpolator(
            cml_list=cml_list,
            variable='R',
            resolution=0.05,
            interpolator=(pycml.spatial.interpolator.
                          OrdinaryKrigingInterpolator(backend='loop')))

        ds = interpolator.loop_over_time()

        assert ds.R.isel(time=0).values.shape == (16, 23)

        # TODO: Check if theses results change when using the 'C' backend
        np.testing.assert_array_almost_equal(
            ds.R.isel(time=24).values[1:4, 1:4],
            np.array([[2.41876631, 3.55747763, 5.21045399],
                      [2.59474309, 3.74337047, 4.95187848],
                      [2.89174467, 3.57024647, 4.49247846]]))
