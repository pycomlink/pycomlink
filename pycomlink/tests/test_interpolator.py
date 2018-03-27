import unittest
import numpy as np
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
