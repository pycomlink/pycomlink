import unittest
import numpy as np
import pycomlink as pycml
from pycomlink.tests.utils import load_and_clean_example_cml_list


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
        cml_list = load_and_clean_example_cml_list()

        interpolator = pycml.spatial.interpolator.ComlinkGridInterpolator(
            cml_list=cml_list,
            variable='rx',
            resolution=0.05,
            interpolator=pycml.spatial.interpolator.IdwKdtreeInterpolator())

        zgrid = interpolator.loop_over_time()

        assert zgrid[0].shape == (16, 23)

        np.testing.assert_array_almost_equal(zgrid[10][1:4, 1:4], np.array(
            [[-46.85263666, -46.87468505, -46.21003706],
             [-46.70700606, -46.92331939, -46.92513674],
             [-46.15441647, -46.55741421, -46.39807402]]))

        np.testing.assert_array_almost_equal(zgrid[20][1:4, 1:4], np.array(
            [[-46.86313914, -46.89160703, -46.26548933],
             [-46.71264833, -46.92780239, -46.96490213],
             [-46.11732763, -46.53814125, -46.4805056]]))

        zgrid_i = interpolator.interpolate_for_i(3)
        np.testing.assert_array_almost_equal(zgrid_i[1:4, 1:4], np.array(
            [[-46.84939582, -46.89709485, -46.26185758],
             [-46.71798389, -46.96183674, -46.97195592],
             [-46.12521554, -46.55562653, -46.48171999]]))

    def test_default_kriging(self):
        cml_list = load_and_clean_example_cml_list()

        interpolator = pycml.spatial.interpolator.ComlinkGridInterpolator(
            cml_list=cml_list,
            variable='rx',
            resolution=0.05,
            interpolator=(pycml.spatial.interpolator.
                          OrdinaryKrigingInterpolator()))

        zgrid = interpolator.loop_over_time()

        assert zgrid[0].shape == (16, 23)

        np.testing.assert_array_almost_equal(zgrid[10][1:4, 1:4], np.array(
            [[-46.37344738, -46.37336952, -46.37318653],
             [-45.62199173, -46.37342665, -46.37327972],
             [-45.25576347, -45.25585099, -46.01024901]]))
