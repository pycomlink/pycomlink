import unittest
import copy
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
        interpolator = pycml.spatial.interpolator.IdwKdtreeInterpolator(nnear=12, p=2)

        xi, yi = np.meshgrid(np.linspace(0, 6, 4), np.linspace(0, 6, 4))

        zi = interpolator(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([2, 4, 3, 5, 1]),
            z=np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            xgrid=xi,
            ygrid=yi,
        )

        np.testing.assert_array_almost_equal(
            zi,
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        )

    def test_without_float(selfself):
        interpolator = pycml.spatial.interpolator.IdwKdtreeInterpolator(nnear=12, p=2)

        xi, yi = np.meshgrid(np.linspace(0, 6, 4), np.linspace(0, 6, 4))

        zi = interpolator(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([2, 4, 3, 5, 1]),
            z=np.array([1, 2, 3, 4, 5]),
            xgrid=xi,
            ygrid=yi,
        )

        np.testing.assert_array_almost_equal(
            zi,
            np.array(
                [
                    [2.05352949, 2.54119688, 4.09027081, 4.42468505],
                    [1.45942756, 1.9760479, 3.56701031, 4.23470411],
                    [2.16589862, 2.0, 3.41317365, 3.54032958],
                    [2.54801273, 2.82949309, 3.66892889, 3.48354789],
                ]
            ),
        )
