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

    def test_without_float(self):
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


class TestOrdiniaryKrigingInterpolator(unittest.TestCase):
    def test_without_nans(self):
        xi, yi = np.meshgrid(np.linspace(0, 6, 4), np.linspace(0, 6, 4))

        interpolator = pycml.spatial.interpolator.OrdinaryKrigingInterpolator(
            nlags=10,
            variogram_model="spherical",
            variogram_parameters=None,
            weight=True,
            n_closest_points=None,
        )

        zi = interpolator(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([2, 3, 3, 5, 3]),
            z=np.array([1, 2, 3, 4, 3]),
            xgrid=xi,
            ygrid=yi,
        )

        np.testing.assert_array_almost_equal(
            zi,
            np.array(
                [
                    [2.11825683, 2.14974322, 2.55817076, 2.54075673],
                    [1.4905079, 1.70555872, 2.76440966, 2.5806816],
                    [2.06495187, 2.60523688, 3.60004479, 2.90631718],
                    [2.53986917, 2.93249053, 3.45374888, 2.90229311],
                ],
            ),
        )

    def test_with_nans(self):
        pass
