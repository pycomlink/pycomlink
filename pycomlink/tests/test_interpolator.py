import unittest

import pycomlink as pycml


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
    def test_temporal_aggregation(self):
        cml_list = pycml.io.examples.get_75_cmls()

        interpolator = pycml.spatial.interpolator.ComlinkGridInterpolator(
            cml_list=cml_list,
            variable='rx',
            resolution=0.05,
            interpolator=pycml.spatial.interpolator.IdwKdtreeInterpolator())

        pass