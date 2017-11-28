import unittest

import pycomlink as pycml


class TestWetDryStdDev(unittest.TestCase):
    def test(self):
        cml = pycml.io.examples.read_one_cml()
        cml.process.wet_dry.std_dev(window_length=30, threshold=0.8)

