import unittest
import numpy as np

from pycomlink.processing.wet_dry.cnn import cnn_wet_dry


class Testcnnpred(unittest.TestCase):
    def test_cnnpred(self):

        # generate random array
        trsl_channel_1 = 10 * np.random.rand(60 * 24 * 5)
        trsl_channel_2 = 10 * np.random.rand(60 * 24 * 5)

        pred = cnn_wet_dry(
            trsl_channel_1, trsl_channel_2, threshold=0.82, batch_size=100, verbose=0
        )

        # check if length of array is the same
        assert len(pred) == 60 * 24 * 5
