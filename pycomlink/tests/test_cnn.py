import unittest
import numpy as np

from pycomlink.processing.wet_dry.cnn import cnn_wet_dry


class Testcnnpred(unittest.TestCase):
    def test_cnnpred(self):
        # generate random array
        trsl_channel_1 = np.arange(0, 60 * 5)
        trsl_channel_2 = np.arange(0, 60 * 5)

        pred = cnn_wet_dry(
            trsl_channel_1,
            trsl_channel_2,
            threshold=0.82,
            batch_size=1,
            verbose=0,
            return_raw_predictions=True,
        )

        # check if length of array is the same
        assert len(pred) == 60 * 5

        # check if array is as expected
        truth = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.7036,
                0.7037,
            ]
        )
        np.testing.assert_almost_equal(np.round(pred, decimals=4)[241:271], truth)
