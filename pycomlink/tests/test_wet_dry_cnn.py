import unittest
import numpy as np

from pycomlink.processing.wet_dry.cnn import cnn_wet_dry


class Testcnnpred(unittest.TestCase):
    def test_cnnpred(self):
        # generate random array
        trsl_channel_1 = np.arange(0, 60 * 8).astype(float)
        trsl_channel_2 = np.arange(0, 60 * 8).astype(float)

        trsl_channel_1[320] = np.nan

        pred_raw = cnn_wet_dry(
            trsl_channel_1,
            trsl_channel_2,
            threshold=None,
            batch_size=1,
            verbose=0,
        )

        pred = cnn_wet_dry(
            trsl_channel_1,
            trsl_channel_2,
            threshold=0.7,
            batch_size=1,
            verbose=0,
        )

        # check if length of array is the same
        assert len(pred_raw) == 60 * 8
        assert len(pred) == 60 * 8

        # check if array is as expected
        truth_raw = np.array(
            [
                0.7035,
                0.7027,
                0.7016,
                0.7007,
                0.6999,
                0.6994,
                0.6984,
                0.6961,
                0.6934,
                0.6907,
                0.688,
                np.nan,
                np.nan,
            ]
        )
        truth = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                np.nan,
                np.nan,
            ]
        )
        np.testing.assert_almost_equal(pred[280:293], truth)
        np.testing.assert_almost_equal(
            np.round(pred_raw, decimals=4)[280:293], truth_raw
        )
