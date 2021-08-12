import unittest
import numpy as np

from pycomlink.processing.anomaly_detection.cnn_ano_detection import (
    cnn_anomaly_detection,
)


class Testcnnpred(unittest.TestCase):
    def test_cnnpred(self):

        trsl_channel_1 = np.arange(0, 5, 0.01).astype(float)
        trsl_channel_2 = np.arange(0, 5, 0.01).astype(float)

        trsl_channel_1[320] = np.nan

        pred_raw = cnn_anomaly_detection(
            trsl_channel_1,
            trsl_channel_2,
            batch_size=1,
            verbose=0,
        )

        assert len(pred_raw) == len(np.arange(0, 5, 0.01))

        truth_raw = np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                0.5839,
                0.5869,
                0.5899,
                0.5931,
                0.5964,
                0.5997,
                0.6031,
            ]
        )
        np.testing.assert_almost_equal(
            np.round(pred_raw, decimals=4)[235:245], truth_raw
        )
