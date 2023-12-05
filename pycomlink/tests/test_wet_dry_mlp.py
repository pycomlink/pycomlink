import unittest
import numpy as np
from pycomlink.processing.wet_dry.mlp import mlp_wet_dry

class Testmlppred(unittest.TestCase):
    def test_mlppred(self):
        # generate random array
        trsl_channel_1 = np.arange(0, 60 * 8).astype(float)
        trsl_channel_2 = np.arange(0, 60 * 8).astype(float)

        trsl_channel_1[310] = np.nan # shorter window than in cnn

        pred_raw = mlp_wet_dry(
            trsl_channel_1,
            trsl_channel_2,
            threshold=None,
        )[:, 1]

        pred = mlp_wet_dry(
            trsl_channel_1,
            trsl_channel_2,
            threshold=0.1, # low threshold for testing
        )

        # check if length of array is the same
        assert len(pred_raw) == 60 * 8
        assert len(pred) == 60 * 8

        # check if array is as expected
        truth_raw = np.array(
            [
                0.08784304,
                0.08941595,
                0.09101421, 
                0.09263814, 
                0.09428804, 
                0.09596423,
                0.09766698, 
                0.09939668, 
                0.10115347, 
                0.10293788, 
                0.10475004,        
                np.nan,
                np.nan,
            ]
        )
        truth = np.array(
            [
                0,
                0,  
                0,  
                0,  
                0,  
                0,  
                0,  
                0,  
                1,
                1,  
                1, 
                np.nan, 
                np.nan,
            ]
        )

        np.testing.assert_almost_equal(pred[280:293], truth)   
        np.testing.assert_almost_equal(
            np.round(pred_raw, decimals=7)[280:293], truth_raw
        )
