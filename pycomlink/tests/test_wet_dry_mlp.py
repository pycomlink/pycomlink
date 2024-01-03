import unittest
import numpy as np
from pycomlink.processing.wet_dry.mlp import mlp_wet_dry

class Testmlppred(unittest.TestCase):
    """
    This runs the same tests as test_wet_dry_cnn.py but with different 
    content in truth_raw.
    """
    
    def test_mlppred(self):
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
            threshold=0.197, # low threshold for testing
        )

        # check if length of array is the same
        assert len(pred_raw) == 60 * 8
        assert len(pred) == 60 * 8
        
        # check if array is as expected
        truth_raw = np.array(
            [
                0.19271295,
                0.19395444,
                0.19520202,
                0.19645563,
                0.19771534,
                0.19898114,
                0.20025298,
                0.20153098,
                0.20281503,
                0.20410511,
                0.20540135,
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
                1,  
                1,  
                1,  
                1,  
                1,
                1,  
                1, 
                np.nan, 
                np.nan,
            ]
        )
        np.testing.assert_almost_equal(pred[280:293], truth)   
        np.testing.assert_almost_equal(pred_raw[280:293], truth_raw, decimal=7)
