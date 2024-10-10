import unittest
import numpy as np
import pandas as pd
import xarray as xr
from pycomlink.processing.wet_dry.mlp import mlp_wet_dry

class Testmlppred(unittest.TestCase):
    """
    This runs the same tests as test_wet_dry_cnn.py but with different 
    content in truth_raw.
    """
    
    def test_mlppred(self):
        # Create some synthetic data
        trsl_ch_1 = np.linspace(0, 20, 60*8).astype(float)
        trsl_ch_2 = np.linspace(0, 20, 60*8).astype(float)
        trsl_ch_1[304] = np.nan # insert a nan
        
        # Add time coordinates
        time = pd.date_range('2021-01-01', periods=trsl_ch_1.size, freq='min')
        
        # Turn into xarray
        trsl_channel_1 = xr.DataArray(trsl_ch_1, coords=[time], dims=['time'])
        trsl_channel_2 = xr.DataArray(trsl_ch_2, coords=[time], dims=['time'])
        
        # Calc wet/dry
        pred = mlp_wet_dry(
            trsl_channel_1,
            trsl_channel_2,
        )[:, 1]

        # check if length of array is the same
        assert len(pred) == 60 * 8
        
        # check if array is as expected
        truth = np.array([
            0.45518796,
            0.45518796,
            0.45518796,
            0.45518796,
            0.45518796,
            0.45518796,
            np.nan,    
            np.nan,    
            np.nan,    
            np.nan,
        ])
 
        np.testing.assert_almost_equal(pred[278+20:288+20], truth)   
