import unittest
import numpy as np
import pycomlink as pycml
import pandas as pd
import xarray as xr
from pycomlink.processing.tensorflow_utils import wet_dry_1d_cnn
import os
import subprocess
import sys

try:
    import requests
except ImportError:
    print("[!] 'requests' not found. Installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


class TestCNNModelExactOutput(unittest.TestCase):
    def test_cnn_prediction_against_truth(self):
        # --- Step 1: Download the test dataset ---
        url = "https://github.com/toufikshit/pycomlink/releases/download/v1/Pycomlink_tensorflow_testingv0.nc"
        local_file = "Pycomlink_tensorflow_testingv0.nc"

        # Download only if not already present
        if not os.path.exists(local_file):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        # --- Step 2: Load the dataset ---
        ds = xr.load_dataset(local_file)

        results = wet_dry_1d_cnn.wet_dry_1d_cnn(ds=ds,return_ds=True)
 

        # --- Step 6: Get stored CNN values from dataset ---
        stored_probs = ds["cnn_prob_TL_Model"].values.flatten()

        # Match prediction length to stored values (should be same)
        model_probs = results["CNN"].values.flatten()

        # --- Step 7: Final assertion (rounded to 4 decimals) ---
        np.testing.assert_array_almost_equal(
            model_probs,
            stored_probs,
            decimal=5  # Note: singular "decimal"
        )


        # --- Step 8: Cleanup downloaded file ---
        os.remove(local_file)


if __name__ == '__main__':
    unittest.main()
