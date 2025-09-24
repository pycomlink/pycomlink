import unittest
import numpy as np
import pycomlink as pycml
import pandas as pd
import xarray as xr
from pycomlink.processing.tensorflow_utils import cnn_refrectored
import os
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

        # --- Step 3: Prepare input for model ---
        cml_ids = ds.cml_id.values

        X_input, ids_out = cnn_refrectored.create_samples(
            ds=ds,
            cml_ids=cml_ids,
            input_vars=["tl1", "tl2"],
            seq_len=180
        )

        X_scaled, _ = cnn_refrectored.scale_features(
            input_seq=X_input,
            input_vars=["tl1", "tl2"]
        )

        # --- Step 4: Load model ---
        json_url = "https://github.com/toufikshit/pycomlink/releases/download/v1/CNN__model_v0_cz.json"
        weights_url = "https://github.com/toufikshit/pycomlink/releases/download/v1/CNN__model_v0_cz.weights.h5"
        json_path, weights_path = cnn_refrectored.resolve_model_paths(
            json_url, weights_url, cache_dir="model_cnn"
        )
        model = cnn_refrectored.load_model_from_local(json_path, weights_path, lr=0.05)

        # --- Step 5: Run inference ---
        y_prob, _ = cnn_refrectored.run_inference(
            model=model,
            model_input=X_scaled,
            threshold=0.1,
            batch_size=32
        )
        

        results = cnn_refrectored.store_predictions(
            merged=ds,
            model_prob=y_prob, #model_prob
            cml_ids=ids_out,
            model_name="TL_Test"
        )


        # --- Step 6: Get stored CNN values from dataset ---
        stored_probs = ds["cnn_prob_TL_Model"].values.flatten()

        # Match prediction length to stored values (should be same)
        model_probs = results["cnn_prob_TL_Test"].values.flatten()

        # --- Step 7: Final assertion (rounded to 4 decimals) ---
        np.testing.assert_almost_equal(
            model_probs,
            stored_probs)

        # --- Step 8: Cleanup downloaded file ---
        os.remove(local_file)


if __name__ == '__main__':
    unittest.main()
