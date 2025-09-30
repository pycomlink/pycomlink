import unittest
import numpy as np
import pandas as pd
import xarray as xr
from pycomlink.processing.tensorflow_utils import run_inference


class TestCNNModelExactOutput(unittest.TestCase):
    def test_cnn_prediction_against_truth(self):
        # --- Step 1: Create synthetic dataset ---

        # Define dimensions
        n_time = 180 
        n_cml = 1
        n_channel = 2

        # Coordinates
        times = pd.date_range("2018-05-10", periods=n_time, freq="1min")
        cml_ids = ["0"]
        lengths = np.array([5.0])  # fixed length
        site_a_lat = np.array([58.0])
        site_a_lon = np.array([1.35])
        site_b_lat = np.array([57.95])
        site_b_lon = np.array([1.40])

        channel_ids = ["channel_1", "channel_2"]
        frequencies = np.array([[2.5e10, 2.6e10]])
        polarizations = np.array([["V", "V"]])

        # Data variables (all constant)
        rsl = np.full((n_channel, n_cml, n_time), -45.0)
        tsl = np.full((n_channel, n_cml, n_time), 15.0)
        tl  = np.full((n_channel, n_cml, n_time), 60.0)
        tl1 = np.full((n_cml, n_time), 59.0)
        tl2 = np.full((n_cml, n_time), 56.0)
        cnn = np.full((n_time, n_cml), np.nan, dtype=np.float32)
        cnn[-1, -1] = 0.07733  # first value, rest NaN

        # Dataset
        ds_constant = xr.Dataset(
        data_vars={
        "rsl": (("channel_id", "cml_id", "time"), rsl),
        "tsl": (("channel_id", "cml_id", "time"), tsl),
        "tl": (("channel_id", "cml_id", "time"), tl),
        "tl1": (("cml_id", "time"), tl1),
        "tl2": (("cml_id", "time"), tl2),
        "CNN": (("time", "cml_id"), cnn),
        },
        coords={
        "time": times,
        "cml_id": cml_ids,
        "length": ("cml_id", lengths),
        "site_a_latitude": ("cml_id", site_a_lat),
        "site_a_longitude": ("cml_id", site_a_lon),
        "site_b_latitude": ("cml_id", site_b_lat),
        "site_b_longitude": ("cml_id", site_b_lon),
        "channel_id": channel_ids,
        "frequency": (("cml_id", "channel_id"), frequencies),
        "polarization": (("cml_id", "channel_id"), polarizations),
            },
        )


        # --- Step 3: Run model on synthetic dataset ---
        results = run_inference.wet_dry_1d_cnn(ds=ds_constant, return_ds=True, prob_name="CNN_test",)

        # --- Step 4: Compare predictions ---
        stored_probs = ds_constant["CNN"].values.flatten()
        model_probs = results["CNN_test"].values.flatten()

        np.testing.assert_array_almost_equal(
            model_probs, stored_probs, decimal=5
        )


if __name__ == "__main__":
    unittest.main()
