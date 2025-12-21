"""
CNN inference pipeline for CML wet/dry classification.

This module provides functions to run inference on Commercial Microwave Link (CML)
data using trained CNN models. The main workflow includes creating sliding windows
from time series data, running batch inference, and redistributing predictions back
to the original data structure.

Main Functions:
    - cnn_wd(): High-level interface for wet/dry classification
    - run_inference(): Core inference pipeline
    - rolling_window(): Creates sliding windows from time series
    - batchify_windows(): Converts a data array into rolling-window batches of specified size.
    - redistribute_results(): Maps predictions back to original coordinates

Example Usage:
    # Run the inference function cnn_wd(). The rest is called automatically
    result = run_inference.cnn_wd(model_path_or_url="url-path-string",data=total_loss_data)
"""

import numpy as np
import xarray as xr

from pycomlink.processing.pytorch_utils.inference_utils import (
    get_model,
    set_device,
)
from pycomlink.processing.pytorch_utils.pytorch_utils import (
    predict_batch,
    build_dataloader,
)


# TODO: Add unit tests for these functions
def rolling_window(timeseries, valid_times, window_size, reflength=60):
    """
    Define a rolling-window of specified size to load sorrounding TL samples for each timestep, .

    Args:
        timeseries (list or np.array): The time series data to be split.
        valid_times (list): A list of valid time indices.
        window_size (int): The size of each batch.
        reflength (int): The reference length for timestamp calculation.

    Returns:
        windowed_series (np.array): A list of batches, each containing a segment of the time series.
        timestep_indices (list): A list of indices corresponding to the target time (end of window - reflength).
    """
    assert window_size > 0, "Window size must be greater than 0"
    assert len(valid_times) == len(
        timeseries
    ), "Valid times must match the length of the timeseries"
    windowed_series = []
    timestep_indices = []
    for start in range(len(timeseries)):
        end = start + window_size
        if end <= len(timeseries):
            windowed_series.append(timeseries[start:end])
            # Use the timestamp that corresponds to the prediction target time
            # This is the end of the window minus the reflength offset
            target_idx = (
                end - int(reflength / 2) if end - int(reflength / 2) >= 0 else end - 1
            )
            timestep_indices.append(valid_times[target_idx])
    return windowed_series, timestep_indices


def batchify_windows(data, window_size, reflength=60):
    """
    Converts a data array into batches of specified window size.

    Args:
        data (xarray.DataArray): The input data array.
        window_size (int): The size of each time series window.
        reflength (int): The reference length for timestamp calculation.

    Returns:
        combined_samples (dict): A dictionary containing concatenated cml_id, time, and data arrays.
    """
    samples = []
    cml_ids = data.cml_id.values
    # iterate over cml_id dimension
    for cml in cml_ids:
        cml_data = data.sel(cml_id=cml)
        # iterate over time dimension
        timeseries = cml_data.values
        valid_times = cml_data.time.values
        windowed_series, timestep_indices = rolling_window(
            timeseries, valid_times, window_size, reflength
        )
        cml_id = np.repeat(cml, len(windowed_series))
        samples.append(
            {
                "cml_id": cml_id,
                "time": timestep_indices,
                "data": np.array(windowed_series),
            }
        )
    # Combine all samples into a single array
    combined_samples = {
        "cml_id": np.concatenate([sample["cml_id"] for sample in samples]),
        "time": np.concatenate([sample["time"] for sample in samples]),
        "data": np.concatenate([sample["data"] for sample in samples]),
    }

    return combined_samples


# Main inference function, pytorch specific
def run_inference(model, data, batch_size=32, reflength=60, reshape=False):
    """
    Function to run inference loop on dataloader batched input data.

    Args:
        model (torch.nn.Module): Loaded PyTorch model.
        data (xarray.DataArray): The input total loss data.
        batch_size (int): The number of samples in each batch.
        reflength (int): The reference length for timestamp calculation.
        reshape (bool or tuple): Whether to reshape the input before passing to the model (default: False).

    Returns:
        dict: dictionary with numpy arrays of predictions cml_ids and time series.
    """

    device = set_device()
    window_size = model.window_size if hasattr(model, "window_size") else 180
    combined_samples = batchify_windows(data, window_size, reflength)
    dataloader, cml_ids, times = build_dataloader(
        combined_samples,
        batch_size,
        reshape=reshape,
    )
    predictions = []

    for batch in dataloader:
        inputs = batch.to(device)
        outputs = predict_batch(model, inputs, device)
        predictions.append(outputs)

    all_predictions = np.concatenate(predictions, axis=0)
    # cml_ids and times are already numpy arrays, no need to collect per batch
    return {"predictions": all_predictions, "cml_ids": cml_ids, "times": times}


def redistribute_results(results, data):
    """
    Redistribute the 1D inference results back to the original data structure.

    Args:
        results (dict): Dictionary containing predictions, cml_ids, and times from inference
        data (xarray.Dataset): Original dataset to add predictions to.

    Returns:
        xarray.Dataset: Dataset with predictions added as a new variable
    """
    predictions = results["predictions"]  # Remove any extra dimensions
    cml_ids = results["cml_ids"]  # Already numpy array
    times = results["times"]  # Already numpy array

    # Get original dimensions
    ref_times = data.time
    ref_cml_ids = data.cml_id

    # Initialize prediction array with NaN values
    pred_array = np.full((len(ref_times), len(ref_cml_ids)), np.nan)

    # Map predictions back to the original grid
    for i, (pred_cml_id, pred_time, pred_value) in enumerate(
        zip(cml_ids, times, predictions)
    ):
        # Find indices in the original data
        try:
            cml_idx = np.where(ref_cml_ids == pred_cml_id)[0][0]
            time_idx = np.where(ref_times == pred_time)[0][0]
            pred_array[time_idx, cml_idx] = pred_value
        except (IndexError, ValueError):
            # Skip if the time or cml_id is not found in the original data
            continue

    # Create a new DataArray to hold the predictions
    pred_data = xr.DataArray(
        pred_array,
        dims=["time", "cml_id"],
        coords={
            "time": ref_times,
            "cml_id": ref_cml_ids,
        },
        name="predictions",
    )

    # Add the predictions to the dataset
    return data.assign(predictions=pred_data)


def cnn_wd(
    model_path_or_url,
    data,
    batch_size=32,
    force_download=False,
    reflength=60,  # TODO: may be generalized in future
    reshape=(0, 2, 1),
):
    """
    Function to run wet/dry inference on input data using loaded trained CNN model.

    Args:
        model_path_or_url (str): Either a path to the trained PyTorch model, or a URL to download
                                 the model from. If URL, will download and cache the model locally.
        data (xarray.DataArray): The input cml total loss data as dataarray.
        batch_size (int): The number of samples in each batch.
        force_download (bool): Force re-download of model if it's a URL (default: False).
        reflength (int): The reference length for timestamp calculation (from config).
        reshape (bool or tuple): Whether to reshape the input before passing to the model (default: (0, 2, 1)).

    Returns:
        xarray.Dataset: Dataset with predictions added as a new variable.

    Reshaping:
        By default input data is permuted to (batch, channels, window) which equivalent to passing (0, 2, 1). 
        For other reshaping, pass a tuple with the desired permutation, e.g. (0,1,2) for (batch, window, channels).
    """

    model = get_model(model_path_or_url, force_download)

    results = run_inference(model, data, batch_size, reflength, reshape)
    data = data.to_dataset(name="TL")  # Convert xarray DataArray to Dataset if needed
    final_results = redistribute_results(results, data)
    return final_results
