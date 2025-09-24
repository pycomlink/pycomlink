"""
CML Wet/Dry Classification Inference Engine.

This module provides the main inference pipeline for Commercial Microwave Link (CML)
wet/dry classification using trained CNN models. It handles the complete workflow from
raw xarray data to predictions, including data preprocessing, windowing, batching,
model inference, and result reconstruction.

Core Functionality:
    The module implements a sliding window approach for time series classification,
    where each prediction is made based on a temporal window of CML measurements.
    The main entry point is the `cnn_wd()` function which provides a high-level
    interface for running inference on CML data.

Data Flow:
    1. Input: xarray.DataArray with shape (time, channels, cml_id)
    2. Preprocessing: Create sliding windows across time dimension
    3. Batching: Group windows into batches for efficient GPU processing
    4. Inference: Run CNN model on batches to get predictions
    5. Reconstruction: Map predictions back to original time/cml_id grid
    6. Output: xarray.Dataset with original data + predictions

Key Components:

Data Preprocessing:
    - rolling_window(): Creates sliding windows from time series data
    - batchify_windows(): Organizes windows across all CML links
    - build_dataloader(): Creates PyTorch DataLoader for batch processing

Inference Pipeline:
    - predict_batch(): Runs model inference on a single batch
    - run_inference(): Orchestrates the complete inference process
    - redistribute_results(): Maps batch predictions back to original structure

Main Interface:
    - cnn_wd(): High-level function for wet/dry classification
    - Supports multiple model sources (local files, URLs, run IDs)
    - Automatic model downloading and caching
    - Configurable batch sizes and processing parameters

Data Shapes and Transformations:
    Input DataArray:    (time, channels, cml_id)
    Windowed Data:      (n_windows, channels, window_size)
    Model Input:        (batch_size, channels, window_size)
    Model Output:       (batch_size, 1) - binary wet/dry predictions
    Final Output:       (time, cml_id) - predictions mapped to original grid

Configuration:
    The module uses configuration parameters from YAML files to control:
    - Window sizes and temporal offsets (reflength parameter)
    - Model architecture and preprocessing parameters
    - Batch sizes for efficient memory usage

Example Usage:
    # Basic usage with local model
    import xarray as xr
    data = xr.open_dataarray("cml_data.nc")
    results = cnn_wd("path/to/model.pth", data)

    # Usage with remote model (auto-download and cache)
    results = cnn_wd("https://example.com/model.pth", data)

    # Usage with training run ID
    results = cnn_wd("2025-01-15_12-34-56abc123", data)

    # Access predictions
    wet_dry_predictions = results['predictions']
    original_data = results['TL']

Notes:
    - The module assumes CML data follows specific naming conventions
    - Window size is typically 180 time steps (3 hours for 1-minute data)
    - Predictions are binary (0=dry, 1=wet) with sigmoid activation
    - Missing data is handled gracefully with NaN placeholders
    - GPU acceleration is used automatically when available

Dependencies:
    - PyTorch for model inference
    - xarray for data handling
    - numpy for numerical operations
    - Custom inference_utils for model loading and device management
"""

import numpy as np
import torch
import xarray as xr

from cml_wd_pytorch.inference.inference_utils import (
    get_model,
    list_cached_models,
    set_device,
)


def predict_batch(model, batch, device):
    """Run model inference on a single batch."""
    model.eval()
    with torch.no_grad():
        inputs = batch.to(device)
        outputs = model(inputs)
    return outputs


# TODO: Add unit tests for these functions
# TODO: move to general utils?
def rolling_window(timeseries, valid_times, window_size, reflength=60):
    """
    Splits the time series into batches of specified size.
    Args:
        timeseries (list or np.array): The time series data to be split.
        valid_times (list): A list of valid time indices.
        window_size (int): The size of each batch.
        reflength (int): The reference length for timestamp calculation (from config).
    Returns:
        windowed_series (np.array): A list of batches, each containing a segment of the time series.
        timestep_indices (list): A list of indices corresponding to the target time (end of window - reflength).
    """
    assert window_size > 0, "Window size must be greater than 0"
    assert len(valid_times) == len(timeseries), (
        "Valid times must match the length of the timeseries"
    )
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


def batchify_windows(data, window_size, batch_size, reflength=60):
    """
    Converts a data array into batches of specified window size.
    Args:
        data (xarray.DataArray): The input data array.
        window_size (int): The size of each time series window.
        batch_size (int): The number of samples in each batch.
        reflength (int): The reference length for timestamp calculation (from config).
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


def build_dataloader(data, window_size, batch_size, device, reflength=60):
    """
    Builds a PyTorch DataLoader from the input data.
    Args:
        data (xarray.DataArray): The input data array.
        window_size (int): The size of each time series window.
        batch_size (int): The number of samples in each batch.
        device (torch.device): The device to run the model on.
        reflength (int): The reference length for timestamp calculation (from config).
    Returns:
        dataloader (torch.utils.data.DataLoader): A DataLoader for the input data.
    """
    combined_samples = batchify_windows(data, window_size, batch_size, reflength)

    # Only batch the data tensor; keep cml_id and time as arrays outside the DataLoader
    tensor_data = torch.tensor(combined_samples["data"], dtype=torch.float32)
    tensor_data = tensor_data.permute(0, 2, 1)  # (batch, channels, window)

    dataloader = torch.utils.data.DataLoader(
        tensor_data, batch_size=batch_size, shuffle=False
    )
    # Return dataloader and metadata arrays
    return dataloader, combined_samples["cml_id"], combined_samples["time"]


def run_inference(model, data, batch_size=32, reflength=60):
    device = set_device()
    window_size = model.window_size if hasattr(model, "window_size") else 180
    dataloader, cml_ids, times = build_dataloader(
        data, window_size, batch_size, device, reflength
    )
    predictions = []

    for batch in dataloader:
        inputs = batch.to(device)
        outputs = predict_batch(model, inputs, device)
        predictions.append(outputs.cpu())

    all_predictions = torch.cat(predictions, dim=0)
    # cml_ids and times are already numpy arrays, no need to collect per batch
    return {"predictions": all_predictions, "cml_ids": cml_ids, "times": times}


def redistribute_results(results, data):
    """
    Redistribute the 1D inference results back to the original data structure.

    Args:
        results (dict): Dictionary containing predictions, cml_ids, and times from inference
        data (xarray.Dataset): Original dataset to add predictions to

    Returns:
        xarray.Dataset: Dataset with predictions added as a new variable
    """
    predictions = (
        results["predictions"].numpy().squeeze()
    )  # Remove any extra dimensions
    cml_ids = results["cml_ids"]  # Already numpy array
    times = results["times"]  # Already numpy array

    # Get original dimensions
    ref_times = data.time.to_numpy()
    ref_cml_ids = data.cml_id.to_numpy()

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
    model_path_or_run_id_or_url,
    data,
    batch_size=32,
    config_path=None,
    force_download=False,
):
    """
    Function to run wet/dry inference on input data using a trained CNN model.
    Args:
        model_path_or_run_id_or_url (str): Either a path to the trained PyTorch model, a run_id,
                                          or a URL to download the model from.
                                          If run_id, will look for model and config in results/{run_id}/
                                          If URL, will download and cache the model locally.
        data (xarray.DataArray): The input data array.
        batch_size (int): The number of samples in each batch.
        config_path (str, optional): Path to config file. If None, uses default config location
                                    or looks for config in results/{run_id}/config.yml if run_id is provided.
        force_download (bool): Force re-download of model if it's a URL (default: False).
    Returns:
        xarray.Dataset: Dataset with predictions added as a new variable.
    """

    model, config = get_model(model_path_or_run_id_or_url, config_path, force_download)

    reflength = config.get("data", {}).get(
        "reflength", 60
    )  # Default to 60 if not found

    results = run_inference(model, data, batch_size, reflength)
    data = data.to_dataset(name="TL")  # Convert xarray DataArray to Dataset if needed
    final_results = redistribute_results(results, data)
    return final_results


def test_cnn_wd():
    """
    Test function to run inference with a sample model and data.
    This is for demonstration purposes and should be replaced with actual data and model paths.
    """

    # Example usage with model URL
    model_url = "https://github.com/jpolz/cml_wd_pytorch/raw/main/data/dummy_model/model_epoch_15.pth"  # Relative path to model
    data = xr.DataArray(
        np.random.rand(1000, 2, 5),
        dims=["time", "channels", "cml_id"],
        coords={
            "time": np.datetime64("2023-01-01") + np.arange(1000),
            "channels": np.arange(2),
            "cml_id": ["A", "B", "C", "D", "E"],
        },
    )
    final_dataset = cnn_wd(str(model_url), data, batch_size=32)
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Final dataset variables: {list(final_dataset.data_vars.keys())}")
    logging.info(f"Final dataset dimensions: {final_dataset.dims}")
    if "predictions" in final_dataset:
        logging.info(f"Predictions shape: {final_dataset['predictions'].shape}")
        logging.info(f"Predictions dimensions: {final_dataset['predictions'].dims}")
        logging.info(
            f"Sample predictions:\n{final_dataset['predictions'][500:505, :3].values}"
        )

    # Example usage with run_id (this would fail in test but shows the interface)
    # final_dataset_from_run_id = cnn_wd("2025-01-15_12-34-56abc123", data, batch_size=32)

    print("Test completed successfully!")
    print(f"Cached models: {len(list_cached_models())}")


if __name__ == "__main__":
    test_cnn_wd()
