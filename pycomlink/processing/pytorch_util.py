"""
Takes an xarray dataarray and runs inference on it using a PyTorch model.
Workflow:
1. Load the model.
2. Prepare the data.
3. Run inference in batches.
4. Collect and return predictions.

Dataarray shape is expected to be (time, channels, cml_id).
Output shape will be (time, channels, cml_id) with predictions for each time step.
Model input is of shape (batch_size, channels, time_window), where target time and
cml_id are captured in the batch.

"""

import os
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

# pytorch dependence is not wanted
import torch            


# TEMPORARY: import cnn model, then move this into function and load it from url
import sys, os


# -------------------- Local temporary solution ------------------------------
# TODO: cnn model will be loaded from url given as a func input
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join('C:/Users/lukas/OneDrive - VUT/TelcoSense/')))
from temp_for_cnn_models.cnn_polz_pytorch_2025 import cnn
# ----------------------------------------------------------------------------






def load_config():
    """
    Load configuration from config.yml file.
    Returns:
        dict: Configuration dictionary
    """
    package_path = Path(
        os.path.abspath(__file__)
    ).parent.parent.absolute()
    #config_path = str(package_path) + "/config/config.yml"            # warning: changed / to \\ for windows usage 
    config_path = str(Path(package_path) / "config" / "config.yml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def predict_batch(model, batch, device):
    model.eval()
    with torch.no_grad():
        inputs = batch.to(device)
        outputs = model(inputs)
    return outputs


# TODO: so this function should load specific model from url
def load_model(model_path, device):
    """
    Loads a PyTorch model from the specified path.
    Args:
        model_path (str): Path to the model file.
        device (torch.device): Device to load the model on.
    Returns:
        model (torch.nn.Module): The loaded PyTorch model.
    """
    # Create the model instance first
    model = cnn(
        final_act="sigmoid"
    )  # Default to sigmoid, might need to be configurable

    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move model to device
    model.to(device)

    # Add window_size attribute (based on the data preprocessing, it's 180)
    model.window_size = 180

    return model


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


def cnn_wd(model_path_or_run_id, data, batch_size=32, config_path=None):
    """
    Function to run wet/dry inference on input data using a trained CNN model.
    Args:
        model_path_or_run_id (str): Either a path to the trained PyTorch model or a run_id.
                                   If run_id, will look for model and config in results/{run_id}/
        data (xarray.DataArray): The input data array.
        batch_size (int): The number of samples in each batch.
        config_path (str, optional): Path to config file. If None, uses default config location
                                    or looks for config in results/{run_id}/config.yml if run_id is provided.
    Returns:
        xarray.Dataset: Dataset with predictions added as a new variable.
    """
    device = set_device()

    # Determine if input is a run_id or model path
    if model_path_or_run_id.endswith(".pth") or "/" in model_path_or_run_id:
        # It's a model path
        model_path = model_path_or_run_id
        model = load_model(model_path, device)

        # Load config to get reflength parameter
        if config_path is None:
            config = load_config()
        else:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
    else:
        # It's a run_id
        run_id = model_path_or_run_id
        package_path = Path(
            os.path.abspath(__file__)
        ).parent.parent.parent.parent.absolute()
        results_dir = Path(package_path) / "results" / run_id

        # Find the latest model file in the models directory
        models_dir = results_dir / "models"
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        model_files = list(models_dir.glob("model_epoch_*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in: {models_dir}")

        # Sort by epoch number and get the latest
        model_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
        latest_model = model_files[-1]
        print(f"Using model: {latest_model}")

        model = load_model(str(latest_model), device)

        # Load config from results directory
        config_file = results_dir / "config.yml"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            print(f"Using config from: {config_file}")
        else:
            print(f"Config file not found at {config_file}, using default config")
            config = load_config()

    reflength = config.get("data", {}).get(
        "reflength", 60
    )  # Default to 60 if not found

    results = run_inference(model, data, batch_size, reflength)
    # Convert xarray DataArray to Dataset if needed
    if isinstance(data, xr.DataArray):
        data = data.to_dataset(name="TL")  
    final_results = redistribute_results(results, data)
    return final_results


def test_cnn_wd():
    """
    Test function to run inference with a sample model and data.
    This is for demonstration purposes and should be replaced with actual data and model paths.
    """
    from pathlib import Path

    # Get repository root directory
    repo_root = Path(__file__).parent.parent.parent.parent

    # Example usage with model path
    model_path = (
        repo_root / "data/dummy_model/model_epoch_0.pth"
    )  # Relative path to model
    data = xr.DataArray(
        np.random.rand(1000, 2, 5),
        dims=["time", "channels", "cml_id"],
        coords={
            "time": np.datetime64("2023-01-01") + np.arange(1000),
            "channels": np.arange(2),
            "cml_id": ["A", "B", "C", "D", "E"],
        },
    )
    final_dataset = cnn_wd(str(model_path), data, batch_size=32)
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




if __name__ == "__main__":
    test_cnn_wd()
