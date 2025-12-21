"""
PyTorch utilities for model inference and data handling.

Central PyTorch utility toolkit for device selection, batch prediction,
DataLoader construction, and TorchScript model loading utilities.

Key Features:
- Automatically select the appropriate device (CPU/GPU) for PyTorch operations.
- Run model inference on single batches.
- Build PyTorch DataLoaders from preprocessed sample data.
- Load TorchScript-exported models for inference.

Main Functions:
    set_device(): Detects and returns the optimal torch.device.
    predict_batch(model, batch, device): Performs inference on a single batch.
    build_dataloader(combined_samples, batch_size): Constructs a DataLoader and returns associated metadata.
    load_model(model_path, device): Loads cached TorchScript model and prepares it for inference.

Example Usage:
    # Set device
    device = set_device()

    # Load model
    model = load_model("path/to/model.pt", device)

    # Build dataloader
    dataloader, cml_ids, times = build_dataloader(combined_samples, batch_size)

    # Run batch inference in loop
    for batch in dataloader:
        outputs = predict_batch(model, batch, device)
"""

from pathlib import Path

import torch


def set_device():
    """Auto-detect and return optimal torch.device (GPU/CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


# This function is for pytorch model inference on a single batch
def predict_batch(model, batch, device):
    """Run model inference on a single batch."""
    with torch.no_grad():
        inputs = batch.to(device)
        outputs = model(inputs).cpu().numpy()
    return outputs


# Build PyTorch DataLoader
def build_dataloader(combined_samples, batch_size, reshape=(0, 2, 1)):
    """
    Builds a PyTorch DataLoader from the input data.

    Args:
        data (xarray.DataArray): Batched input data array.
        window_size (int): The size of each time series window.
        batch_size (int): The number of samples in each batch.
        reshape (bool or tuple): Whether to reshape the input before passing to the model (default: (0, 2, 1)).

    Returns:
        dataloader (torch.utils.data.DataLoader): A DataLoader for the input data.
    """
    # Only batch the data tensor; keep cml_id and time as arrays outside the DataLoader
    tensor_data = torch.tensor(combined_samples["data"], dtype=torch.float32)

    # Handle reshape parameter - can be bool or tuple
    if isinstance(reshape, bool):
        if reshape:
            reshape = (0, 2, 1)  # Use default reshape
        else:
            reshape = None  # No reshape
    elif isinstance(reshape, tuple):
        reshape = tuple(int(x) for x in reshape)

    if reshape is not None:
        tensor_data = tensor_data.permute(*reshape)  # (batch, channels, window)

    dataloader = torch.utils.data.DataLoader(
        tensor_data, batch_size=batch_size, shuffle=False
    )
    # Return dataloader and metadata arrays
    return dataloader, combined_samples["cml_id"], combined_samples["time"]


# Starting attempt to load model using jit
def load_model(model_path, device):
    """
    Load PyTorch model from local file path using torch.jit.load (TorchScript)

    Args:
        model_path (str): Path to the model file.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded PyTorch model.
    """
    model_path = Path(model_path)

    # Load JIT scripted model
    try:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()  # Set to evaluation mode

        # Add window_size attribute (based on the data preprocessing, it's 180)
        model.window_size = 180
        print(f"✅ Loaded exported model from: {model_path}")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load exported model {model_path}: {e}")
