

import torch
from pathlib import Path



def set_device():
    """Auto-detect and return appropriate device (GPU/CPU)."""
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
def build_dataloader(combined_samples, batch_size):
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
    # Only batch the data tensor; keep cml_id and time as arrays outside the DataLoader
    tensor_data = torch.tensor(combined_samples["data"], dtype=torch.float32)
    tensor_data = tensor_data.permute(0, 2, 1)  # (batch, channels, window)

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

