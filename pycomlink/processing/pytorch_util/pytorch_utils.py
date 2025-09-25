import torch

# This function is for pytorch model inference on a single batch
def predict_batch(model, batch, device):
    """Run model inference on a single batch."""
    model.eval()
    with torch.no_grad():
        inputs = batch.to(device)
        outputs = model(inputs).cpu().numpy()
    return outputs


# Build PyTorch DataLoader
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