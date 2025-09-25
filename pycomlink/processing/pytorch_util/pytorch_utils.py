import torch
from pathlib import Path
import sys, os


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



# TODO: update this to be able to load either pt2 file, or if not present, load pth weights and .py architecture
# this way there could be some operability for all platforms: either download automatically from url and use pt2
# or download .py and .pth manually and provide the directory
def load_model(model_path, device):
    """
    Load PyTorch model from file path.

    Args:
        model_path (str): Path to the model file.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded PyTorch model.
    """
    model_path = Path(model_path)
    
    if model_path.suffix == ".pt2":
        # Load exported model using torch.export This option works only on Linux
        try:
            exported_program = torch.export.load(str(model_path))
            model = exported_program.module()
            model.to(device)

            # Add window_size attribute (based on the data preprocessing, it's 180)
            model.window_size = 180
            print(f"✅ Loaded exported model from: {model_path}")
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load exported model {model_path}: {e} \n"
                               "Note: loading .pt2 files is currently only possible on Linux os.")
        
    elif model_path.suffix == ".pth":
        # Load the model class and .pth weights, Supported on Windows
  
        # Add the model path into env
        sys.path.append(os.path.abspath(Path(model_path).parent.absolute()))
        from cnn_polz_pytorch_2025 import cnn           # Temporary solution
        model = cnn(
            final_act="sigmoid"
        )  # Default to sigmoid, might need to be configurable
  
        # Load the state dict
        try:
            # First try with weights_only=True for security
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:
            # Fall back to weights_only=False for compatibility with older model files
            # This should only be used with trusted model files
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)

        # Move model to device
        model.to(device)

        # Add window_size attribute (Hardcoded based on the data preprocessing, it's 180)
        model.window_size = 180

        return model
    


# Starting attempt to load model using jit

'''
def load_model_jit(model_path, device):
    """
    Load PyTorch model from file path using torch.jit.load
    """
    model_path = Path(model_path)
    
    assert model_path.suffix == ".pt2", "Model file must be a .pt2 file"

    # Load exported model using torch.export
    try:
        torch.jit.load(f, map_location=None, _extra_files=None, _restore_shapes=False)

        # Add window_size attribute (based on the data preprocessing, it's 180)
        model.window_size = 180
        print(f"✅ Loaded exported model from: {model_path}")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load exported model {model_path}: {e}")

'''