"""
Inference utilities for CML wet/dry classification models.

This module provides utility functions for loading, caching, and managing PyTorch models
used for Commercial Microwave Link (CML) wet/dry classification inference. It supports
multiple model loading mechanisms including:

- Local file paths (.pth files)
- Remote URLs with automatic download and caching

Key Features:
    - Automatic model downloading and caching from URLs
    - Smart model loading with fallback for PyTorch compatibility
    - Support for different model sources (local, remote)
    - GPU/CPU device detection and management

Main Functions:
    - get_model(): Universal model loader supporting multiple input types
    - load_model(): Load PyTorch model from local path
    - download_and_cache_model(): Download and cache models from URLs
    - set_device(): Auto-detect and set appropriate device (GPU/CPU)

Cache Management:
    Models downloaded from URLs are cached locally in ~/.cml_wd_pytorch/models/
    to avoid repeated downloads. Cache can be managed with clear_model_cache()
    and list_cached_models() functions.

Example Usage:
    # Load from local path
    model = get_model("path/to/model.pth")

    # Load from URL (with automatic caching)
    model = get_model("https://example.com/model.pth")
"""

import hashlib
import sys,os
import urllib.request
from pathlib import Path

import torch



def set_device():
    """Auto-detect and return appropriate device (GPU/CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def download_and_cache_model(
    model_url, cache_dir="~/.cml_wd_pytorch/models", force_download=False
):
    """
    Download and cache a model from URL.

    Args:
        model_url (str): URL to download the model from
        cache_dir (str): Local directory to cache models
        force_download (bool): Force re-download even if cached

    Returns:
        Path: Path to the cached model file
    """
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from URL hash to avoid conflicts
    url_hash = hashlib.md5(model_url.encode()).hexdigest()
    model_filename = f"model_{url_hash}.pth"
    cached_path = cache_dir / model_filename

    if not cached_path.exists() or force_download:
        print(f"Downloading model from {model_url}...")
        urllib.request.urlretrieve(model_url, cached_path)
        print(f"Model cached at {cached_path}")
    else:
        print(f"Using cached model at {cached_path}")

    return cached_path


def clear_model_cache(cache_dir="~/.cml_wd_pytorch/models"):
    """
    Clear the model cache directory.

    Args:
        cache_dir (str): Cache directory to clear
    """
    cache_dir = Path(cache_dir).expanduser()
    if cache_dir.exists():
        for file in cache_dir.glob("*.pth"):
            file.unlink()
        print(f"Cleared cache at {cache_dir}")
    else:
        print(f"Cache directory {cache_dir} does not exist")


def list_cached_models(cache_dir="~/.cml_wd_pytorch/models"):
    """
    List all cached models.

    Args:
        cache_dir (str): Cache directory to list

    Returns:
        list: List of cached model files
    """
    cache_dir = Path(cache_dir).expanduser()
    if cache_dir.exists():
        return list(cache_dir.glob("*.pth"))
    return []


def load_model(model_path, device):
    """
    Load PyTorch model from file path.

    Args:
        model_path (str): Path to the model file.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded PyTorch model.
    """

    # Add the model path into env
    sys.path.append(os.path.abspath(Path(model_path).parent.absolute()))
    #sys.path.append(os.path.abspath(Path(model_path).absolute()))
    # TODO: new solution needed for when the path can be dir or .pth file
    from cnn_polz_pytorch_2025 import cnn           # Temporary solution
    
    # Create the model instance first
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

    # Add window_size attribute (based on the data preprocessing, it's 180)
    model.window_size = 180
    return model




def _load_model_from_url(model_url, force_download=False):
    """Load model, weights from URL by downloading and caching it."""
    device = set_device()

    # Download and cache the model
    model_path = download_and_cache_model(model_url, force_download=force_download)

    # Load model with weights
    model = load_model(str(model_path), device)
    return model


def _load_model_from_local_path(model_path):
    """Load model from local file path."""
    device = set_device()

    # Load the model
    model = load_model(model_path, device)
    return model




def get_model(model_path_or_url, force_download=False):
    """
    Load a model from a local path, or URL.

    Args:
        model_path_or_url (str): Either a path to the trained PyTorch model,
                                          or a URL to download the model from.
                                          If URL, will download and cache the model locally.
        force_download (bool): Force re-download of model if it's a URL (default: False).

    Returns:
        model - The loaded PyTorch model.
    """
    # Determine input type and delegate to appropriate handler
    if model_path_or_url.startswith(("http://", "https://")):
        # It's a URL
        return _load_model_from_url(
            model_path_or_url, force_download
        )
    elif (
        model_path_or_url.endswith(".pth")
        or "/" in model_path_or_url
    ):
        # It's a local model path
        return _load_model_from_local_path(model_path_or_url)
    else:
        # It's neithe url, nor path
        raise Exception(f"Provided string: '{model_path_or_url}' , is neither directory path, nor web url")
