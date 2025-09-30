"""
Inference utilities for CML wet/dry classification models.

This module provides utility functions for loading, caching, and managing PyTorch models
used for Commercial Microwave Link (CML) wet/dry classification inference. It supports
multiple model loading mechanisms including:

- Local file paths (.pt files)
- Remote URLs with automatic download and caching

Key Features:
    - Automatic model downloading and caching from URLs
    - Smart model loading with fallback for PyTorch compatibility
    - Support for different model sources (local, remote)
    - GPU/CPU device detection and management

Main Functions:
    - get_model(): Universal model loader supporting multiple input types
    - download_and_cache_model(): Download and cache models from URLs
    - set_device(): Auto-detect and set appropriate device (GPU/CPU)

Cache Management:
    Models downloaded from URLs are cached locally in ~/.cml_wd_pytorch/models/
    to avoid repeated downloads, function checks if the file already exists.
    Cache can be managed with clear_model_cache() and list_cached_models() functions.

Example Usage:
    # Load from local path
    model = get_model("path/to/model.pt")

    # Load from URL (with automatic caching)
    model = get_model("https://example.com/model.pt")
"""

import hashlib
import urllib.request
from pathlib import Path

from pycomlink.processing.pytorch_utils.pytorch_utils import (
    load_model,
    set_device,
)


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
    model_filename = f"model_{url_hash}.pt"
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
        for file in cache_dir.glob("*"):
            file.unlink()
        print(f"Cleared cache at {cache_dir}")
    else:
        print(f"Cache directory {cache_dir} does not exist")


def list_cached_models(cache_dir="~/.cml_wd_pytorch/models", suffix=".pt"):
    """
    List all cached models.

    Args:
        cache_dir (str): Cache directory to list

    Returns:
        list: List of cached model files
    """
    cache_dir = Path(cache_dir).expanduser()
    if cache_dir.exists():
        return list(cache_dir.glob(suffix))
    return []


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
    # TODO: this is the function that also could in the future decide if to use tensorflow or pytorch
    # Determine input type and delegate to appropriate handler
    if model_path_or_url.startswith(("http://", "https://")):
        # It's a URL
        return _load_model_from_url(model_path_or_url, force_download)
    elif model_path_or_url.endswith(".pt") or "/" in model_path_or_url:
        # It's a local model path
        return _load_model_from_local_path(model_path_or_url)
    elif model_path_or_url.endswith(".pth") or model_path_or_url.endswith(".pt2"):
        # It's a legacy model path
        raise Exception(
            ".pth and .pt2 models are currently not supported. Please convert to .pt format using torch.jit.script()."
        )
    else:
        # It's neither url, nor path
        raise Exception(
            f"Provided string: '{model_path_or_url}' , is neither directory path, nor web url"
        )
