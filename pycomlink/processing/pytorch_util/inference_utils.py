"""
Inference utilities for CML wet/dry classification models.

This module provides utility functions for loading, caching, and managing PyTorch models
used for Commercial Microwave Link (CML) wet/dry classification inference. It supports
multiple model loading mechanisms including:

- Local file paths (.pth files)
- Remote URLs with automatic download and caching
- Run IDs from training results directories

Key Features:
    - Automatic model downloading and caching from URLs
    - Smart model loading with fallback for PyTorch compatibility
    - Configuration management with flexible config loading
    - Support for different model sources (local, remote, run-based)
    - GPU/CPU device detection and management

Main Functions:
    - get_model(): Universal model loader supporting multiple input types
    - load_model(): Load PyTorch model from local path
    - download_and_cache_model(): Download and cache models from URLs
    - load_config(): Load YAML configuration files
    - set_device(): Auto-detect and set appropriate device (GPU/CPU)

Cache Management:
    Models downloaded from URLs are cached locally in ~/.cml_wd_pytorch/models/
    to avoid repeated downloads. Cache can be managed with clear_model_cache()
    and list_cached_models() functions.

Example Usage:
    # Load from local path
    model, config = get_model("path/to/model.pth")

    # Load from URL (with automatic caching)
    model, config = get_model("https://example.com/model.pth")

    # Load from training run ID
    model, config = get_model("2025-01-15_12-34-56abc123")
"""

import hashlib
import os
import urllib.request
from pathlib import Path

import torch
import yaml



# -------------------- Local temporary solution ------------------------------
# TEMPORARY: import cnn model, then move this into function and load it from url
import sys, os
# TODO: cnn model will be loaded from url given as a func input
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join('C:/Users/lukas/Documents/OpenSense/temp_for_cnn_models/')))
from cnn_polz_pytorch_2025 import cnn
# ----------------------------------------------------------------------------




def set_device():
    """Auto-detect and return appropriate device (GPU/CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def load_config():
    """
    Load configuration from default config.yml file.

    Returns:
        dict: Configuration dictionary.
    """
    package_path = Path(os.path.abspath(__file__)).parent.parent.parent.absolute()
    config_path = str(Path(package_path) / "config" / "config.yml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


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


def _load_config_from_path(config_path):
    """Load config from specific path or use default."""
    if config_path is None:
        return load_config()
    else:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)


def _load_model_from_url(model_url, config_path=None, force_download=False):
    """Load model from URL by downloading and caching it."""
    device = set_device()

    # Download and cache the model
    model_path = download_and_cache_model(model_url, force_download=force_download)
    model = load_model(str(model_path), device)

    # Load config
    config = _load_config_from_path(config_path)

    return model, config


def _load_model_from_local_path(model_path, config_path=None):
    """Load model from local file path."""
    device = set_device()

    # Load the model
    model = load_model(model_path, device)

    # Load config
    config = _load_config_from_path(config_path)

    return model, config


def _load_model_from_run_id(run_id, config_path=None):
    """Load model from training run ID by finding latest model in results directory."""
    device = set_device()

    # Find the results directory
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

    # Load the model
    model = load_model(str(latest_model), device)

    # Load config from results directory (or fallback to provided/default)
    if config_path is None:
        config_file = results_dir / "config.yml"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            print(f"Using config from: {config_file}")
        else:
            print(f"Config file not found at {config_file}, using default config")
            config = load_config()
    else:
        config = _load_config_from_path(config_path)

    return model, config


def get_model(model_path_or_run_id_or_url, config_path=None, force_download=False):
    """
    Load a model from a local path, run_id, or URL.

    Args:
        model_path_or_run_id_or_url (str): Either a path to the trained PyTorch model, a run_id,
                                          or a URL to download the model from.
                                          If run_id, will look for model and config in results/{run_id}/
                                          If URL, will download and cache the model locally.
        config_path (str, optional): Path to config file. If None, uses default config location
                                    or looks for config in results/{run_id}/config.yml if run_id is provided.
        force_download (bool): Force re-download of model if it's a URL (default: False).

    Returns:
        tuple: (model, config) - The loaded PyTorch model and configuration dictionary.
    """
    # Determine input type and delegate to appropriate handler
    if model_path_or_run_id_or_url.startswith(("http://", "https://")):
        # It's a URL
        return _load_model_from_url(
            model_path_or_run_id_or_url, config_path, force_download
        )
    elif (
        model_path_or_run_id_or_url.endswith(".pth")
        or "/" in model_path_or_run_id_or_url
    ):
        # It's a local model path
        return _load_model_from_local_path(model_path_or_run_id_or_url, config_path)
    else:
        # It's a run_id
        return _load_model_from_run_id(model_path_or_run_id_or_url, config_path)
