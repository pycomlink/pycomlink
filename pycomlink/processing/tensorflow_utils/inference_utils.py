# pycomlink/processing/inference_utils.py

# Standard library imports
import urllib.request
from pathlib import Path

# Third-party imports
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

#         attrs={"description": "Predicted probabilities from CNN model", "source": "Model Trained on Czechia D"}----------------------
# TensorFlow Component Accessors
# -----------------------
from pycomlink.processing.tensorflow_utils.lazy_tf import get_tf


def get_model_from_json():
    """Get TensorFlow Keras model_from_json function."""
    return get_tf().keras.models.model_from_json


def get_optimizer_class():
    """Get TensorFlow Keras SGD optimizer class."""
    return get_tf().keras.optimizers.SGD


# -----------------------
# Core Functions
# -----------------------
def create_samples(ds, cml_ids, input_vars, seq_len=180, time_dim="time"):
    """
    Create time series samples from CML dataset for model inference.

    Args:
        ds: xarray Dataset containing CML data
        cml_ids: List of CML IDs to process
        input_vars: List of variable names to use as features
        seq_len: Length of time sequences (default: 180)
        time_dim: Name of the time dimension (default: "time")

    Returns:
        tuple: (X, ids, time_idx) where X is the feature array,
               ids are the CML IDs, and time_idx are the time indices
    """
    X, ids, time_idx = [], [], []

    for cml in cml_ids:
        try:
            ds_cml = ds.sel(cml_id=cml)
            inputs = np.stack(
                [ds_cml[var].values.astype(np.float32) for var in input_vars], axis=-1
            )
        except KeyError:
            continue

        if len(inputs) < seq_len:
            continue

        for t in range(seq_len - 1, len(inputs)):
            x_seq = inputs[t - seq_len + 1 : t + 1]
            if np.all(np.isfinite(x_seq)):
                X.append(x_seq)
                ids.append(cml)
                time_idx.append(t)

    return np.array(X, dtype=np.float32), np.array(ids), np.array(time_idx)


def scale_features(input_seq, input_vars):
    """
    Scale input features using StandardScaler for each variable independently.

    Args:
        input_seq: Input sequence array of shape (n_samples, seq_len, n_features)
        input_vars: List of variable names (used for feature dimension)

    Returns:
        numpy.ndarray: Scaled input sequence with same shape as input
    """
    input_scaled = np.empty_like(input_seq, dtype=np.float32)

    for i in range(len(input_vars)):
        scaler = StandardScaler()
        feat = input_seq[:, :, i].reshape(-1, 1)
        input_scaled[:, :, i] = scaler.fit_transform(feat).reshape(
            input_seq.shape[0], input_seq.shape[1]
        )

    return input_scaled


def is_url(path):
    """
    Check if a path is a URL.

    Args:
        path: String path to check

    Returns:
        bool: True if path is a URL, False otherwise
    """
    return path.startswith("http://") or path.startswith("https://")


def download_and_cache(url, cache_dir, force_download=False):
    """
    Download a file from URL and cache it locally.

    Args:
        url: URL to download from
        cache_dir: Directory to cache the file
        force_download: If True, re-download even if cached file exists

    Returns:
        Path: Path to the cached file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    cached_path = cache_dir / filename

    if not cached_path.exists() or force_download:
        print(f"[↓] Downloading {filename} from {url}")
        urllib.request.urlretrieve(url, cached_path)
    else:
        print(f"[→] Using cached file: {cached_path}")

    return cached_path


def resolve_model_paths(
    json_source, weights_source, cache_dir="model_cnn", force_download=False
):
    """
    Resolve model file paths, downloading from URLs if necessary.

    Args:
        json_source: Path or URL to model JSON file
        weights_source: Path or URL to model weights file
        cache_dir: Directory for caching downloaded files (default: "model_cnn")
        force_download: If True, re-download even if cached files exist

    Returns:
        tuple: (json_path, weights_path) as strings
    """
    cache_dir = Path.cwd() / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    json_path = (
        download_and_cache(json_source, cache_dir, force_download)
        if is_url(json_source)
        else Path(json_source)
    )
    weights_path = (
        download_and_cache(weights_source, cache_dir, force_download)
        if is_url(weights_source)
        else Path(weights_source)
    )

    return str(json_path), str(weights_path)


def load_model_from_local(
    json_path=None,
    weights_path=None,
    lr=0.05,
    loss="binary_crossentropy",
    optimizer=None,
):
    """
    Load a TensorFlow model from local JSON and weights files.

    Args:
        json_path: Path to model architecture JSON file
        weights_path: Path to model weights file
        lr: Learning rate for optimizer (default: 0.05)
        loss: Loss function (default: "binary_crossentropy")
        optimizer: Custom optimizer, if None uses SGD with default parameters

    Returns:
        tensorflow.keras.Model: Compiled TensorFlow model
    """
    with open(json_path, "r") as f:
        model_json = f.read()

    model = get_model_from_json()(model_json)
    model.load_weights(weights_path)

    if optimizer is None:
        optimizer = get_optimizer_class()(
            learning_rate=lr, decay=1e-3, momentum=0.9, nesterov=True
        )

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    print("[✓] Model loaded and compiled.")
    return model


def run_inference(model, model_input, batch_size=128):
    """
    Run model inference on input data.

    Args:
        model: Compiled TensorFlow model
        model_input: Input data array
        batch_size: Batch size for prediction (default: 128)

    Returns:
        numpy.ndarray: Flattened prediction probabilities as float32
    """
    y_prob = model.predict(
        model_input.astype(np.float32), batch_size=batch_size, verbose=1
    )
    return np.ravel(y_prob).astype(np.float32)


def store_predictions(
    ds,
    model_prob,
    cml_ids,
    time_indices,
    model_name="model",
    var_name=None,
    time_dim="time",
    cml_dim="cml_id",
    cml_ids_to_use=None,
) -> xr.DataArray:
    """
    Store model predictions in an xarray DataArray aligned with the original dataset.

    Args:
        ds: Original xarray Dataset for dimension/coordinate reference
        model_prob: Array of model prediction probabilities
        cml_ids: Array of CML IDs corresponding to predictions
        time_indices: Array of time indices corresponding to predictions
        model_name: Name identifier for the model (default: "model")
        var_name: Variable name for the DataArray (default: f"cnn_prob_{model_name}")
        time_dim: Name of time dimension (default: "time")
        cml_dim: Name of CML dimension (default: "cml_id")
        cml_ids_to_use: Subset of CML IDs to process (default: all unique CML IDs)

    Returns:
        xarray.DataArray: Predictions organized by time and CML dimensions
    """
    model_prob = np.asarray(model_prob, dtype=np.float32).ravel()
    cml_ids = np.asarray(cml_ids).ravel()
    time_indices = np.asarray(time_indices).ravel()

    if (
        model_prob.shape[0] != cml_ids.shape[0]
        or model_prob.shape[0] != time_indices.shape[0]
    ):
        raise ValueError("Length of model_prob, cml_ids, and time_indices must match.")

    all_cml_ids = ds[cml_dim].values
    all_time = ds[time_dim].values
    n_time, n_cml = len(all_time), len(all_cml_ids)

    if cml_ids_to_use is None:
        cml_ids_to_use = np.unique(cml_ids)

    if var_name is None:
        var_name = f"cnn_prob_{model_name}"

    cnn_storage = np.full((n_time, n_cml), np.nan, dtype=np.float32)

    for cml_id in tqdm(
        cml_ids_to_use, desc=f"Storing predictions ({model_name})", unit="CML"
    ):
        mask = cml_ids == cml_id
        if not np.any(mask):
            continue

        cml_idx = np.where(all_cml_ids == cml_id)[0]
        if cml_idx.size == 0:
            continue

        t_idx = time_indices[mask]
        p_vals = model_prob[mask]

        valid = t_idx < n_time
        cnn_storage[t_idx[valid], cml_idx[0]] = p_vals[valid]

    return xr.DataArray(
        data=cnn_storage,
        dims=(time_dim, cml_dim),
        coords={time_dim: all_time, cml_dim: all_cml_ids},
        name=var_name,
        attrs={
            "description": "Predicted probabilities from CNN model",
            "source": "Model Trained on Czechia D",
        },
    )
