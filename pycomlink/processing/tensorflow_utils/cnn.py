
import os
import sys
import subprocess
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import requests
import tempfile
import hashlib
import urllib.request
from pathlib import Path
import shutil

# -----------------------
# TensorFlow Lazy Handling
# -----------------------
tf = None

def get_tf():
    global tf
    if tf is None:
        tf = ensure_tensorflow_installed()
        print_tf_device_info(tf)
    return tf

def ensure_tensorflow_installed():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        print("[!] TensorFlow not found. Installing...")
        has_gpu = detect_gpu_without_tf()
        package = "tensorflow[and-cuda]" if has_gpu else "tensorflow"
        print(f"[→] Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        import tensorflow as tf
        return tf

def detect_gpu_without_tf():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        print("[✓] GPU detected via nvidia-smi.")
        return True
    except Exception:
        print("[ℹ️] No GPU detected (nvidia-smi not found or failed).")
        return False

def print_tf_device_info(tf):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[✅] TensorFlow is using GPU: {len(gpus)} device(s) found.")
        for gpu in gpus:
            print(f"    → {gpu.name}")
    else:
        print("[⚠️] TensorFlow is using CPU. No GPU detected.")

# -----------------------
# TF Component Accessors
# -----------------------
def get_model_from_json():
    return get_tf().keras.models.model_from_json

def get_model_class():
    return get_tf().keras.models.Model

def get_layers():
    return get_tf().keras.layers

def get_callbacks():
    return get_tf().keras.callbacks

def get_optimizer_class():
    return get_tf().keras.optimizers.SGD

def get_metrics():
    return get_tf().keras.metrics

def get_losses():
    tf = get_tf()
    return tf.keras.losses.MeanSquaredError, tf.keras.losses.MeanAbsoluteError

def get_backend():
    return get_tf().keras.backend

# -----------------------
# Core CNN Utils
# -----------------------

def split_cml_ids(cml_ids, n_train):
    cml_ids = list(cml_ids)
    train_ids = cml_ids[:n_train]
    val_ids = cml_ids[n_train:]
    return train_ids, val_ids

def create_samples(ds, cml_ids, input_vars, seq_len=4):
    X, ids = [], []

    for cml in cml_ids:
        try:
            ds_cml = ds.sel(cml_id=cml)
            inputs = np.stack([ds_cml[var].values for var in input_vars], axis=-1)
        except KeyError:
            continue

        if len(inputs) < seq_len:
            continue

        for t in range(seq_len - 1, len(inputs)):
            x_seq = inputs[t - seq_len + 1 : t + 1]

            if np.all(np.isfinite(x_seq)):
                X.append(x_seq)
                ids.append(cml)

    return np.array(X), np.array(ids)

def scale_features(input_seq, input_vars):
    scalers = {}
    input_scaled = np.empty_like(input_seq)

    for i, var in enumerate(input_vars):
        scaler = StandardScaler()
        train_feat = input_seq[:, :, i].reshape(-1, 1)
        input_scaled[:, :, i] = scaler.fit_transform(train_feat).reshape(input_seq.shape[0], input_seq.shape[1])
        scalers[var] = scaler

    return input_scaled, scalers

def is_url(path):
    return path.startswith("http://") or path.startswith("https://")

def download_and_cache(url, cache_dir, force_download=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    cached_path = cache_dir / filename

    if not cached_path.exists() or force_download:
        print(f"[↓] Downloading {filename} from {url}")
        urllib.request.urlretrieve(url, cached_path)
        print(f"[✓] Saved to: {cached_path}")
    else:
        print(f"[→] Using cached file: {cached_path}")

    return cached_path

def resolve_model_paths(json_source, weights_source, cache_dir="model_cnn", force_download=False):
    cache_dir = Path(os.getcwd()) / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[🗂️] Using model cache directory: {cache_dir}")

    json_path = download_and_cache(json_source, cache_dir, force_download) if is_url(json_source) else Path(json_source)
    weights_path = download_and_cache(weights_source, cache_dir, force_download) if is_url(weights_source) else Path(weights_source)

    return str(json_path), str(weights_path)

def load_model_from_local(json_path=None, weights_path=None, full_keras=None, lr=0.05, loss="binary_crossentropy", optimizer=None):
    tf = get_tf()

    if full_keras is not None and full_keras.endswith(".full.keras") and weights_path is None and json_path is None:
        model = tf.keras.models.load_model(full_keras)
        print(f"[✓] Full PROD model loaded from: {full_keras}")
        return model

    model_from_json = get_model_from_json()
    with open(json_path, "r") as f:
        model_json = f.read()

    model = model_from_json(model_json)
    model.load_weights(weights_path)

    if optimizer is None:
        optimizer = get_optimizer_class()(learning_rate=lr, decay=1e-3, momentum=0.9, nesterov=True)

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    print(f"[✓] Model loaded from architecture + weights (RESTORE mode).")
    return model

def run_inference(model, model_input, threshold=0.1, batch_size=128):
    y_prob = np.ravel(model.predict(model_input, batch_size=batch_size, verbose=1))
    y_pred = y_prob > threshold
    print(f"[✓] Prediction Completed — {len(y_pred)} Samples.")
    return y_prob, y_pred

def store_predictions(
    ds,
    model_prob,
    cml_ids,
    model_name="model",
    var_name=None,
    time_dim="time",
    cml_dim="cml_id",
    cml_ids_to_use=None,
    time_index=None
) -> xr.DataArray:
    """
    Return an xarray.DataArray with CNN prediction probabilities (not modifying ds).
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing time and CML dimensions (used for coords).
    model_prob : np.ndarray
        Prediction probabilities (flattened).
    cml_ids : np.ndarray
        CML ID for each prediction (must match length of model_prob).
    model_name : str
        Name of the model (used for metadata and default variable name).
    var_name : str, optional
        Name of the returned variable. Defaults to 'cnn_prob_<model_name>'.
    time_dim : str
        Name of the time dimension in ds.
    cml_dim : str
        Name of the CML dimension in ds.
    cml_ids_to_use : np.ndarray, optional
        Subset of CMLs to include.
    time_index : np.ndarray, optional
        Optional time index to override ds[time_dim].values.
        
    Returns
    -------
    xr.DataArray
        Prediction array with dimensions (time, cml_id).
    """
    model_prob = np.asarray(model_prob).ravel()
    cml_ids = np.asarray(cml_ids).ravel()
    assert model_prob.shape[0] == cml_ids.shape[0], "Mismatch in model_prob and cml_ids length."

    all_cml_ids = ds[cml_dim].values
    all_time = ds[time_dim].values if time_index is None else time_index
    n_time, n_cml = len(all_time), len(all_cml_ids)

    if cml_ids_to_use is None:
        cml_ids_to_use = np.unique(cml_ids)

    if var_name is None:
        var_name = f"cnn_prob_{model_name}"

    # Empty array for predictions
    cnn_storage = np.full((n_time, n_cml), np.nan, dtype=np.float32)

    for cml_id in tqdm(cml_ids_to_use, desc=f"Storing predictions ({model_name})", unit="CML"):
        mask = (cml_ids == cml_id)
        if not np.any(mask):
            continue

        preds = model_prob[mask]
        L = min(len(preds), n_time)
        series = pd.Series(preds[:L], index=all_time[:L]).reindex(all_time)

        cml_idx = np.where(all_cml_ids == cml_id)[0]
        if cml_idx.size == 0:
            continue

        cnn_storage[:, cml_idx[0]] = series.values.astype(np.float32)

    # Return as xarray.DataArray
    result = xr.DataArray(
        data=cnn_storage,
        dims=(time_dim, cml_dim),
        coords={time_dim: all_time, cml_dim: all_cml_ids},
        name=var_name,
        attrs={
            "description": f"Predicted probabilities from CNN model '{model_name}'",
            "source": f"Inference result for model '{model_name}'"
        }
    )

    return result

