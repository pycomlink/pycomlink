# pycomlink/processing/inference_utils.py

import xarray as xr
from pathlib import Path

# Standard library imports
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request


# Third-party imports
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm




# -----------------------
# TensorFlow Component Accessors
# -----------------------
from pycomlink.processing.tensorflow_utils.lazy_tf import get_tf



def get_model_from_json(): return get_tf().keras.models.model_from_json
def get_optimizer_class(): return get_tf().keras.optimizers.SGD

# -----------------------
# Core Functions
# -----------------------
def create_samples(ds, cml_ids, input_vars, seq_len=180, time_dim="time"):
    X, ids, time_idx = [], [], []

    for cml in cml_ids:
        try:
            ds_cml = ds.sel(cml_id=cml)
            inputs = np.stack([ds_cml[var].values.astype(np.float32) for var in input_vars], axis=-1)
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
    input_scaled = np.empty_like(input_seq, dtype=np.float32)

    for i in range(len(input_vars)):
        scaler = StandardScaler()
        feat = input_seq[:, :, i].reshape(-1, 1)
        input_scaled[:, :, i] = scaler.fit_transform(feat).reshape(input_seq.shape[0], input_seq.shape[1])
    
    return input_scaled

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
    else:
        print(f"[→] Using cached file: {cached_path}")
    
    return cached_path

def resolve_model_paths(json_source, weights_source, cache_dir="model_cnn", force_download=False):
    cache_dir = Path.cwd() / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    json_path = download_and_cache(json_source, cache_dir, force_download) if is_url(json_source) else Path(json_source)
    weights_path = download_and_cache(weights_source, cache_dir, force_download) if is_url(weights_source) else Path(weights_source)

    return str(json_path), str(weights_path)

def load_model_from_local(json_path=None, weights_path=None, lr=0.05, loss="binary_crossentropy", optimizer=None):
    tf = get_tf()
    with open(json_path, "r") as f:
        model_json = f.read()

    model = get_model_from_json()(model_json)
    model.load_weights(weights_path)

    if optimizer is None:
        optimizer = get_optimizer_class()(learning_rate=lr, decay=1e-3, momentum=0.9, nesterov=True)

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    print("[✓] Model loaded and compiled.")
    return model

def run_inference(model, model_input, batch_size=128):
    y_prob = model.predict(model_input.astype(np.float32), batch_size=batch_size, verbose=1)
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
    cml_ids_to_use=None
) -> xr.DataArray:
    model_prob = np.asarray(model_prob, dtype=np.float32).ravel()
    cml_ids = np.asarray(cml_ids).ravel()
    time_indices = np.asarray(time_indices).ravel()

    if model_prob.shape[0] != cml_ids.shape[0] or model_prob.shape[0] != time_indices.shape[0]:
        raise ValueError("Length of model_prob, cml_ids, and time_indices must match.")

    all_cml_ids = ds[cml_dim].values
    all_time = ds[time_dim].values
    n_time, n_cml = len(all_time), len(all_cml_ids)

    if cml_ids_to_use is None:
        cml_ids_to_use = np.unique(cml_ids)

    if var_name is None:
        var_name = f"cnn_prob_{model_name}"

    cnn_storage = np.full((n_time, n_cml), np.nan, dtype=np.float32)

    for cml_id in tqdm(cml_ids_to_use, desc=f"Storing predictions ({model_name})", unit="CML"):
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
        attrs={"description": f"Predicted probabilities from CNN model", "source": f"Model Trained on Czechia D"}
    )
