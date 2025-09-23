import xarray as xr
import numpy as np
import pandas as pd
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout
)
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import metrics
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
from tqdm import tqdm
import requests
import tempfile
import hashlib
import urllib.request
from pathlib import Path
import requests
import tempfile
import shutil




######################################################################################################
# ✅ CNN Builder
def build_cnn(
    window,
    n_filters,
    kernel_size=3,
    n_fc_neurons=128,
    dropout=0.3,
    task='classification',
    n_features=1
):
    assert len(n_filters) == 6, "n_filters must contain 6 values."

    input1 = Input(shape=(window, n_features))
    x = input1

    # Block 1
    x = Conv1D(n_filters[0], kernel_size, padding='same', activation='relu')(x)
    x = Conv1D(n_filters[0], kernel_size, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Block 2
    x = Conv1D(n_filters[1], kernel_size, padding='same', activation='relu')(x)
    x = Conv1D(n_filters[1], kernel_size, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Block 3
    x = Conv1D(n_filters[2], kernel_size, padding='same', activation='relu')(x)
    x = Conv1D(n_filters[2], kernel_size, padding='same', activation='relu')(x)
    if window >= 32:
        x = MaxPooling1D(pool_size=2)(x)

    # Block 4
    x = Conv1D(n_filters[3], kernel_size, padding='same', activation='relu')(x)
    x = Conv1D(n_filters[3], kernel_size, padding='same', activation='relu')(x)
    if window >= 64:
        x = MaxPooling1D(pool_size=2)(x)

    # Block 5
    x = Conv1D(n_filters[4], kernel_size, padding='same', activation='relu')(x)
    x = Conv1D(n_filters[4], kernel_size, padding='same', activation='relu')(x)
    if window >= 96:
        x = MaxPooling1D(pool_size=2)(x)

    # Block 6
    x = Conv1D(n_filters[5], kernel_size, padding='same', activation='relu')(x)
    x = GlobalAveragePooling1D()(x)

    # Fully connected
    x = Dense(n_fc_neurons, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(n_fc_neurons, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(n_fc_neurons // 2, activation='relu')(x)
    x = Dropout(dropout)(x)

    # Output layer
    if task == 'classification':
        out = Dense(1, activation='sigmoid')(x)
    elif task == 'regression':
        out = Dense(1, activation='linear')(x)
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    return Model(inputs=input1, outputs=out)
######################################################################################################
# Custom regression metrics
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
######################################################################################################
def bias(y_true, y_pred):
    return K.mean(y_pred - y_true)
######################################################################################################
def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

######################################################################################################
def train_model(
    model,
    X_train,
    y_train=None,
    X_val=None,
    y_val=None,
    task='classification',
    batch_size=32,
    epochs=10,
    learning_rate=0.005,
    decay=1e-3,
    momentum=0.9,
    patience=5
):
    """
    Train a Keras model on either NumPy arrays or tf.data.Dataset.
    """

    # Select loss and metrics
    if task == 'classification':
        loss = 'binary_crossentropy'
        selected_metrics = [
            'accuracy',
            'mse',
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.FalseNegatives(name="fn")
        ]
        monitor_metric = 'val_precision'
        monitor_mode = 'max'

    elif task == 'regression':
        loss = tf.keras.losses.MeanSquaredError()
        selected_metrics = [
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            rmse,
            bias,
            r2_score
        ]
        monitor_metric = 'val_rmse'
        monitor_mode = 'min'

    else:
        raise ValueError("task must be either 'classification' or 'regression'")

    model.compile(
        loss=loss,
        optimizer=SGD(learning_rate=learning_rate, decay=decay, momentum=momentum, nesterov=True),
        metrics=selected_metrics
    )

    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        verbose=1,
        mode=monitor_mode,
        restore_best_weights=True
    )

    # Device detection
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"Training on: {device} | Task: {task}")

    # Fit depending on input type
    with tf.device(device):
        if isinstance(X_train, tf.data.Dataset):
            history = model.fit(
                X_train,
                validation_data=X_val,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=1
            )
        else:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=1
            )

    return history


######################################################################################################

def is_url(path):
    return path.startswith("http://") or path.startswith("https://")
######################################################################################################
def download_and_cache(url, cache_dir, force_download=False):
    """
    Download a file from a URL and cache it locally with its original filename.
    
    Parameters
    ----------
    url : str
        The URL of the file to download.
    cache_dir : str or Path
        Directory where the file should be saved.
    force_download : bool
        If True, force re-download even if file exists.

    Returns
    -------
    Path
        Path to the cached file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract filename from URL
    filename = url.split("/")[-1]
    cached_path = cache_dir / filename

    if not cached_path.exists() or force_download:
        print(f"[↓] Downloading {filename} from {url}")
        urllib.request.urlretrieve(url, cached_path)
        print(f"[✓] Saved to: {cached_path}")
    else:
        print(f"[→] Using cached file: {cached_path}")

    return cached_path
    
######################################################################################################
def resolve_model_paths(json_source, weights_source, cache_dir="model_cnn", force_download=False):
    """
    Resolves model architecture and weights paths from local files or URLs.

    If URLs are provided, downloads and caches them into a persistent directory.

    Parameters
    ----------
    json_source : str
        Path or URL to the model JSON file.
    weights_source : str
        Path or URL to the weights H5 file.
    cache_dir : str
        Directory to use for caching downloaded files (default: 'model_cnn').
    force_download : bool
        If True, re-download even if files exist.

    Returns
    -------
    json_path : Path
        Resolved path to the model JSON file.
    weights_path : Path
        Resolved path to the model weights file.
    """
    # Ensure cache directory exists
    cache_dir = Path(os.getcwd()) / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[🗂️] Using model cache directory: {cache_dir}")

    # Resolve JSON path
    if is_url(json_source):
        json_path = download_and_cache(json_source, cache_dir=cache_dir, force_download=force_download)
    else:
        json_path = Path(json_source)

    # Resolve weights path
    if is_url(weights_source):
        weights_path = download_and_cache(weights_source, cache_dir=cache_dir, force_download=force_download)
    else:
        weights_path = Path(weights_source)

    return str(json_path), str(weights_path)

######################################################################################################
def load_model_from_local(
    json_path=None,
    weights_path=None,
    full_keras=None,
    lr=0.05,
    loss="binary_crossentropy",
    optimizer=None
):
    """
    Load a Keras model from local files.
    Supports:
    - Full model file (.full.keras)
    - Architecture (.json) + Weights (.h5)

    Parameters
    ----------
    json_path : str or Path, optional
        Path to model architecture file (.json)
    weights_path : str or Path, optional
        Path to weights file (.h5)
    full_keras : str or Path, optional
        Path to full model file (.full.keras)
    lr : float, optional
        Learning rate used if optimizer is not provided.
    loss : str or callable, optional
        Loss function to use for model compilation.
    optimizer : tf.keras.optimizers.Optimizer, optional
        Optimizer instance. If None, defaults to SGD with given learning rate.
    """
    # Case 1: Full model path (PROD mode)
    if full_keras is not None and full_keras.endswith(".full.keras") and weights_path is None and json_path is None:
        model = tf.keras.models.load_model(full_keras)
        print(f"[✓] Full PROD model loaded from: {full_keras}")
        return model

    # Case 2: JSON + weights (RESTORE mode)
    with open(json_path, "r") as f:
        model_json = f.read()

    model = model_from_json(model_json)
    model.load_weights(weights_path)

    if optimizer is None:
        optimizer = SGD(learning_rate=lr, decay=1e-3, momentum=0.9, nesterov=True)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    print(f"[✓] Model loaded from architecture + weights (RESTORE mode).")
    return model

######################################################################################################
def run_inference(model,X_input, threshold=0.1, lr=0.05, batch_size=128, force_download=False):
    """
    Run inference using a Keras model from local paths or URLs (JSON + H5).
    If URLs are used, the files are cached into a visible local 'model_cache/' directory.
    """

    y_prob = np.ravel(model.predict(X_input, batch_size=batch_size, verbose=1))
    y_pred = y_prob > threshold

    print(f"[✓] Prediction Completed — {len(y_pred)} Samples.")
    return y_prob, y_pred

######################################################################################################
def store_predictions(
    merged: xr.Dataset,
    y_prob: np.ndarray,
    cml_ids: np.ndarray,
    
    model_name: str = "model",
    var_name: str = None,
    time_dim: str = "time",
    cml_dim: str = "cml_id",
    cml_ids_to_use: np.ndarray | None = None,
    time_index: np.ndarray | None = None,
):
    """
    Store CNN probabilities into an xarray.Dataset for a single model.

    Parameters
    ----------
    merged : xr.Dataset
        Dataset to write into.
    y_prob : np.ndarray
        Flattened prediction probabilities.
    cml_ids : np.ndarray
        CML ID for each sample in y_prob (must match length).
    model_name : str
        Name of the model, used for annotation.
    var_name : str, optional
        Name of the variable to store in the dataset.
        If None, will default to 'cnn_prob_<model_name>'.
    time_dim : str
        Name of the time dimension in the dataset.
    cml_dim : str
        Name of the CML ID dimension in the dataset.
    cml_ids_to_use : np.ndarray, optional
        Only store predictions for this subset of CMLs (default: all in cml_ids).
    time_index : np.ndarray, optional
        Use this as the time index for alignment (default: merged[time_dim].values)
    
    Returns
    -------
    merged : xr.Dataset
        Dataset with added variable containing the CNN predictions.
    """
    y_prob = np.asarray(y_prob).ravel()
    cml_ids = np.asarray(cml_ids).ravel()
    assert y_prob.shape[0] == cml_ids.shape[0], "Mismatch in y_prob and cml_ids length."

    all_cml_ids = merged[cml_dim].values
    all_time = merged[time_dim].values if time_index is None else time_index
    n_time, n_cml = len(all_time), len(all_cml_ids)

    if cml_ids_to_use is None:
        cml_ids_to_use = np.unique(cml_ids)

    if var_name is None:
        var_name = f"cnn_prob_{model_name}"

    # Create empty storage for this model
    cnn_storage = np.full((n_time, n_cml), np.nan, dtype=np.float32)

    for cml_id in tqdm(cml_ids_to_use, desc=f"Storing predictions ({model_name})", unit="CML"):
        mask = (cml_ids == cml_id)
        if not np.any(mask):
            continue

        preds = y_prob[mask]
        L = min(len(preds), n_time)
        series = pd.Series(preds[:L], index=all_time[:L]).reindex(all_time)

        cml_idx = np.where(all_cml_ids == cml_id)[0]
        if cml_idx.size == 0:
            continue

        cnn_storage[:, cml_idx[0]] = series.values.astype(np.float32)

    merged[var_name] = ((time_dim, cml_dim), cnn_storage)
    merged[var_name].attrs.update({
        "description": f"Predicted probabilities from CNN model '{model_name}'",
        "source": f"Inference result for model '{model_name}'"
    })

    return merged


######################################################################################################
def save_model(model, name_prefix="cnn_model", mode="PROD"):
    """
    Saves a Keras model to disk in different modes:
    - 'PROD'     : full restore (full model + arch + weights)
    - 'RESTORE'  : architecture (.json) + weights (.h5)
    - 'MIN'      : only weights (.h5)

    Parameters
    ----------
    model : tf.keras.Model
        The trained Keras model to save.
    name_prefix : str
        Prefix for saved files.
    mode : str
        One of ['PROD', 'RESTORE', 'MIN'].
    """
    mode = mode.upper()
    valid_modes = ["PROD", "RESTORE", "MIN"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}.")

    # Create directory
    save_dir = os.path.join(os.getcwd(), "Models")
    os.makedirs(save_dir, exist_ok=True)

    saved_files = []

    if mode in ["PROD", "RESTORE"]:
        # Save architecture
        json_path = os.path.join(save_dir, f"{name_prefix}.json")
        with open(json_path, "w") as json_file:
            json_file.write(model.to_json())
        saved_files.append(json_path)

        # Save weights
        weights_path = os.path.join(save_dir, f"{name_prefix}.weights.h5")
        model.save_weights(weights_path)
        saved_files.append(weights_path)

    if mode == "PROD":
        # Save full model (weights + architecture + optimizer)
        full_path = os.path.join(save_dir, f"{name_prefix}.full.keras")
        model.save(full_path, include_optimizer=True)
        saved_files.append(full_path)

    if mode == "MIN":
        # Save only weights
        weights_path = os.path.join(save_dir, f"{name_prefix}.weights.h5")
        model.save_weights(weights_path)
        saved_files.append(weights_path)

    print(f"[✓] Model saved in '{mode}' mode to '{save_dir}':")
    for f in saved_files:
        print(f" - {os.path.basename(f)}")

