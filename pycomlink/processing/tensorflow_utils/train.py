import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils import resample


def split_cml_ids(cml_ids, n_train):
    """
    Split a list of CML IDs into training and validation sets.

    Parameters
    ----------
    cml_ids : list or np.ndarray
        List of CML IDs (strings or integers).
    n_train : int
        Number of IDs to include in training set.

    Returns
    -------
    train_ids : list
        List of training CML IDs.
    val_ids : list
        List of validation CML IDs.
    """
    cml_ids = list(cml_ids)  # ensure list
    train_ids = cml_ids[:n_train]
    val_ids = cml_ids[n_train:]
    return train_ids, val_ids

def create_rain_labels(ds, rainrate_var="R_radolan", threshold=0.1):
    """
    Create binary rain labels from a continuous rain rate variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the rain rate variable.
    rainrate_var : str
        Name of the rain rate variable in the dataset.
    threshold : float
        Threshold above which rain is considered present.

    Returns
    -------
    xarray.DataArray
        Binary rain label (1 if rain > threshold, else 0).
    """
    return (ds[rainrate_var] > threshold).astype(np.int8)

def create_lstm_samples(ds, rain, cml_ids, interp_vars, seq_len=4, rainrate_var="rr"):
    """
    Create LSTM input sequences and corresponding labels from a dataset.

    Returns:
        X, y, ids, rainrates
    """
    X, y, ids, rainrates = [], [], [], []

    for cml in cml_ids:
        try:
            ds_cml = ds.sel(cml_id=cml)
            rain_cml = rain.sel(cml_id=cml)

            inputs = np.stack([ds_cml[var].values for var in interp_vars], axis=-1)
            labels = rain_cml.values
            rainrate_vals = ds_cml[rainrate_var].values
        except KeyError:
            continue

        if len(inputs) < seq_len:
            continue

        for t in range(seq_len - 1, len(inputs)):
            x_seq = inputs[t - seq_len + 1:t + 1]
            y_label = labels[t]
            rr = rainrate_vals[t]

            if np.all(np.isfinite(x_seq)) and np.isfinite(y_label) and np.isfinite(rr):
                X.append(x_seq)
                y.append(y_label)
                ids.append(cml)
                rainrates.append(rr)

    return np.array(X), np.array(y), np.array(ids), np.array(rainrates)


def scale_features(X_train_seq, X_val_seq, interp_vars):
    """
    Standardize features per variable using training data statistics.

    Returns:
        X_train_scaled, X_val_scaled, scalers
    """
    scalers = {}
    X_train_scaled = np.empty_like(X_train_seq)
    X_val_scaled = np.empty_like(X_val_seq)

    for i, var in enumerate(interp_vars):
        scaler = StandardScaler()
        train_feat = X_train_seq[:, :, i].reshape(-1, 1)
        val_feat = X_val_seq[:, :, i].reshape(-1, 1)

        X_train_scaled[:, :, i] = scaler.fit_transform(train_feat).reshape(X_train_seq.shape[0],            X_train_seq.shape[1])
        X_val_scaled[:, :, i] = scaler.transform(val_feat).reshape(X_val_seq.shape[0],                      X_val_seq.shape[1])

        scalers[var] = scaler

    return X_train_scaled, X_val_scaled, scalers


def balance_lstm_data(X, y, interp_vars, seq_len=4, method="smote", random_state=42):
    """
    Balance time-series LSTM data using SMOTE, ADASYN, or repetition.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (samples, seq_len, features).
    y : np.ndarray
        Labels array of shape (samples,).
    interp_vars : list
        List of variable names (used to determine feature count).
    seq_len : int
        Sequence length (used for reshaping).
    method : str
        Balancing method. Options: "smote", "adasyn", "repeat".
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_bal : np.ndarray
        Balanced input array, reshaped to (samples, seq_len, features).
    y_bal : np.ndarray
        Balanced label array.
    """
    n_features = len(interp_vars)

    if method in ("smote", "adasyn"):
        # Flatten for oversampling: (samples, seq_len * features)
        X_2d = X.reshape((X.shape[0], seq_len * n_features))

        if method == "smote":
            sampler = SMOTE(random_state=random_state)
        elif method == "adasyn":
            sampler = ADASYN(random_state=random_state)

        X_bal_2d, y_bal = sampler.fit_resample(X_2d, y)
        X_bal = X_bal_2d.reshape((-1, seq_len, n_features))

    elif method == "repeat":
        # Identify class indices
        class_0 = np.where(y == 0)[0]
        class_1 = np.where(y == 1)[0]

        if len(class_0) > len(class_1):
            majority_idx, minority_idx = class_0, class_1
        else:
            majority_idx, minority_idx = class_1, class_0

        # Upsample minority class
        minority_upsampled_idx = resample(
            minority_idx,
            replace=True,
            n_samples=len(majority_idx),
            random_state=random_state
        )

        combined_idx = np.concatenate([majority_idx, minority_upsampled_idx])
        np.random.shuffle(combined_idx)

        X_bal = X[combined_idx]
        y_bal = y[combined_idx]

    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'smote', 'adasyn', or 'repeat'.")

    return X_bal, y_bal


def create_tf_dataset(X, y, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
