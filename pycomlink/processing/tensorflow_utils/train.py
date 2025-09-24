import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils import resample

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

def balance_data(X, y, interp_vars, seq_len=4, method="smote", random_state=42):
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
