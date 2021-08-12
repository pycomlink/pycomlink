import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.compat.v1.keras.backend import set_session
import pkg_resources


# Limit GPU memory usage to avoid processes to run out of memory.
# For a list of processes blocking GPU memory on an nvidia GPU type 'nvidia-smi' in the terminal.
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.visible_device_list = "0"
set_session(tf.compat.v1.Session(config=config))


def get_model_file_path():
    return pkg_resources.resource_filename(
        "pycomlink", "/processing/anomaly_detection/cnn_model"
    )


# load model
modelh5_fn = str(get_model_file_path() + "/model_anomaly_detection_60.h5")
# modelh5_fn = '/pd/home/glawion-l/pycomlink-1/pycomlink/processing/anomaly_detection/cnn_model/model_anomaly_detection_60.h5'
model = tf.keras.models.load_model(modelh5_fn, compile=False)


def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(
        a, shape=shape, strides=strides, writeable=False
    )


def cnn_anomaly_detection(
    trsl_channel_1,
    trsl_channel_2,
    batch_size=100,
    verbose=0,
):
    """
     Anomaly detection using the CNN based on channel 1 and channel 2 of a CML

    Parameters
    ----------
    trsl_channel_1 : iterable of float
         Time series of received signal level of channel 1
    trsl_channel_2 : iterable of float
         Time series of received signal level of channel 2
    batch_size : int
        Batch size for parallel computing. Set to 1 when using a CPU!
    verbose : int
        Toggles Keras text output during prediction. Default is off.


    Returns
    -------
    iterable of int
        Time series of anomaly detection prognosis

    Note
    ----


    References
    ----------
    .. [1]  Polz, J., Schmidt, L., Glawion, L., Graf, M., Werner, C., Chwala, C., Mollenhauer, H., Rebmann, C., Kunstmann, H., and Bumberger, J.:
    Supervised and unsupervised machine-learning for automated quality control of environmental sensor data, EGU General Assembly 2021, online,
    19–30 Apr 2021, EGU21-14485,
    https://doi.org/10.5194/egusphere-egu21-14485, 2021.

    """
    df = pd.DataFrame()
    df["trsl1"] = trsl_channel_1.copy()
    df["trsl2"] = trsl_channel_2.copy()
    df["med1"] = df["trsl1"].rolling(60 * 12, min_periods=3 * 60, center=False).median()
    df["med2"] = df["trsl2"].rolling(60 * 12, min_periods=3 * 60, center=False).median()
    df["trsl1"] = df["trsl1"].sub(df["med1"])
    df["trsl2"] = df["trsl2"].sub(df["med2"])

    df = df.fillna(value=-9999)

    x_fts = np.moveaxis(
        np.array(
            [
                _rolling_window(df["trsl1"].values, 60),
                _rolling_window(df["trsl2"].values, 60),
            ]
        ),
        source=0,
        destination=-1,
    )

    cnn_pred = np.ravel(model.predict(x_fts, batch_size=batch_size, verbose=verbose))

    for i in range(len(cnn_pred)):
        if (
            -9999 in df["trsl1"].values[i : i + 60]
            or -9999 in df["trsl2"].values[i : i + 60]
        ):
            cnn_pred[i] = np.nan

    df["prediction"] = np.concatenate((np.repeat(np.nan, 59), cnn_pred), axis=0)

    return df.prediction.values.reshape(len(trsl_channel_1))
