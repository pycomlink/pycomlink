import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
import pkg_resources

def get_test_data_path():
    return pkg_resources.resource_filename("pycomlink", "/processing/wet_dry/cnn_model_files")

modelh5_fn = str(get_test_data_path() + "/model_2020.002.180m.h5")
modeljson_fn = str(get_test_data_path() + "/model_2020.002.180m.json")

def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(
        a, shape=shape, strides=strides, writeable=False
    )


def cnn_wet_dry(trsl_channel_1, trsl_channel_2, threshold, batch_size=100, verbose=0):
    """
    Wet dry classification using the CNN based on channel 1 and channel 2 of a CML

    Parameters
    ----------
    trsl_channel_1 : iterable of float
         Time series of received signal level of channel 1
    trsl_channel_2 : iterable of float
         Time series of received signal level of channel 2
    threshold : float
         Threshold between 0 and 1 which has to be surpassed to classifiy a period as 'wet'

    Returns
    -------
    iterable of int
        Time series of wet/dry classification

    Note
    ----
    Implementation of CNN method [1]_

    References
    ----------
    .. [1] Polz, J., Chwala, C., Graf, M., and Kunstmann, H.: Rain event detection in commercial microwave link
    attenuation data using convolutional neural networks, Atmos. Meas. Tech., 13, 3835–3853,
    https://doi.org/10.5194/amt-13-3835-2020, 2020.
    """

    # load json and create model
    json_file = open(modeljson_fn, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(modelh5_fn)
    model.compile(
        loss="binary_crossentropy",
        optimizer=SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True),
        metrics=["accuracy"],
    )
    #################
    # Normalization #
    #################

    df = pd.DataFrame()
    df["trsl1"] = trsl_channel_1
    df["trsl2"] = trsl_channel_2
    idx = df.index
    df["med1"] = df["trsl1"].rolling(72 * 60, min_periods=2 * 60, center=False).median()
    df["med2"] = df["trsl2"].rolling(72 * 60, min_periods=2 * 60, center=False).median()
    df["trsl1"] = df["trsl1"].sub(df["med1"])
    df["trsl2"] = df["trsl2"].sub(df["med2"])
    df = df.dropna()
    df = df.reindex(idx, fill_value=-9999)

    #########################
    # generate numpy arrays #
    #########################

    X_fts = np.moveaxis(
        np.array(
            [
                _rolling_window(df["trsl1"].values, 180),
                _rolling_window(df["trsl2"].values, 180),
            ]
        ),
        0,
        -1,
    )
    cnn_pred = np.ravel(model.predict(X_fts, batch_size=batch_size, verbose=verbose))

    for i in range(len(cnn_pred)):
        if (
                -9999 in df["trsl1"].values[i - 151: i + 31]
                or -9999 in df["trsl2"].values[i - 151: i + 31]
        ):
            cnn_pred[i] = np.nan

    df["prediction"] = np.concatenate(
        (np.repeat(np.nan, 150), cnn_pred, np.repeat(np.nan, 29)), axis=0
    )

    df = df.reindex(idx, fill_value=np.nan)
    return df.prediction.values > threshold
