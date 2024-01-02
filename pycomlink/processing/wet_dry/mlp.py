import numpy as np
from numpy.lib.stride_tricks import sliding_window_view 
import tensorflow as tf
import pkg_resources
import pandas as pd

def get_model_file_path():
    return pkg_resources.resource_filename(
        "pycomlink", "/processing/wet_dry/mlp_model_files"
    )

model = tf.keras.models.load_model(str(get_model_file_path() + "/model_mlp.keras"))

def mlp_wet_dry(
    trsl_channel_1, 
    trsl_channel_2,
    threshold=None, # 0.5 is often good, or argmax
):
    """
    Wet dry classification using a simple neural network:
    
    This MLP calculates wet and dry periods using a 40 minutes rolling window 
    for the CML signal loss from two sublinks (trsl_channel_1 and 
    trsl_channel_2) with temporal resolution equal to 1 minute. It consists of 
    one fully connected hidden layers with 20 neurons using the relu 
    activation function. The MLP was trained to predict rainfall recorded 
    at narby disdrometers at 10 minute resolution for one month of data with 14 
    pairs of CMLs and disdrometers from different locations in Norway. The MLP 
    was trained using MLPClassifier from sklearn and then transformed 
    to tensorflow to be compatible with the pycomlink environment. 
    
    If only one channel is available from the CML, use that channel for both
    trsl_channel_1 and trsl_channel_2. 
    
    The error "WARNING:absl:Skipping variable loading for optimizer 'Adam', 
    because it has 13 variables whereas the saved optimizer has 1 variables." 
    can safely be ignored. 

    Parameters
    ----------
    trsl_channel_1 : iterable of float
         Time series of received signal level of channel 1
    trsl_channel_2 : iterable of float
         Time series of received signal level of channel 2
    threshold : float 
        Threshold (0 - 1) for setting event as wet or dry. If set to None 
        (default), returns the continuous probability of wet [0, 1] from the 
        logistic activation function.

    Returns
    -------
    iterable of float
        Time series of wet/dry probability or (if threshold is provided) 
        wet dry classification 
        
    References
    ----------


    """
    # Normalization 
    trsl_channel_1_norm =  (trsl_channel_1 - np.nanmean(trsl_channel_1)) / np.nanstd(trsl_channel_1)
    trsl_channel_2_norm = (trsl_channel_2 - np.nanmean(trsl_channel_2)) / np.nanstd(trsl_channel_2)

    # add nan to start and end
    windowsize = 40 # use two channels 
    x_start = np.ones([int(windowsize/2), windowsize*2])*np.nan
    x_end = np.ones([int(windowsize/2)- 1, windowsize*2])*np.nan
    
    # sliding window
    sliding_window_ch1 = sliding_window_view(
        trsl_channel_1_norm, 
        window_shape = windowsize
    )
    
    sliding_window_ch2 = sliding_window_view(
        trsl_channel_2_norm, 
        window_shape = windowsize
    )

    x_fts = np.vstack(
        [x_start, np.hstack([sliding_window_ch1, sliding_window_ch2]), x_end]
    )
    
    mlp_pred = np.zeros([x_fts.shape[0], 2])*np.nan
    indices = np.argwhere(~np.isnan(x_fts).any(axis = 1)).ravel()
    
    if indices.size > 0: # else: predictions are kept as nan
        mlp_pred_ = model.predict(x_fts[indices], verbose=0)
        mlp_pred[indices] = mlp_pred_        
    
    if threshold == None:
        return mlp_pred 
    else:
        mlp_pred = mlp_pred[:, 1]
        mlp_pred[indices] = mlp_pred[indices] > threshold
        return mlp_pred
