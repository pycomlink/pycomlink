import numpy as np
from numpy.lib.stride_tricks import sliding_window_view 
import tensorflow as tf
import pkg_resources

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
    Wet dry classification using a simple neural network based on channel 1 and channel 2 of a CML

    Parameters
    ----------
    trsl_channel_1 : iterable of float
         Time series of received signal level of channel 1
    trsl_channel_2 : iterable of float
         Time series of received signal level of channel 2
    threshold : float 
        Threshold (0 - 1) for setting event as wet or dry. Default None uses 
        the continuous output from the logistic function.
    

    Returns
    -------
    iterable of float
        Time series of wet/dry probability or (if threshold is provided) 
        wet dry classification 
        
    References
    ----------


    """
    # Normalization 
    trsl_channel_1_norm = (trsl_channel_1 - np.nanmean(trsl_channel_1)) / np.nanstd(trsl_channel_1)
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
