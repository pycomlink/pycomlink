#----------------------------------------------------------------------------
# Name:         
# Purpose:      
#
# Authors:      
#
# Created:      
# Copyright:    (c) Christian Chwala 2016
# Licence:      The MIT License
#----------------------------------------------------------------------------

import numpy as np


def std_dev_classification(data, window_length, threshold):

    """Perform wet/dry classification with Rolling standard deviation method

    Parameters
    ----------
    data : iterable of float
         Time series of received signal level
    window_length : int
         Length of the sliding window
    threshold : int
         Threshold which has to be surpassed to classifiy a period as 'wet'

    Returns
    -------
    iterable of int
        Time series of wet/dry classification

    Note
    ----
    Implementation of Rolling standard deviation method [1]_

    References
    ----------
    .. [1] Schleiss, M. and Berne, A.: "Identification of dry and rainy periods
           using telecommunication microwave links", IEEE Geoscience and
           Remote Sensing, 7, 611-615, 2010

    """

    roll_std_dev = rolling_std_dev(data, window_length)
    nan_index = np.isnan(roll_std_dev)

    wet = np.zeros_like(roll_std_dev, dtype=np.bool)
    wet[~nan_index] = roll_std_dev[~nan_index] > threshold

    return wet, {'roll_std_dev': roll_std_dev}


def rolling_window(a, window):

    """Define sliding window

    Parameters
    ----------
    a : iterable of float
         Time series of values
    window : int
         Length of the sliding window
    """

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_std_dev(x, window_length, pad_only_left=False):

    """Calculate standard deviation in sliding window

    Parameters
    ----------
    x : iterable of float
         Time series of values
    window_length : int
         Length of the sliding window
    pad_only_left : bool
        Default is False

    Returns
    -------
    array of float
        standard deviation in sliding window

    """

    import numpy as np
    roll_std_dev = np.std(rolling_window(x, window_length), 1)
    pad_nan = np.zeros(window_length-1)
    pad_nan[:] = np.NaN
    # add NaN to start and end of array
    ix_mid = len(pad_nan) // 2
    if pad_only_left == False:
        roll_std_dev = np.concatenate((pad_nan[:ix_mid],
                                       roll_std_dev,
                                       pad_nan[ix_mid:]))
    elif pad_only_left == True:
        roll_std_dev = np.concatenate((pad_nan, roll_std_dev))
    else:
        ValueError('pad_only_left must be either True or False')
    return roll_std_dev
