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

import pandas as pd

import pycomlink as pycml


def std_dev_classification(data, window_length, threshold):

    """Perform wet/dry classification with Rolling standard deviation method

    Parameters
    ----------
    data : iterable of float, Comlink or ComlinkChannel
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

    if isinstance(data, pycml.Comlink):
        cml = data
        for channel_name, channel in cml.channels.iteritems():
            ts_wet, ts_roll_std_dev = std_dev_classification(
                                                data=channel,
                                                window_length=window_length,
                                                threshold=threshold)
            channel._df['wet'] = ts_wet

            # TODO: What to do with intermediate data, here roll_std_dev time series

            # TODO: Write to something like a processing_info dict for each channel

        return cml

    if isinstance(data, pycml.ComlinkChannel):
        cmlch = data
        # TODO: Maybe better resolve the request of 'trsl' in ComlinkChannel
        try:
            trsl = cmlch.trsl
        except:
            print 'Could not find TRSL in channel. Using TX-RX or only RX.'
            try:
                trsl = cmlch.tx - cmlch.rx
            except:
                trsl = cmlch.rx

        wet, roll_std_dev = std_dev_classification(trsl.values,
                                                   window_length=window_length,
                                                   threshold=threshold)
        ts_wet = pd.Series(data=wet, index=cmlch._df.index)
        ts_roll_std_dev = pd.Series(data=roll_std_dev, index=cmlch._df.index)
        return ts_wet, ts_roll_std_dev

    else:
        roll_std_dev = rolling_std_dev(data, window_length)
        wet = roll_std_dev > threshold
        return wet, roll_std_dev


def rolling_window(a, window):

    """Define sliding window

    Parameters
    ----------
    a : iterable of float
         Time series of values
    window : int
         Length of the sliding window
    """

    import numpy as np
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
    ix_mid = len(pad_nan)/2
    if pad_only_left == False:
        roll_std_dev = np.concatenate((pad_nan[:ix_mid],
                                       roll_std_dev,
                                       pad_nan[ix_mid:]))
    elif pad_only_left == True:
        roll_std_dev = np.concatenate((pad_nan, roll_std_dev))
    else:
        ValueError('pad_only_left must be either True or False')
    return roll_std_dev
