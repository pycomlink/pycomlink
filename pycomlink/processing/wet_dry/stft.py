from __future__ import print_function
from __future__ import division

# ----------------------------------------------------------------------------
# Name:
# Purpose:
#
# Authors:
#
# Created:
# Copyright:    (c) Christian Chwala 2016
# Licence:      The MIT License
# ----------------------------------------------------------------------------


from builtins import range
import numpy as np
from matplotlib import mlab
from matplotlib.mlab import specgram as specg
import pandas as pd


def stft_classification(
    rsl,
    window_length,
    threshold,
    f_divide,
    t_dry_start=None,
    t_dry_stop=None,
    dry_length=None,
    mirror=False,
    window=None,
    Pxx=None,
    f=None,
    f_sampling=1 / 60,
):

    """Perform wet/dry classification with Rolling Fourier-transform method

    Parameters
    ----------
    rsl : iterable of float
         Time series of received signal level
    window_length : int
         Length of the sliding window
    threshold : int
         Threshold which has to be surpassed to classifiy a period as 'wet'
    f_divide : float
          Parameter for classification with method Fourier transformation
    t_dry_start : int
        Index of starting point dry period
    t_dry_stop : int
        Index of end of dry period
    dry_length : int
        Length of dry period that will be automatically identified in the
        provided rsl time series
    mirror : bool (defaults to False)
        Mirroring values in window at end of time series
    window : array of float, optional
        Values of window function. If not given a Hamming window function is
        applied (Default is None)
    Pxx : 2-D array of float, optional
        Spectrogram used for the wet/dry classification.
        Gets computed if not given (Default is None)
    f : array of float, optional
        Frequencies corresponding to the rows in Pxx.
        Gets computed if not given. (Default is None)
    f_sampling : float, optional
        Sampling frequency (samples per time unit). It is used to calculate
        the Fourier frequencies, freqs, in cycles per time unit.
        (Default is 1/60.0)
    mirror : bool

    Returns
    -------
    iterable of int
        Time series of wet/dry classification
    dict
        Dictionary holding information about the classification

    Note
    ----
    Implementation of Rolling Fourier-transform method [2]_

    References
    ----------
    .. [2] Chwala, C., Gmeiner, A., Qiu, W., Hipp, S., Nienaber, D., Siart, U.,
           Eibert, T., Pohl, M., Seltmann, J., Fritz, J. and Kunstmann, H.:
           "Precipitation observation using microwave backhaul links in the
           alpine and pre-alpine region of Southern Germany", Hydrology
           and Earth System Sciences, 16, 2647-2661, 2012

    """

    # Calculate spectrogram Pxx if it is not supplied as function argument
    if Pxx is None:
        # Set up sliding window for STFT
        if mirror:
            # Window length has to be even
            if window_length % 2 == 0:
                NFFT = window_length
            else:
                NFFT = window_length + 1
        else:
            NFFT = window_length

        if window is None:
            window = np.hamming(window_length)

        # Calculate spectrogram using STFT
        Pxx, f, t = specg(
            rsl, NFFT=NFFT, Fs=f_sampling, noverlap=NFFT - 1, window=window
        )

    elif Pxx is not None and f is not None:
        print("Skipping spectrogram calculation and using supplied Pxx")
        #
        # TODO: check that Pxx has the correct size
        #
        # ..... assert len(Pxx[0]) == len(rsl) - window_length
    elif Pxx is not None and f is None:
        raise ValueError("You have to supply f if you supply Pxx")
    else:
        raise ValueError("This should be impossible")

    # Add NaNs as the missing spectral data at the beginning and end of
    # the time series (stemming from the window length)
    N_diff = len(rsl) - len(Pxx[0])
    N_missing_start = np.floor(N_diff / 2)

    if mirror:
        for i in range((len(rsl) - 1) - (NFFT / 2 - 1), len(rsl)):
            rsl_mirr = np.concatenate(
                (rsl[i - (NFFT / 2 - 1) : i], rsl[i - (NFFT / 2 - 1) : i][::-1])
            )
            Pxx_mirr, f, t = specg(
                rsl_mirr, NFFT=NFFT, Fs=f_sampling, noverlap=NFFT - 1, window=window
            )

            if i == (len(rsl) - 1) - (NFFT / 2 - 1):
                Pxx_end = Pxx_mirr
            else:
                Pxx_end = np.append(Pxx_end, Pxx_mirr, 1)
        Pxx_extended = np.concatenate(
            (nans([len(Pxx), N_missing_start]), Pxx, Pxx_end), 1
        )
    else:
        N_missing_end = N_diff - N_missing_start
        Pxx_extended = np.concatenate(
            (nans([len(Pxx), N_missing_start]), Pxx, nans([len(Pxx), N_missing_end])), 1
        )

    if (t_dry_start is None) and (t_dry_stop is None) and (dry_length is not None):
        # Find dry period
        t_dry_start, t_dry_stop = find_lowest_std_dev_period(rsl, dry_length)
    elif (
        (t_dry_start is not None) and (t_dry_stop is not None) and (dry_length is None)
    ):
        # Do nothing, since t_dry_start and t_dry_stop are defined
        pass
    else:
        raise AttributeError(
            "Either `t_dry_start` and `t_dry_stop` or "
            "`dry_length` have to be supplied."
        )

    # Calculate mean dry spectrum
    P_dry_mean = np.nanmean(Pxx_extended[:, t_dry_start:t_dry_stop], axis=1)

    # Normalize the power spectrogram with the mean dry spectrum.
    # The array([...]) syntax is needed to transpose P_dry_mean to
    # a column vector (1D arrays cannot be transposed in Numpy)
    P_norm = Pxx_extended / np.array([P_dry_mean]).T

    i_f_divide_low = np.where(f <= f_divide)
    i_f_divide_high = np.where(f > f_divide)
    N_f_divide_low = len(i_f_divide_low)
    N_f_divide_high = len(i_f_divide_high)

    P_norm_low = np.mean(P_norm[i_f_divide_low], axis=0)
    P_norm_high = np.mean(P_norm[i_f_divide_high], axis=0)
    P_sum_diff = P_norm_low / N_f_divide_low - P_norm_high / N_f_divide_high

    nan_index = np.isnan(P_sum_diff)
    wet = np.zeros_like(P_sum_diff, dtype=bool)
    wet[~nan_index] = P_sum_diff[~nan_index] > threshold

    info = {
        "P_norm": P_norm,
        "P_sum_diff": P_sum_diff,
        "Pxx": Pxx_extended,
        "P_dry_mean": P_dry_mean,
        "f": f,
    }

    return wet, info


def find_lowest_std_dev_period(rsl, window_length=600):

    """Find beginning and end of dry period

    Parameters
    ----------
    rsl : iterable of float
         Time series of received signal level
    window_length : int, optional
         Length of window for identifying dry period (Default is 600)

    Returns
    -------
    int
        Index of beginning of dry period
    int
        Index of end of dry period

    """

    roll_std_dev = pd.DataFrame(rsl).rolling(window_length).std()
    dry_stop = mlab.find(roll_std_dev == np.nanmin(roll_std_dev))
    dry_stop = dry_stop[0]
    dry_start = dry_stop - window_length
    return dry_start, dry_stop


####################
# Helper functions #
####################


def nans(shape, dtype=float):
    """Helper function for wet/dry classification"""
    a = np.empty(np.asarray(shape, dtype=int), dtype)
    a.fill(np.nan)
    return a
