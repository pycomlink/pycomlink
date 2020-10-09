import tensorflow

def cnn_wet_dry(trsl_channel_1, trsl_channel_2, threshold):
    """
    Wet dry classification using the CNN based on channel 1 and channel 2 of a CML

    Parameters
    ----------
    trsl_channel_1 : iterable of float
         Time series of received signal level of channel 1
    trsl_channel_2 : iterable of float
         Time series of received signal level of channel 1
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



    return