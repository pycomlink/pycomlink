import numpy as np


def _calc_A_min_max(tx_min, tx_max, rx_min, rx_max, gT=1.0, gR=0.6, window=7):
    """Calculate rain rate from attenuation using the A-R Relationship
    Parameters
    ----------
    gT : float, optional
        induced bias
    gR : float, optional
        induced bias
    window: int, optional
        number of previous measurements to use for zero-level calculation
    Returns
    -------
    float or iterable of float
        Ar_max
    Note
    ----
    Based on: "Empirical Study of the Quantization Bias Effects in
    Commercial Microwave Links Min/Max Attenuation
    Measurements for Rain Monitoring" by OSTROMETZKY J., ESHEL A.
    """

    # quantization bias correction
    Ac_max = tx_max - rx_min - (gT + gR) / 2
    Ac_min = tx_min - rx_max + (gT + gR) / 2

    Ac_max[np.isnan(Ac_max)] = np.rint(np.nanmean(Ac_max))
    Ac_min[np.isnan(Ac_min)] = np.rint(np.nanmean(Ac_min))

    # zero-level calculation
    Ar_max = np.full(Ac_max.shape, 0.0)
    for i in range(window, len(Ac_max)):
        Ar_max[i] = Ac_max[i] - Ac_min[i - window : i + 1].min()
    Ar_max[Ar_max < 0.0] = 0.0
    Ar_max[0:window] = np.nan

    return Ar_max
