import numpy as np
import xarray as xr
from pycomlink.processing.k_R_relation import a_b


def nearby_determine_reference_level(pmin, pmax, wet, n_average_dry=96):
    """
    Determine reference/baseline level during rain events as Overeem et al.
    (2016). The baseline ist the median of all dry time steps during the last
    `n_average_dry` dry time steps.
    ----------
    pmin : xarray.DataArray
         Time series of pmin.
    pmax : xarray.DataArray
        Time series of pmax. If not available e.g. because the min-max data
        is derived from instantaneous sampled CML data and has the same
        temporal resolution as the instantaneous CML data, substitute pmax
        with pmin so pmin and pmax are identical.
    wet : xarray.DataArray
         DataArray consisting of time series with wet-dry classification.
    n_average_dry: int
        Number of timesteps which are used to calculate the reference levek
        (baseline) from.

    Returns
    -------
    pref : xarray.DataArray
        Reference signal level, also called baseline
    References
    ----------
    .. [1] Overeem, A., Leijnse, H., and Uijlenhoet, R.: Retrieval algorithm
    for rainfall mapping from microwave links in a cellular communication
    network, Atmos. Meas. Tech., 9, 2425–2444,
    https://doi.org/10.5194/amt-9-2425-2016, 2016.
    """

    dry = (wet == 0).where(~np.isnan(wet))

    pmean = ((pmin + pmax) / 2).where(dry == 1)

    pref = pmean.rolling(time=n_average_dry, min_periods=1).median()

    return pref


def nearby_correct_received_signals(pmin, pmax, wet, pref):
    """
    Correcting pmin and pmax to prevent rainfall estimation during dry time
    steps.
    ----------
    pmin : xarray.DataArray
         Time series of pmin.
    pmax : xarray.DataArray
        Time series of pmax. If not available e.g. because the min-max data
        is derived from instantaneous sampled CML data and has the same
        temporal resolution as the instantaneous CML data, substitute pmax
        with pmin so pmin and pmax are identical.
    wet : xarray.DataArray
         DataArray consisting of time series with wet-dry classification.
    pref : xarray.DataArray
        Reference signal level, sometimes also called baseline
    Returns
    -------
    p_c_min : xarray.DataArray
        Corrected pmin
    p_c_max : xarray.DataArray
        Corrected pmax, if no pmax is given, p_c_max is identical to p_c_min
    References
    ----------
    .. [1] Overeem, A., Leijnse, H., and Uijlenhoet, R.: Retrieval algorithm
    for rainfall mapping from microwave links in a cellular communication
    network, Atmos. Meas. Tech., 9, 2425–2444,
    https://doi.org/10.5194/amt-9-2425-2016, 2016.
    """

    p_c_min = xr.where(cond=(pmin < pref) & (wet == 1), x=pmin, y=pref)

    p_c_max = xr.where(cond=(p_c_min < pref) & (pmin < pref), x=pmax, y=pref)

    return p_c_min, p_c_max


def nearby_rainfall_retrival(
    pref,
    p_c_min,
    p_c_max,
    F,
    length,
    f_GHz=None,
    pol=None,
    a=None,
    b=None,
    a_b_approximation="ITU_2005",
    waa_max=2.3,
    alpha=0.33,
    F_value_correction=True,
):
    """
    Calculating R from corrected pmin and pmax values using the `k-R`-relation.
    Wet antenna is derived via Schleiss et al. (2010), the factor `alpha`
    determines the contribution of the minimum and maximum path-averaged
    rainfall intensity to the returend rain rate. Optionally, the F-Score
    derived from `nearby_wetdry()` can be used to remove outliers.
    ----------
    pref : xr.DataArray
       Reference signal level, sometimes also called baseline
    p_c_min : xr.DataArray
        Corrected pmin
    p_c_max : xr.DataArray
        Corrected pmax
    F : xr.DataArray
        Computed filter to remove outliers
    f_GHz : float, optional
        Frequency in GHz. If provided together with `pol`, it will be used to
        derive the parameters a and b for the k-R power law.
    pol : string, optional
        Polarization, that is either 'H' for horizontal or 'V' for vertical. Has
        to be provided together with `f_GHz`. It will be used to derive the
        parameters a and b for the k-R power law.
    a : float, optional
        Parameter of A-R relationship
    b : float, optional
        Parameter of A-R relationship
    a_b_approximation : string
        Specifies which approximation for the k-R power law shall be used. See
        the function `a_b` for details.
    waa_max : float
        Maximum value of wet antenna attenuation
    alpha : float
        Between 0 and 1. B

    F_values_correction=True
    """

    # Make sure that we only continue if a correct combination of optional
    # args is used
    if (f_GHz is not None) and (pol is not None) and (a is None) and (b is None):
        if type(f_GHz) is not np.ndarray:
            f_GHz = f_GHz.values
        if type(pol) is not np.ndarray:
            pol = pol.values

        # check if freq and pol have the same size
        if f_GHz.shape == pol.shape:
            shape_save = f_GHz.shape

            a, b = [], []
            for i_freq, i_pol in zip(f_GHz.flatten(), pol.flatten()):
                a_tmp, b_tmp = a_b(
                    f_GHz=i_freq, pol=i_pol, approx_type=a_b_approximation
                )
                a.append(a_tmp)
                b.append(b_tmp)
            a = np.reshape(np.array(a), shape_save)
            b = np.reshape(np.array(b), shape_save)

            # turn a and b values to xarray.DataArray and check whether
            # no or the dim channel_id is available
            if "channel_id" in list(pref.dims):
                a = xr.DataArray(
                    a, coords=dict(cml_id=pref.cml_id, channel_id=pref.channel_id)
                )
                b = xr.DataArray(
                    b, coords=dict(cml_id=pref.cml_id, channel_id=pref.channel_id)
                )
            else:
                a = xr.DataArray(a, coords=dict(cml_id=pref.cml_id))
                b = xr.DataArray(b, coords=dict(cml_id=pref.cml_id))

        else:
            raise IndexError("Size of `f_GHz` and `pol` must be identical.")

    elif (a is not None) and (b is not None) and (f_GHz is None) and (pol is None):
        # in this case we use `a` and `b` from args
        pass
    else:
        raise ValueError(
            "Either `f_GHz` and `pol` or `a` and `b` have to be passed. Any "
            "other combination is not allowed."
        )

    # calculate minimum and maximum rain-induced attenuation
    A_min = pref - p_c_max
    A_max = pref - p_c_min

    # retrieve rainfall intensities
    r_min = ((1 / a) ** (1 / b) * (((A_min - waa_max) / length))) ** b
    r_max = ((1 / a) ** (1 / b) * (((A_max - waa_max) / length))) ** b

    # not allowing negative rain intensities
    r_min = xr.where(A_min - waa_max < 0, 0, r_min)
    r_max = xr.where(A_max - waa_max < 0, 0, r_max)

    # weighted mean path averaged rainfall intensity
    R = (alpha * r_max) + ((1 - alpha) * r_min)

    # correct rainfall intensities by removing outliers defined by the F score
    if F_value_correction is True:
        R = R.where(~(F <= -32.5))

    return R