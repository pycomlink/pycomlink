import numpy as np
import xarray as xr
from pycomlink.processing.k_R_relation import a_b
from pycomlink.processing.xarray_wrapper import xarray_apply_along_time_dim

def nearby_determine_reference_level(
        pmin, pmax, wet, n_average_dry=96, min_periods=1):
    """
    Determine reference/baseline level during rain events as Overeem et al.
    (2016). The baseline ist the median of all dry time steps during the last
    `n_average_dry` dry time steps.

    Parameters
    ----------
    pmin : xarray.DataArray
         Time series of pmin.
    pmax : xarray.DataArray
        Time series of pmax. If not available e.g. because the min-max data
        is derived from instantaneous sampled CML data and has the same
        temporal resolution as the instantaneous CML data, then substitute pmax
        with pmin so pmin and pmax are identical. This is not optimal because
        the rainfall estimation in nearby_rainfall_retrival() uses a weighted
        average between the rain rate derived from pmin and pmax. Using
        pmin to substitute pmax can lead to overestimation of rainfall.

    wet : xarray.DataArray
         DataArray consisting of time series with wet-dry classification.
    n_average_dry: int
        Number of timesteps which are used to calculate the reference levek
        (baseline) from.
    min_periods: int
        Number of periods which have to be available in the last n_average_dry
        time steps to calculate a reference level.


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

    pref = pmean.rolling(time=n_average_dry, min_periods=min_periods).median()

    return pref


def nearby_correct_received_signals(pmin, pmax, wet, pref):
    """
    Correcting pmin and pmax so that no rainfall estimation is carried out
    during dry time steps. All time steps of pmin which are not classified wet
    and pmin is smaller than pref are set to pref. Similarly, all time steps of
    pmax where either the corrected pmin (p_c_min) is not smaller than pref or
    pmax is not smaller than pref are set to pref. This ensures that only wet
    time steps are used for rainfall estimation an and that pmax is not above
    pref  which would lead to an overestimation of rainfall.

    Parameter
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
    p_c_max = xr.where(cond=(p_c_min < pref) & (pmax < pref), x=pmax, y=pref)

    return p_c_min, p_c_max

@xarray_apply_along_time_dim()
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
    F_value_threshold=-32.5,
):
    """
    Calculating R from corrected pmin and pmax values using the `k-R`-relation
    using a and b values from ITU tables (1), (2). Please note that these values
    are not the same values as used in (3) Overeem et al. 2016 who derived
    a and b by fitting a R-k relation to their DSD data. Therefore, there is a
    deviation between a_overeem and 1/a_ITU if b is not equal to 1.
    Wet antenna is derived via Schleiss et al. (2010), the factor `alpha`
    determines the contribution of the minimum and maximum path-averaged
    rainfall intensity to the returend rain rate. The F-Score
    derived from `nearby_wetdry()` can be used to remove outliers.

    Parameters
    ----------
    pref : xr.DataArray
       Reference signal level, sometimes also called baseline
    p_c_min : xr.DataArray
        Corrected pmin
    p_c_max : xr.DataArray
        Corrected pmax
    F : xr.DataArray
        Computed filter to remove outliers
    f_GHz : xr.DataArray or np.array optional
        Frequency in GHz. If provided together with `pol`, it will be used to
        derive the parameters a and b for the k-R power law. If xr.DataArray,
        the coords must match those of pref, p_c_min, p_c_max and F. If np.array,
        the shape must match the shape of pref, p_c_min, p_c_max and F.
    pol : xr.DataArray, np.array or string, optional
        Polarization, that is either 'H' for horizontal or 'V' for vertical. Has
        to be provided together with `f_GHz`. It will be used to derive the
        parameters a and b for the k-R power law. If xr.DataArray,
        the coords must match those of pref, p_c_min, p_c_max and F. If np.array,
        the shape must match the shape of pref, p_c_min, p_c_max and F. If it is
        a str, it will be expanded to the shape of f_GHz.
    a : xr.DataArray, np.array, or iterable of those, optional
        Parameter of k-R relationship which can be taken from ITU (1) or (2).
        Note that it is not equal to 1/a used in Overeem et al. 2016 who used a
        R-k relation and own DSD data to derive values for a and b.
    b : xr.DataArray, np.array, or iterable of those, optional
        Parameter of k-R relationship which can be taken from ITU (1) or (2).
        Note that it is not equal to 1/b used in Overeem et al. 2016 who used a
        R-k relation and own DSD data to derive values for a and b.
    a_b_approximation : string
        Specifies which approximation for the k-R power law shall be used. See
        the function `a_b` for details.
    waa_max : float
        Maximum value of wet antenna attenuation
    alpha : float
        Between 0 and 1, determines the contribution of the minimum and maximum
        path-averaged rainfall intensity derive from p_c_min and p_c_max.
    F_value_threshold: float
        Outlier detection value calculated in `nearby_wetdry()` can be used to
        remove outliers. Set to `None` if it should not be used.

    Returns
    ----------
    R: xr.DataArray
        Rain rate in mm/h.

    References:
    ----------
    (1) & (2) Specific attenuation model for rain for use in prediction
    methods. Geneva, Switzerland: ITU-R.
    Versions within pycomlink:
    (1) P.838-2 (04/2003)
    (2) P.838-3 (03/2005)
    Retrieved from https://www.itu.int/rec/R-REC-P.838

    (3) Overeem, A., Leijnse, H., & Uijlenhoet, R. (2016). Retrieval algorithm
    for rainfall mapping from microwave links in a cellular communication
    network. Atmospheric Measurement Techniques, 9(5), 2425–2444.
    https://doi.org/10.5194/amt-9-2425-2016
    """

    # Make sure that we only continue if a correct combination of optional args is used
    if (f_GHz is not None) and (pol is not None) and (a is None) and (b is None):
        # f_GHz and pol must be np.arrays within this function before fed to
        # a_b(), otherwise a_b() can return a xr.DataArray with non-matching
        # dimensions in certain cases. That interferes with our xarray-wrapper
        # decorator.
        f_GHz = np.atleast_1d(f_GHz).astype(float)
        pol = np.atleast_1d(pol)
        a, b = a_b(f_GHz, pol=pol, approx_type=a_b_approximation)
    elif (a is not None) and (b is not None) and (f_GHz is None) and (pol is None):
        # in this case we use `a` and `b` from args
        pass
    else:
        raise ValueError(
            "Either `f_GHz` and `pol` or `a` and `b` have to be passed. "
            "Any other combination is not allowed."
        )

    # calculate minimum and maximum rain-induced attenuation
    A_min = pref - p_c_max
    A_max = pref - p_c_min

    # remove wet antenna attenuation
    A_min = A_min - waa_max
    A_max = A_max - waa_max

    # remove negative attenuation
    A_min = xr.where(A_min < 0, 0, A_min)
    A_max = xr.where(A_max < 0, 0, A_max)

    # retrieve rainfall intensities
    r_min = ((A_min) / (a * length)) ** (1 / b)
    r_max = ((A_max) / (a * length)) ** (1 / b)

    # weighted mean path averaged rainfall intensity
    R = (alpha * r_max) + ((1 - alpha) * r_min)

    # correct rainfall intensities by removing outliers defined by the F score
    if F_value_threshold is not None:
        R[F <= F_value_threshold] = np.nan

    return R