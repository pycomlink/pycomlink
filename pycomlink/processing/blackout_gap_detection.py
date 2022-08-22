import numpy as np
import numba
from .xarray_wrapper import xarray_apply_along_time_dim


@xarray_apply_along_time_dim()
def get_blackout_start_and_end(rsl, rsl_threshold):
    """
    Identify nan gaps as blackout gaps caused by heavy attenuation based on
    a rs threshold and return the start and end of each gap

    Parameters
    ----------
    rsl: xr.DataArray, np.array
        Time series of received signal level (rsl)
    rsl_threshold: numeric
        Value below which the start or end of a NaN gap have to be to count
        as blackout gaps

    Returns
    -------
    gap_start, gap_end

    """

    # required to avoid RuntimeWarning when comparing to NaNs
    with np.errstate(invalid="ignore"):
        rsl_below_threshold = rsl < rsl_threshold
    rsl_nan = np.isnan(rsl)
    gap_start = np.roll(rsl_nan, -1) & rsl_below_threshold
    gap_end = np.roll(rsl_nan, 1) & rsl_below_threshold

    return gap_start, gap_end


@numba.jit(nopython=True)
def created_blackout_gap_mask_from_start_end_markers(rsl, gap_start, gap_end):
    """
    Deriving a mask which masks all detected blackout gaps.

    Parameters
    ----------
    rsl: xr.DataArray, np.array
        Time series of received signal level (rsl)
    gap_start: bool
        bool list with the start of detected blackout gaps
    gap_end: bool
        bool list with the end of detected blackout gaps

    Returns
    -------
    mask

    """

    mask = np.zeros(rsl.shape, dtype=np.bool_)
    in_blackout_gap = False
    for i in range(len(rsl)):
        if gap_start[i] == True:
            in_blackout_gap = True
        if (gap_start[i] == False) and ~np.isnan(rsl[i]):
            in_blackout_gap = False
        if gap_end[i] == True:
            in_blackout_gap = False
        if in_blackout_gap and np.isnan(rsl[i]):
            mask[i] = True
    return mask
