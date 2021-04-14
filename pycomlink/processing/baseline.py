from builtins import range
import numpy as np
import pandas as pd

from numba import jit

from .xarray_wrapper import xarray_loop_vars_over_dim


################################################
# Functions for setting the RSL baseline level #
################################################


@xarray_loop_vars_over_dim(vars_to_loop=["trsl", "wet"], loop_dim="channel_id")
def baseline_constant(trsl, wet, n_average_last_dry=1):
    """
    Build baseline with constant level during a `wet` period

    Parameters
    ----------
    trsl : numpy.array or pandas.Series
        Transmitted signal level minus received signal level (TRSL) or
        received signal level or t
    wet : numpy.array or pandas.Series
        Information if classified index of times series is wet (True)
        or dry (False). Note that `NaN`s in `wet` will lead to `NaN`s in
        `baseline` also after the `NaN` period since it is then not clear
        whether or not there was a change of wet/dry within the `NaN` period.
    n_average_last_dry: int, default = 1
        Number of last baseline values before start of wet event that should
        be averaged to get the value of the baseline during the wet event.
        Note that this values should not be too large because the baseline
        might be at an expected level, e.g. if another wet event is
        ending shortly before.

    Returns
    -------
    baseline : numpy.array
          Baseline during wet period

    """

    return _numba_baseline_constant(
        trsl=np.asarray(trsl, dtype=np.float64),
        wet=np.asarray(wet, dtype=np.bool),
        n_average_last_dry=n_average_last_dry,
    )


@jit(nopython=True)
def _numba_baseline_constant(trsl, wet, n_average_last_dry):
    baseline = np.zeros_like(trsl, dtype=np.float64)
    baseline[0:n_average_last_dry] = trsl[0:n_average_last_dry]
    for i in range(n_average_last_dry, len(trsl)):
        if np.isnan(wet[i]):
            baseline[i] = np.NaN
        elif wet[i] & ~wet[i-1]:
            baseline[i] = np.mean(baseline[(i-n_average_last_dry) : i])
        elif wet[i] & wet[i-1]:
            baseline[i] = baseline[i - 1]
        else:
            baseline[i] = trsl[i]
    return baseline


def baseline_linear(rsl, wet, ignore_nan=False):
    """
    Build baseline with linear interpolation from start till end of `wet` period

    Parameters
    ----------
    rsl : numpy.array or pandas.Series
          Received signal level or transmitted signal level minus received
          signal level
    wet : numpy.array or pandas.Series
          Information if classified index of times series is wet (True)
          or dry (False). Note that `NaN`s in `wet` will lead to `NaN`s in
          `baseline` also after the `NaN` period since it is then not clear
          wheter there was a change of wet/dry within the `NaN` period.

    Returns
    -------
    baseline : numpy.array
          Baseline during wet period

    """

    if type(rsl) == pd.Series:
        rsl = rsl.values
    if type(wet) == pd.Series:
        wet = wet.values

    rsl = rsl.astype(np.float64)
    wet = wet.astype(np.float64)

    return _numba_baseline_linear(rsl, wet, ignore_nan)


@jit(nopython=True)
def _numba_baseline_linear(rsl, wet, ignore_nan=False):
    baseline = np.zeros_like(rsl, dtype=np.float64)
    baseline[0] = rsl[0]
    last_dry_i = 0
    last_dry_rsl = rsl[0]
    last_i_is_wet = False
    found_nan = False

    for i in range(1, len(rsl)):
        rsl_i = rsl[i]
        wet_i = wet[i]
        is_wet = wet_i

        # Check for NaN values.
        if np.isnan(is_wet):
            # If NaNs should be ignored, continue with the last wet/dry state
            if ignore_nan:
                is_wet = last_i_is_wet
            else:
                found_nan = True
                # raise ValueError('There must not be `NaN`s in `wet` if '
                #                 '`ignore_nan` is set to `True`.')

        # at the beginning of a wet period
        if is_wet and not last_i_is_wet:
            last_i_is_wet = True
        # within a wet period
        if is_wet and last_i_is_wet:
            last_i_is_wet = True
        # at the end of a wet period, do the baseline interpolation
        elif last_i_is_wet and not is_wet:
            if found_nan:
                baseline[last_dry_i : i + 1] = np.NaN
            else:
                # !! Only works correctly with 'i+1'. With 'i' the first dry
                # !! baseline value is kept at 0. No clue why we need the '+1'
                baseline[last_dry_i : i + 1] = np.linspace(
                    last_dry_rsl, rsl_i, i - last_dry_i + 1
                )
            found_nan = False
            last_i_is_wet = False
            last_dry_i = i
            last_dry_rsl = rsl_i
        # within a dry period
        elif not last_i_is_wet and not is_wet:
            if found_nan:
                baseline[i] = np.NaN
            else:
                baseline[i] = rsl_i
            found_nan = False
            last_i_is_wet = False
            last_dry_i = i
            last_dry_rsl = rsl_i
        else:
            # print('This should be impossible')
            raise
    return baseline
