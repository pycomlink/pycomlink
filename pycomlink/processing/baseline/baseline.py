from builtins import range
import numpy as np
import pandas as pd

from numba import jit

################################################
# Functions for setting the RSL baseline level #
################################################


def baseline_constant(rsl, wet):
    """
    Build baseline with constant level during a `wet` period

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

    return _numba_baseline_constant(rsl, wet)


@jit(nopython=True)
def _numba_baseline_constant(rsl, wet):
    baseline = np.zeros_like(rsl, dtype=np.float64)
    baseline[0] = rsl[0]
    for i in range(1,len(rsl)):
        if np.isnan(wet[i]):
            baseline[i] = np.NaN
        elif wet[i]:
            baseline[i] = baseline[i-1]
        else:
            baseline[i] = rsl[i]
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
                #raise ValueError('There must not be `NaN`s in `wet` if '
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
                baseline[last_dry_i:i+1] = np.NaN
            else:
                # !! Only works correctly with 'i+1'. With 'i' the first dry
                # !! baseline value is kept at 0. No clue why we need the '+1'
                baseline[last_dry_i:i+1] = np.linspace(last_dry_rsl,
                                                       rsl_i,
                                                       i-last_dry_i+1)
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
            #print('This should be impossible')
            raise
    return baseline


