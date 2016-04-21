import numpy as np
import pandas as pd

from numba.decorators import jit

################################################
# Functions for setting the RSL baseline level #
################################################


def baseline_constant(rsl, wet):
    """Baseline determination during wet period by keeping the RSL level constant
        at the level of the preceding dry period

    Parameters
    ----------
    rsl : iterable of float
          Received signal level or
          transmitted power level minus received power level
    wet : iterable of bool
          Information if classified index of times series is wet (True) or dry (False)

    Returns
    -------
    iterable of float
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
        if wet[i]:
            baseline[i] = baseline[i-1]
        else:
            baseline[i] = rsl[i]
    return baseline


def baseline_linear(rsl, wet):
    """Baseline determination during wet period by interpolating the RSL level
        linearly between two enframing dry periods

    Parameters
    ----------
    rsl : iterable of float
          Received signal level or
          transmitted power level minus received power level
    wet : iterable of bool
          Information if classified index of times series is wet (True) or dry (False)

    Returns
    -------
    iterable of float
          Baseline during wet period

    """

    if type(rsl) == pd.Series:
        rsl = rsl.values
    if type(wet) == pd.Series:
        wet = wet.values

    rsl = rsl.astype(np.float64)
    wet = wet.astype(np.float64)

    return _numba_baseline_linear(rsl, wet)


@jit(nopython=True)
def _numba_baseline_linear(rsl, wet):
    baseline = np.zeros_like(rsl, dtype=np.float64)
    baseline[0] = rsl[0]
    last_dry_i = 0
    last_dry_rsl = rsl[0]
    last_i_is_wet = False

    for i in range(1, len(rsl)):
        rsl_i = rsl[i]
        wet_i = wet[i]

        is_wet = wet_i

        # Check for NaN values. If NaN, then continue with
        # the last wet/dry state
        if np.isnan(is_wet):
            is_wet = last_i_is_wet

        # at the beginning of a wet period
        if is_wet and not last_i_is_wet:
            last_i_is_wet = True
        # within a wet period
        if is_wet and last_i_is_wet:
            last_i_is_wet = True
        # at the end of a wet period, do the baseline interpolation
        elif last_i_is_wet and not is_wet:
            # !! Only works correctly with 'i+1'. With 'i' the first dry
            # !! baseline value is kept at 0. No clue why we need the '+1'
            baseline[last_dry_i:i+1] = np.linspace(last_dry_rsl,
                                                   rsl_i,
                                                   i-last_dry_i+1)
            last_i_is_wet = False
            last_dry_i = i
            last_dry_rsl = rsl_i
        # within a dry period
        elif not last_i_is_wet and not is_wet:
            baseline[i] = rsl_i
            last_i_is_wet = False
            last_dry_i = i
            last_dry_rsl = rsl_i
        else:
            #print('This should be impossible')
            raise
    return baseline


