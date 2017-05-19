#----------------------------------------------------------------------------
# Name:         wet_antenna
# Purpose:      Estimation and removal of wet antenna effects
#
# Authors:      Christian Chwala
#
# Created:      01.12.2014
# Copyright:    (c) Christian Chwala 2014
# Licence:      The MIT License
#----------------------------------------------------------------------------

import numpy as np
import pandas as pd

from numba.decorators import jit


########################################
# Functions for wet antenna estimation #
########################################

@jit(nopython=True)
def _numba_waa_schleiss(rsl, baseline, waa_max, delta_t, tau, wet):
    """Calculate wet antenna attenuation

    Parameters
    ----------
        A : float
              Attenuation value
        waa : float
              Value of wet antenna attenuation at the preceding timestep
        waa_max : float
               Maximum value of wet antenna attenuation
        delta_t : float
               Parameter for wet antnenna attenation model
        tau : float
              Parameter for wet antnenna attenation model
        wet : int or float
               Wet/dry classification information.

    Returns
    -------
       float
           Value of wet antenna attenuation

    Note
    ----
        The wet antenna adjusting is based on a peer-reviewed publication [3]_

    References
    ----------
    .. [3] Schleiss, M., Rieckermann, J. and Berne, A.: "Quantification and
                modeling of wet-antenna attenuation for commercial microwave
                links", IEEE Geoscience and Remote Sensing Letters, 10, 2013
    """

    waa = np.zeros_like(rsl, dtype=np.float64)
    A = rsl - baseline

    for i in range(1,len(rsl)):
        if wet[i] == True:
            waa[i] = min(A[i],
                         waa_max,
                         waa[i-1] + (waa_max-waa[i-1])*3*delta_t/tau)
        else:
            waa[i] = min(A[i],
                         waa_max)
    return waa


def waa_adjust_baseline(rsl, baseline, wet, waa_max, delta_t, tau):
    
    """Calculate baseline adjustion due to wet antenna
        
    Parameters
    ----------
        rsl : iterable of float
                Time series of received signal level
        baseline : iterable of float
                Time series of baseline for rsl        
        waa_max : float
                  Maximum value of wet antenna attenuation   
        delta_t : float
                  Parameter for wet antnenna attenation model    
        tau : float
              Parameter for wet antnenna attenation model         
        wet : iterable of int or iterable of float
               Time series with wet/dry classification information. 
               
    Returns
    -------
       iterable of float
           Adjusted time series of baseline
       iterable of float
           Time series of wet antenna attenuation
        
    """     
    
    if type(rsl) == pd.Series:
        rsl = rsl.values
    if type(baseline) == pd.Series:
        baseline = baseline.values
    if type(wet) == pd.Series:
        wet = wet.values

    rsl = rsl.astype(np.float64)
    baseline = baseline.astype(np.float64)
    wet = wet.astype(np.float64)

    waa = _numba_waa_schleiss(rsl, baseline, waa_max, delta_t, tau, wet)

    #return baseline + waa, waa
    return baseline + waa
