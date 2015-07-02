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


########################################
# Functions for wet antenna estimation #
########################################

def waa_Schleiss_recursive(A, waa, waa_max, delta_t, tau, wet):
    
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
    
    if wet==True:
        waa = min(A, waa_max, 
                  waa + (waa_max-waa)*3*delta_t/tau)
    else:
        waa = min(A, waa_max)
    return waa

def waa_adjust_baseline(rsl, baseline, waa_max, delta_t, tau, wet):
    
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
    
    waa = np.zeros_like(rsl)
    A = rsl - baseline
    # iterate of A timeseries and recursively apply waa_Schleiss
    for i in range(1,len(rsl)):
        waa[i] = waa_Schleiss_recursive(A[i], 
                                        waa[i-1], 
                                        waa_max=waa_max, 
                                        delta_t=delta_t, 
                                        tau=tau,
                                        wet=wet[i])
    return baseline + waa, waa


