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
    ''' Calculate wet antenna attenuation according to Schleiss et al 2013'''
    if wet==True:
        waa = min(A, waa_max, 
                  waa + (waa_max-waa)*3*delta_t/tau)
    else:
        waa = min(A, waa_max)
    return waa

def waa_adjust_baseline(rsl, baseline, waa_max, delta_t, tau, wet):
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


