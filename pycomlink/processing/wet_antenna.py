# coding=utf-8
# ----------------------------------------------------------------------------
# Name:         wet_antenna
# Purpose:      Estimation of wet antenna effects
#
# Authors:      Christian Chwala
#
# Created:      01.12.2014
# Copyright:    (c) Christian Chwala 2014
# Licence:      The MIT License
# ----------------------------------------------------------------------------

from builtins import range

import numpy as np
import scipy.interpolate
from numba import jit

from pycomlink.processing import k_R_relation
from .xarray_wrapper import xarray_loop_vars_over_dim


########################################
# Functions for wet antenna estimation #
########################################


@jit(nopython=True)
def _numba_waa_schleiss_2013(rsl, baseline, wet, waa_max, delta_t, tau):
    """Fast loop using numba to calculate WAA according to Schleiss et al 2013

    Parameters
    ----------
        rsl : iterable of float
                Time series of received signal level
        baseline : iterable of float
                Time series of baseline for rsl
        wet : iterable of int or iterable of float
               Time series with wet/dry classification information.
        waa_max : float
                  Maximum value of wet antenna attenuation
        delta_t : float
                  Parameter for wet antnenna attenation model
        tau : float
              Parameter for wet antnenna attenation model

    Returns
    -------
       iterable of float
           Time series of wet antenna attenuation
    """

    waa = np.zeros_like(rsl, dtype=np.float64)
    A = rsl - baseline

    for i in range(1, len(rsl)):
        if wet[i] == True:
            waa[i] = min(
                A[i], waa_max, waa[i - 1] + (waa_max - waa[i - 1]) * 3 * delta_t / tau
            )
        else:
            waa[i] = min(A[i], waa_max)
    return waa


@xarray_loop_vars_over_dim(
    vars_to_loop=["rsl", "baseline", "wet"], loop_dim="channel_id"
)
def waa_schleiss_2013(rsl, baseline, wet, waa_max, delta_t, tau):
    """Calculate WAA according to Schleiss et al 2013

    Parameters
    ----------
        rsl : iterable of float
                Time series of received signal level
        baseline : iterable of float
                Time series of baseline for rsl
        wet : iterable of int or iterable of float
               Time series with wet/dry classification information.
        waa_max : float
                  Maximum value of wet antenna attenuation
        delta_t : float
                  Parameter for wet antenna attention model
        tau : float
              Parameter for wet antenna attenuation model

    Returns
    -------
       iterable of float
           Time series of wet antenna attenuation

    Note
    ----
        The wet antenna adjusting is based on a peer-reviewed publication [1]_

    References
    ----------
    .. [1] Schleiss, M., Rieckermann, J. and Berne, A.: "Quantification and
                modeling of wet-antenna attenuation for commercial microwave
                links", IEEE Geoscience and Remote Sensing Letters, 10, 2013
    """

    waa = _numba_waa_schleiss_2013(
        rsl=np.asarray(rsl, dtype=np.float64),
        baseline=np.asarray(baseline, dtype=np.float64),
        wet=np.asarray(wet, dtype=np.float64),
        waa_max=waa_max,
        delta_t=delta_t,
        tau=tau,
    )

    return waa


def waa_leijnse_2008_from_A_obs(
    A_obs,
    f_Hz,
    L_km,
    T_K=293.0,
    gamma=2.06e-5,
    delta=0.24,
    n_antenna=np.complex(1.73, 0.014),
    l_antenna=0.001,
):
    """Calculate wet antenna attenuation according to Leijnse et al. 2008

    Calculate the wet antenna attenuation from observed attenuation,
    using the method proposed in [2]_, assuming a rain rate dependent
    thin flat water film on the antenna.

    The equations proposed in [2]_ calculate the WAA from the rain rate R.
    With CML data the rain rates is not directly available. We need to use
    the observed attenuation to derive the WAA. This is done here by building
    a look-up-table for the relation between A_obs and WAA, where A_obs is
    calculated as A_obs = A_rain + WAA. A_rain is derived from the A-R relation
    for the given CML frequency and length.

    Parameters
    ----------
    A_obs : array-like or scalar
        Observed attenuation
    f_Hz : array-like or scalar (but only either `R` or `f_Hz` can be array)
        Frequency of CML in Hz
    L_km : float
        Lenght of CML in kilometer
    gamma : float
        Parameter that determines the magnitutde of the water film thickness
    delta : float
        Parameter that determines the non-linearity of the relation
        between water film thickness and rain rates
    n_antenna : float
        Refractive index of antenna material
    l_antenna : float
        Thickness of antenna cover

    Returns
    -------
    waa : array-like
        Wet antenna attenuation in dB

    References
    ----------

    .. [2]  H. Leijnse, R. Uijlenhoet, J.N.M. Stricker: "Microwave link rainfall
           estimation: Effects of link length and frequency, temporal sampling,
           power resolution, and wet antenna attenuation", Advances in
           Water Resources, Volume 31, Issue 11, 2008, Pages 1481-1493,
           https://doi.org/10.1016/j.advwatres.2008.03.004.

    """

    # Generate mapping from A_obs to WAA
    A_rain = np.logspace(-10, 3, 100)
    A_rain[0] = 0

    R = k_R_relation.calc_R_from_A(A_rain, L_km=L_km, f_GHz=f_Hz / 1e9, R_min=0)
    waa = waa_leijnse_2008(
        f_Hz=f_Hz,
        R=R,
        gamma=gamma,
        delta=delta,
        T_K=T_K,
        n_antenna=n_antenna,
        l_antenna=l_antenna,
    )
    A_obs_theoretical = A_rain + waa

    mapping = scipy.interpolate.interp1d(
        A_obs_theoretical, waa, assume_sorted=True, kind="linear"
    )
    return mapping(A_obs)


def waa_leijnse_2008(
    R,
    f_Hz,
    T_K=293.0,
    gamma=2.06e-5,
    delta=0.24,
    n_antenna=np.complex(1.73, 0.014),
    l_antenna=0.001,
):
    """Calculate wet antenna attenuation according to Leijnse et al. 2008

    Calculate the wet antenna attenuation assuming a rain rate dependent
    thin flat water film on the antenna following the results from [3]_.

    Water film thickness:
        l = gamma * R ** delta

    Parameters
    ----------
    R : array-like or scalar
        Rain rate in mm/h
    f_Hz : array-like or scalar (but only either `R` or `f_Hz` can be array)
        Frequency of CML in Hz
    gamma : float
        Parameter that determines the magnitutde of the water film thickness
    delta : float
        Parameter that determines the non-linearity of the relation
        between water film thickness and rain rates
    n_antenna : float
        Refractive index of antenna material
    l_antenna : float
        Thickness of antenna cover

    Returns
    -------
    waa : array-like
        Wet antenna attenuation in dB

    References
    ----------

    .. [3]  H. Leijnse, R. Uijlenhoet, J.N.M. Stricker: "Microwave link rainfall
           estimation: Effects of link length and frequency, temporal sampling,
           power resolution, and wet antenna attenuation", Advances in
           Water Resources, Volume 31, Issue 11, 2008, Pages 1481-1493,
           https://doi.org/10.1016/j.advwatres.2008.03.004.

    """

    R = np.asanyarray(R)

    n_air = 1
    c = 299792458

    l = gamma * R ** delta

    n_water = np.sqrt(eps_water(f_Hz=f_Hz, T_K=T_K))

    expo = 1j * 2 * np.pi * f_Hz / c
    x1 = (
        (n_air + n_water)
        * (n_water + n_antenna)
        * (n_antenna + n_air)
        * np.exp(-expo * (n_antenna * l_antenna + n_water * l))
    )
    x2 = (
        (n_air - n_water)
        * (n_water - n_antenna)
        * (n_antenna + n_air)
        * np.exp(-expo * (n_antenna * l_antenna - n_water * l))
    )
    x3 = (
        (n_air + n_water)
        * (n_water - n_antenna)
        * (n_antenna - n_air)
        * np.exp(expo * (n_antenna * l_antenna - n_water * l))
    )
    x4 = (
        (n_air - n_water)
        * (n_water + n_antenna)
        * (n_antenna - n_air)
        * np.exp(expo * (n_antenna * l_antenna + n_water * l))
    )

    y1 = (n_air + n_antenna) ** 2 * np.exp(-expo * n_antenna * l_antenna)
    y2 = -((n_air - n_antenna) ** 2) * np.exp(expo * n_antenna * l_antenna)

    waa = 10 * np.log10(np.abs((x1 + x2 + x3 + x4) / (2 * n_water * (y1 + y2))) ** 2)

    # Assure that numeric inaccuracy does not lead to waa > 0 for R == 0
    waa[R == 0] = 0

    return waa


def eps_water(f_Hz, T_K):
    """Calculate the dielectric permitiviy of water

    Formulas taken from dielectric permittivity of liquid water
    without salt according to
    Liebe et al. 1991 Int. J. IR+mm Waves 12(12), 659-675

    Based on MATLAB code by Christian Mätzler, June 2002
    Cosmetic changes by Christian Chwala, August 2012

    Parameters
    ----------
    f_Hz : array-like
        Frequency in Hz
    T_K : float
        Temperature in Kelvin

    Returns
    -------

    eps : np.complex

    """

    f_GHz = f_Hz * 1e-9

    teta = 1 - 300.0 / T_K
    e0 = 77.66 - 103.3 * teta
    e1 = 0.0671 * e0
    f1 = 20.2 + 146.4 * teta + 316 * teta * teta
    e2 = 3.52 + 7.52 * teta
    # Note that "Liebe et al 1993, Propagation Modeling of Moist Air and
    # Suspended Water/Ice Particles at Frequencies Below 1000 GHz,
    # AGARD Conference Proc. 542" uses just e2 = 3.52. For our frequency
    # and temperature range the difference is negligible, though.

    f2 = 39.8 * f1
    eps = e2 + (e1 - e2) / (1 - 1j * f_GHz / f2) + (e0 - e1) / (1 - 1j * f_GHz / f1)
    return eps
