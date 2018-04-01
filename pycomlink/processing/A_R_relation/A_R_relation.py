from __future__ import division
import numpy as np


############################################
# Functions for A-R power law calculatoins #
############################################

def calc_R_from_A(A, L, f_GHz=None, a=None, b=None, pol='H', R_min=0.1):
    """Calculate rain rate from attenuation using the A-R Relationship
    
    Parameters
    ----------
    A : float or iterable of float
        Attenuation of microwave signal
    f_GHz : float, optional
        Frequency in GHz
    pol : string
        Polarization, default is 'H'
    a : float, optional
        Parameter of A-R relationship
    b : float, optional
        Parameter of A-R relationship
    L : float
        length of the link
    R_min : float
        Minimal rain rate in mm/h. Everything below will be set to zero.
        
    Returns
    -------
    float or iterable of float
        Rain rate    
    
    Note
    ----    
    The A-R Relationship is defined as
    
    .. math:: A = aR^{b}    
    
    """
    if f_GHz is not None:
        a, b = a_b(f_GHz, pol=pol)

    R = np.zeros_like(A)

    nan_index = np.isnan(A)
    R[nan_index] = np.nan

    # This ignores the numpy warning stemming from A >=0 where A contains NaNs
    with np.errstate(invalid='ignore'):
        R[~nan_index & (A >= 0)] = (A[~nan_index & (A >= 0)] / (a*L)) ** (1/b)
        R[~nan_index & (R < R_min)] = 0
    return R


def calc_R_from_A_min_max(Ar_max, L, f_GHz=None, a=None, b=None, pol='H',
                          R_min=0.1, k=90):
    """Calculate rain rate from attenuation using the A-R Relationship
    Parameters
    ----------
    Ar_max : float or iterable of float
        Attenuation of microwave signal (with min/max measurements of RSL/TSL)
    f_GHz : float, optional
        Frequency in GHz
    pol : string
        Polarization, default is 'H'
    a : float, optional
        Parameter of A-R relationship
    b : float, optional
        Parameter of A-R relationship
    L : float
        length of the link
    R_min : float
        Minimal rain rate in mm/h. Everything below will be set to zero.
    k : int, optional
        number of measurements between two consecutive measurement of rx/tx
    Returns
    -------
    float or iterable of float
        Rain rate
    Note
    ----
    Based on: "Empirical Study of the Quantization Bias Effects in
    Commercial Microwave Links Min/Max Attenuation
    Measurements for Rain Monitoring" by OSTROMETZKY J., ESHEL A.
    """

    # calculate rain-rate using the calibrated power law (with wet-antenna)
    euler_gamma = 0.57721566
    if f_GHz is not None:
        a, b = a_b(f_GHz, pol=pol)

    a_max = a * (np.log(k) + euler_gamma) ** b

    # calculate rainfall
    R = np.zeros_like(Ar_max, dtype='float')

    nan_index = np.isnan(Ar_max)
    R[nan_index] = np.nan

    # This ignores the numpy warning stemming from A >=0 where A contains NaNs
    with np.errstate(invalid='ignore'):
        R[~nan_index & (Ar_max >= 0)] = \
          (Ar_max[~nan_index & (Ar_max >= 0)] / (a_max*L)) ** (1.0/b)
        R[~nan_index & (R < R_min)] = 0.0
    return R


def a_b(f_GHz, pol, approx_type='ITU'):
    """Approximation of parameters for A-R relationship
    
    Parameters
    ----------
    f_GHz : int, float or np.array of these
            Frequency of the microwave link in GHz
    pol : str
            Polarization of the microwave link 
    approx_type : str, optional
            Approximation type (the default is 'ITU', which implies parameter
            approximation using a table recommanded by ITU)
            
    Returns
    -------
    a,b : float
          Parameters of A-R relationship      
     
    Note
    ----     
    The frequency value must be between 1 Ghz and 100 GHz.    
    
    The polarization has to be indicated by 'h' or 'H' for horizontal and 
    'v' or 'V' for vertical polarization respectively.
    
    Currently only 'ITU' for approx_type is accepted. The approximation makes
    use of a table recommanded by ITU [4]_.   
    
    References
    ----------
    .. [4] ITU, "ITU-R: Specific attenuation model for rain for use in 
        prediction methods", International Telecommunication Union, 2013 
         
    """
    from scipy.interpolate import interp1d

    f_GHz = np.asarray(f_GHz)

    if f_GHz.min() < 1 or f_GHz.max() > 100:
        raise ValueError('Frequency must be between 1 Ghz and 100 GHz.')
    else:
        if pol == 'V' or pol == 'v':
            f_a = interp1d(ITU_table[0, :], ITU_table[2, :], kind='cubic')
            f_b = interp1d(ITU_table[0, :], ITU_table[4, :], kind='cubic')
        elif pol == 'H' or pol == 'h':
            f_a = interp1d(ITU_table[0, :], ITU_table[1, :], kind='cubic')
            f_b = interp1d(ITU_table[0, :], ITU_table[3, :], kind='cubic')
        else:
            raise ValueError('Polarization must be V, v, H or h.')
        a = f_a(f_GHz)
        b = f_b(f_GHz)
    return a, b


ITU_table = np.array([
  [1.000e+0, 2.000e+0, 4.000e+0, 6.000e+0, 7.000e+0, 8.000e+0, 1.000e+1, 
   1.200e+1, 1.500e+1, 2.000e+1, 2.500e+1, 3.000e+1, 3.500e+1, 4.000e+1, 
   4.500e+1, 5.000e+1, 6.000e+1, 7.000e+1, 8.000e+1, 9.000e+1, 1.000e+2],
  [3.870e-5, 2.000e-4, 6.000e-4, 1.800e-3, 3.000e-3, 4.500e-3, 1.010e-2,
   1.880e-2, 3.670e-2, 7.510e-2, 1.240e-1, 1.870e-1, 2.630e-1, 3.500e-1, 
   4.420e-1, 5.360e-1, 7.070e-1, 8.510e-1, 9.750e-1, 1.060e+0, 1.120e+0],
  [3.520e-5, 1.000e-4, 6.000e-4, 1.600e-3, 2.600e-3, 4.000e-3, 8.900e-3,
   1.680e-2, 3.350e-2, 6.910e-2, 1.130e-1, 1.670e-1, 2.330e-1, 3.100e-1,
   3.930e-1, 4.790e-1, 6.420e-1, 7.840e-1, 9.060e-1, 9.990e-1, 1.060e+0],
  [9.120e-1, 9.630e-1, 1.121e+0, 1.308e+0, 1.332e+0, 1.327e+0, 1.276e+0,
   1.217e+0, 1.154e+0, 1.099e+0, 1.061e+0, 1.021e+0, 9.790e-1, 9.390e-1,
   9.030e-1, 8.730e-1, 8.260e-1, 7.930e-1, 7.690e-1, 7.530e-1, 7.430e-1],
  [8.800e-1, 9.230e-1, 1.075e+0, 1.265e+0, 1.312e+0, 1.310e+0, 1.264e+0, 
   1.200e+0, 1.128e+0, 1.065e+0, 1.030e+0, 1.000e+0, 9.630e-1, 9.290e-1,
   8.970e-1, 8.680e-1, 8.240e-1, 7.930e-1, 7.690e-1, 7.540e-1, 7.440e-1]])
    
    