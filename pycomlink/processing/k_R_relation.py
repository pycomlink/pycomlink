from __future__ import division
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from .xarray_wrapper import xarray_apply_along_time_dim


############################################
# Functions for k-R power law calculations #
############################################


@xarray_apply_along_time_dim()
def calc_R_from_A(
    A,
    L_km,
    f_GHz=None,
    pol=None,
    a=None,
    b=None,
    a_b_approximation="ITU_2005",
    R_min=0.1,
):
    """Calculate rain rate from path-integrated attenuation using the k-R power law

    Note that either `f_GHz` and `pol` or `a` and `b` have to be provided. The former
    option calculates the parameters `a` and `b` for the k-R power law internally
    based on frequency and polarization.

    Parameters
    ----------
    A : float or iterable of float
        Path-integrated attenuation of microwave link signal
    L_km : float
        Length of the link in km
    f_GHz : float, np.array, or xr.DataArray optional
        Frequency in GHz. If provided together with `pol`, it will be used to
        derive the parameters a and b for the k-R power law.
    pol : string, np.array or xr.DataArray optional
        Polarization, that is either 'horizontal' for horizontal or 'vertical'
        for vertical. 'H', 'h' and 'Horizontal' as well as 'V', 'v' and 'Vertical'
        are also allowed. Has to be provided together with `f_GHz`. It will be
        used to derive the parameters a and b for the k-R power law. Must have
        same shape as f_GHz or be a str. If it is a str, it will be expanded to
        the shape of f_GHz.
    a : float, optional
        Parameter of A-R relationship
    b : float, optional
        Parameter of A-R relationship
    a_b_approximation : string
        Specifies which approximation for the k-R power law shall be used. See the
        function `a_b` for details.
    R_min : float
        Minimal rain rate in mm/h. Everything below will be set to zero.

    Returns
    -------
    float or iterable of float
        Rain rate

    Note
    ----
    The A-R and k-R relation are defined as

    .. math:: A = k L_{km} = aR^{b} L_{km}

    where `A` is the path-integrated attenuation in dB and `k` is the specific
    attenuation in dB/km.

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

    A = np.atleast_1d(A).astype(float)
    R = np.zeros_like(A)

    nan_index = np.isnan(A)
    R[nan_index] = np.nan

    # This ignores the numpy warning stemming from A >=0 where A contains NaNs
    with np.errstate(invalid="ignore"):
        R[~nan_index & (A >= 0)] = (A[~nan_index & (A >= 0)] / (a * L_km)) ** (1 / b)
        R[~nan_index & (R < R_min)] = 0
    return R


def calc_R_from_A_min_max(
    Ar_max, L, f_GHz=None, a=None, b=None, pol="H", R_min=0.1, k=90
):
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
    R = np.zeros_like(Ar_max, dtype="float")

    nan_index = np.isnan(Ar_max)
    R[nan_index] = np.nan

    # This ignores the numpy warning stemming from A >=0 where A contains NaNs
    with np.errstate(invalid="ignore"):
        R[~nan_index & (Ar_max >= 0)] = (
            Ar_max[~nan_index & (Ar_max >= 0)] / (a_max * L)
        ) ** (1.0 / b)
        R[~nan_index & (R < R_min)] = 0.0
    return R


def a_b(f_GHz, pol, approx_type="ITU_2005"):
    """Approximation of parameters a and b for k-R power law

    Parameters
    ----------
    f_GHz : int, float, np.array or xr.DataArray
        Frequency of the microwave link(s) in GHz.
    pol : str, np.array or xr.DataArray
        Polarization, that is either 'horizontal' for horizontal or 'vertical'
        for vertical. 'H', 'h' and 'Horizontal' as well as 'V', 'v' and 'Vertical'
        are also allowed. Must have same shape as f_GHz or be a str. If it is a
        str, it will be expanded to the shape of f_GHz.
    approx_type : str, optional
        Approximation type (the default is 'ITU_2005', which implies parameter
        approximation using a table recommanded by ITU in 2005. An older version
        of 2003 is available via 'ITU_2003'.)

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
    use of a table recommanded by ITU [4]. There are two versions available,
    P.838-2 (04/2003) and P.838-3 (03/2005).

    References
    ----------
    .. [4] ITU, "ITU-R: Specific attenuation model for rain for use in
        prediction methods", International Telecommunication Union, P.838-2 (04/2003) P.838-3 (03/2005)

    """
    if isinstance(f_GHz, xr.DataArray):
        return_xarray = True
        f_GHz_coords = f_GHz.coords
    else:
        return_xarray = False

    f_GHz = xr.DataArray(f_GHz)

    if isinstance(pol, str):
        pol = xr.full_like(f_GHz, pol, dtype=object)
    pol = xr.DataArray(pol)

    if f_GHz.shape != pol.shape:
        raise ValueError("Frequency and polarization must have identical shape.")

    f_GHz = np.atleast_1d(f_GHz)
    pol = np.atleast_1d(pol)

    pol_v_str_variants = ['v','V','vertical','Vertical']
    pol_h_str_variants = ['h','H','horizontal','Horizontal']

    if f_GHz.min() < 1 or f_GHz.max() > 100:
        raise ValueError("Frequency must be between 1 Ghz and 100 GHz.")
    if not np.isin(pol,pol_v_str_variants+pol_h_str_variants).all():
        raise ValueError("Polarization must be V, v, Vertical, vertical, H,"
                         "Horizontal or horizontal.")
    # select ITU table
    if approx_type == "ITU_2003":
        ITU_table = ITU_table_2003.copy()
    elif approx_type == "ITU_2005":
        ITU_table = ITU_table_2005.copy()
    else:
        raise ValueError("Approximation type not available.")

    interp_a_v = interp1d(ITU_table[0, :], ITU_table[2, :], kind="cubic")
    interp_b_v = interp1d(ITU_table[0, :], ITU_table[4, :], kind="cubic")
    interp_a_h = interp1d(ITU_table[0, :], ITU_table[1, :], kind="cubic")
    interp_b_h = interp1d(ITU_table[0, :], ITU_table[3, :], kind="cubic")

    a_v = interp_a_v(f_GHz)
    b_v = interp_b_v(f_GHz)
    a_h = interp_a_h(f_GHz)
    b_h = interp_b_h(f_GHz)

    a = np.full_like(f_GHz, fill_value=np.nan, dtype=float)
    b = np.full_like(f_GHz, fill_value=np.nan, dtype=float)

    pol_mask_v = np.isin(pol, pol_v_str_variants)
    pol_mask_h = np.isin(pol, pol_h_str_variants)

    a[pol_mask_h] = a_h[pol_mask_h]
    a[pol_mask_v] = a_v[pol_mask_v]
    b[pol_mask_h] = b_h[pol_mask_h]
    b[pol_mask_v] = b_v[pol_mask_v]

    # If f_GHz is a xr.DataArray a and b should be also returned as xr.DataArray
    # with identical coordinates
    if return_xarray:
        a = xr.DataArray(a, coords=f_GHz_coords)
        b = xr.DataArray(b, coords=f_GHz_coords)

    return a, b


# fmt: off

# ITU recommendations table from 2005
# (row order: frequency, k_H, k_V, alpha_H, alpha_V)
ITU_table_2005 = np.array(
    [
        [1.0000e+00, 1.5000e+00, 2.0000e+00, 2.5000e+00, 3.0000e+00,
         3.5000e+00, 4.0000e+00, 4.5000e+00, 5.0000e+00, 5.5000e+00,
         6.0000e+00, 7.0000e+00, 8.0000e+00, 9.0000e+00, 1.0000e+01,
         1.1000e+01, 1.2000e+01, 1.3000e+01, 1.4000e+01, 1.5000e+01,
         1.6000e+01, 1.7000e+01, 1.8000e+01, 1.9000e+01, 2.0000e+01,
         2.1000e+01, 2.2000e+01, 2.3000e+01, 2.4000e+01, 2.5000e+01,
         2.6000e+01, 2.7000e+01, 2.8000e+01, 2.9000e+01, 3.0000e+01,
         3.1000e+01, 3.2000e+01, 3.3000e+01, 3.4000e+01, 3.5000e+01,
         3.6000e+01, 3.7000e+01, 3.8000e+01, 3.9000e+01, 4.0000e+01,
         4.1000e+01, 4.2000e+01, 4.3000e+01, 4.4000e+01, 4.5000e+01,
         4.6000e+01, 4.7000e+01, 4.8000e+01, 4.9000e+01, 5.0000e+01,
         5.1000e+01, 5.2000e+01, 5.3000e+01, 5.4000e+01, 5.5000e+01,
         5.6000e+01, 5.7000e+01, 5.8000e+01, 5.9000e+01, 6.0000e+01,
         6.1000e+01, 6.2000e+01, 6.3000e+01, 6.4000e+01, 6.5000e+01,
         6.6000e+01, 6.7000e+01, 6.8000e+01, 6.9000e+01, 7.0000e+01,
         7.1000e+01, 7.2000e+01, 7.3000e+01, 7.4000e+01, 7.5000e+01,
         7.6000e+01, 7.7000e+01, 7.8000e+01, 7.9000e+01, 8.0000e+01,
         8.1000e+01, 8.2000e+01, 8.3000e+01, 8.4000e+01, 8.5000e+01,
         8.6000e+01, 8.7000e+01, 8.8000e+01, 8.9000e+01, 9.0000e+01,
         9.1000e+01, 9.2000e+01, 9.3000e+01, 9.4000e+01, 9.5000e+01,
         9.6000e+01, 9.7000e+01, 9.8000e+01, 9.9000e+01, 1.0000e+02],
        [2.5900e-05, 4.4300e-05, 8.4700e-05, 1.3210e-04, 1.3900e-04,
         1.1550e-04, 1.0710e-04, 1.3400e-04, 2.1620e-04, 3.9090e-04,
         7.0560e-04, 1.9150e-03, 4.1150e-03, 7.5350e-03, 1.2170e-02,
         1.7720e-02, 2.3860e-02, 3.0410e-02, 3.7380e-02, 4.4810e-02,
         5.2820e-02, 6.1460e-02, 7.0780e-02, 8.0840e-02, 9.1640e-02,
         1.0320e-01, 1.1550e-01, 1.2860e-01, 1.4250e-01, 1.5710e-01,
         1.7240e-01, 1.8840e-01, 2.0510e-01, 2.2240e-01, 2.4030e-01,
         2.5880e-01, 2.7780e-01, 2.9720e-01, 3.1710e-01, 3.3740e-01,
         3.5800e-01, 3.7890e-01, 4.0010e-01, 4.2150e-01, 4.4310e-01,
         4.6470e-01, 4.8650e-01, 5.0840e-01, 5.3020e-01, 5.5210e-01,
         5.7380e-01, 5.9560e-01, 6.1720e-01, 6.3860e-01, 6.6000e-01,
         6.8110e-01, 7.0200e-01, 7.2280e-01, 7.4330e-01, 7.6350e-01,
         7.8350e-01, 8.0320e-01, 8.2260e-01, 8.4180e-01, 8.6060e-01,
         8.7910e-01, 8.9740e-01, 9.1530e-01, 9.3280e-01, 9.5010e-01,
         9.6700e-01, 9.8360e-01, 9.9990e-01, 1.0159e+00, 1.0315e+00,
         1.0468e+00, 1.0618e+00, 1.0764e+00, 1.0908e+00, 1.1048e+00,
         1.1185e+00, 1.1320e+00, 1.1451e+00, 1.1579e+00, 1.1704e+00,
         1.1827e+00, 1.1946e+00, 1.2063e+00, 1.2177e+00, 1.2289e+00,
         1.2398e+00, 1.2504e+00, 1.2607e+00, 1.2708e+00, 1.2807e+00,
         1.2903e+00, 1.2997e+00, 1.3089e+00, 1.3179e+00, 1.3266e+00,
         1.3351e+00, 1.3434e+00, 1.3515e+00, 1.3594e+00, 1.3671e+00],
        [3.0800e-05, 5.7400e-05, 9.9800e-05, 1.4640e-04, 1.9420e-04,
         2.3460e-04, 2.4610e-04, 2.3470e-04, 2.4280e-04, 3.1150e-04,
         4.8780e-04, 1.4250e-03, 3.4500e-03, 6.6910e-03, 1.1290e-02,
         1.7310e-02, 2.4550e-02, 3.2660e-02, 4.1260e-02, 5.0080e-02,
         5.8990e-02, 6.7970e-02, 7.7080e-02, 8.6420e-02, 9.6110e-02,
         1.0630e-01, 1.1700e-01, 1.2840e-01, 1.4040e-01, 1.5330e-01,
         1.6690e-01, 1.8130e-01, 1.9640e-01, 2.1240e-01, 2.2910e-01,
         2.4650e-01, 2.6460e-01, 2.8330e-01, 3.0260e-01, 3.2240e-01,
         3.4270e-01, 3.6330e-01, 3.8440e-01, 4.0580e-01, 4.2740e-01,
         4.4920e-01, 4.7120e-01, 4.9320e-01, 5.1530e-01, 5.3750e-01,
         5.5960e-01, 5.8170e-01, 6.0370e-01, 6.2550e-01, 6.4720e-01,
         6.6870e-01, 6.9010e-01, 7.1120e-01, 7.3210e-01, 7.5270e-01,
         7.7300e-01, 7.9310e-01, 8.1290e-01, 8.3240e-01, 8.5150e-01,
         8.7040e-01, 8.8890e-01, 9.0710e-01, 9.2500e-01, 9.4250e-01,
         9.5980e-01, 9.7670e-01, 9.9320e-01, 1.0094e+00, 1.0253e+00,
         1.0409e+00, 1.0561e+00, 1.0711e+00, 1.0857e+00, 1.1000e+00,
         1.1139e+00, 1.1276e+00, 1.1410e+00, 1.1541e+00, 1.1668e+00,
         1.1793e+00, 1.1915e+00, 1.2034e+00, 1.2151e+00, 1.2265e+00,
         1.2376e+00, 1.2484e+00, 1.2590e+00, 1.2694e+00, 1.2795e+00,
         1.2893e+00, 1.2989e+00, 1.3083e+00, 1.3175e+00, 1.3265e+00,
         1.3352e+00, 1.3437e+00, 1.3520e+00, 1.3601e+00, 1.3680e+00],
        [9.6910e-01, 1.0185e+00, 1.0664e+00, 1.1209e+00, 1.2322e+00,
         1.4189e+00, 1.6009e+00, 1.6948e+00, 1.6969e+00, 1.6499e+00,
         1.5900e+00, 1.4810e+00, 1.3905e+00, 1.3155e+00, 1.2571e+00,
         1.2140e+00, 1.1825e+00, 1.1586e+00, 1.1396e+00, 1.1233e+00,
         1.1086e+00, 1.0949e+00, 1.0818e+00, 1.0691e+00, 1.0568e+00,
         1.0447e+00, 1.0329e+00, 1.0214e+00, 1.0101e+00, 9.9910e-01,
         9.8840e-01, 9.7800e-01, 9.6790e-01, 9.5800e-01, 9.4850e-01,
         9.3920e-01, 9.3020e-01, 9.2140e-01, 9.1290e-01, 9.0470e-01,
         8.9670e-01, 8.8900e-01, 8.8160e-01, 8.7430e-01, 8.6730e-01,
         8.6050e-01, 8.5390e-01, 8.4760e-01, 8.4140e-01, 8.3550e-01,
         8.2970e-01, 8.2410e-01, 8.1870e-01, 8.1340e-01, 8.0840e-01,
         8.0340e-01, 7.9870e-01, 7.9410e-01, 7.8960e-01, 7.8530e-01,
         7.8110e-01, 7.7710e-01, 7.7310e-01, 7.6930e-01, 7.6560e-01,
         7.6210e-01, 7.5860e-01, 7.5520e-01, 7.5200e-01, 7.4880e-01,
         7.4580e-01, 7.4280e-01, 7.4000e-01, 7.3720e-01, 7.3450e-01,
         7.3180e-01, 7.2930e-01, 7.2680e-01, 7.2440e-01, 7.2210e-01,
         7.1990e-01, 7.1770e-01, 7.1560e-01, 7.1350e-01, 7.1150e-01,
         7.0960e-01, 7.0770e-01, 7.0580e-01, 7.0400e-01, 7.0230e-01,
         7.0060e-01, 6.9900e-01, 6.9740e-01, 6.9590e-01, 6.9440e-01,
         6.9290e-01, 6.9150e-01, 6.9010e-01, 6.8880e-01, 6.8750e-01,
         6.8620e-01, 6.8500e-01, 6.8380e-01, 6.8260e-01, 6.8150e-01],
        [8.5920e-01, 8.9570e-01, 9.4900e-01, 1.0085e+00, 1.0688e+00,
         1.1387e+00, 1.2476e+00, 1.3987e+00, 1.5317e+00, 1.5882e+00,
         1.5728e+00, 1.4745e+00, 1.3797e+00, 1.2895e+00, 1.2156e+00,
         1.1617e+00, 1.1216e+00, 1.0901e+00, 1.0646e+00, 1.0440e+00,
         1.0273e+00, 1.0137e+00, 1.0025e+00, 9.9300e-01, 9.8470e-01,
         9.7710e-01, 9.7000e-01, 9.6300e-01, 9.5610e-01, 9.4910e-01,
         9.4210e-01, 9.3490e-01, 9.2770e-01, 9.2030e-01, 9.1290e-01,
         9.0550e-01, 8.9810e-01, 8.9070e-01, 8.8340e-01, 8.7610e-01,
         8.6900e-01, 8.6210e-01, 8.5520e-01, 8.4860e-01, 8.4210e-01,
         8.3570e-01, 8.2960e-01, 8.2360e-01, 8.1790e-01, 8.1230e-01,
         8.0690e-01, 8.0170e-01, 7.9670e-01, 7.9180e-01, 7.8710e-01,
         7.8260e-01, 7.7830e-01, 7.7410e-01, 7.7000e-01, 7.6610e-01,
         7.6230e-01, 7.5870e-01, 7.5520e-01, 7.5180e-01, 7.4860e-01,
         7.4540e-01, 7.4240e-01, 7.3950e-01, 7.3660e-01, 7.3390e-01,
         7.3130e-01, 7.2870e-01, 7.2620e-01, 7.2380e-01, 7.2150e-01,
         7.1930e-01, 7.1710e-01, 7.1500e-01, 7.1300e-01, 7.1100e-01,
         7.0910e-01, 7.0730e-01, 7.0550e-01, 7.0380e-01, 7.0210e-01,
         7.0040e-01, 6.9880e-01, 6.9730e-01, 6.9580e-01, 6.9430e-01,
         6.9290e-01, 6.9150e-01, 6.9020e-01, 6.8890e-01, 6.8760e-01,
         6.8640e-01, 6.8520e-01, 6.8400e-01, 6.8280e-01, 6.8170e-01,
         6.8060e-01, 6.7960e-01, 6.7850e-01, 6.7750e-01, 6.7650e-01]
    ]
)

ITU_table_2003 = np.array(
    [
        [1.000e0,  2.000e0,  4.000e0,  6.000e0,  7.000e0,  8.000e0,  1.000e1,  1.200e1,  1.500e1,  2.000e1,  2.500e1,  3.000e1,  3.500e1,  4.000e1,  4.500e1,  5.000e1,  6.000e1,  7.000e1,  8.000e1,  9.000e1,  1.000e2],
        [3.870e-5, 2.000e-4, 6.000e-4, 1.800e-3, 3.000e-3, 4.500e-3, 1.010e-2, 1.880e-2, 3.670e-2, 7.510e-2, 1.240e-1, 1.870e-1, 2.630e-1, 3.500e-1, 4.420e-1, 5.360e-1, 7.070e-1, 8.510e-1, 9.750e-1, 1.060e0,  1.120e0],
        [3.520e-5, 1.000e-4, 6.000e-4, 1.600e-3, 2.600e-3, 4.000e-3, 8.900e-3, 1.680e-2, 3.350e-2, 6.910e-2, 1.130e-1, 1.670e-1, 2.330e-1, 3.100e-1, 3.930e-1, 4.790e-1, 6.420e-1, 7.840e-1, 9.060e-1, 9.990e-1, 1.060e0],
        [9.120e-1, 9.630e-1, 1.121e0,  1.308e0,  1.332e0,  1.327e0,  1.276e0,  1.217e0,  1.154e0,  1.099e0,  1.061e0,  1.021e0,  9.790e-1, 9.390e-1, 9.030e-1, 8.730e-1, 8.260e-1, 7.930e-1, 7.690e-1, 7.530e-1, 7.430e-1],
        [8.800e-1, 9.230e-1, 1.075e0,  1.265e0,  1.312e0,  1.310e0,  1.264e0,  1.200e0,  1.128e0,  1.065e0,  1.030e0,  1.000e0,  9.630e-1, 9.290e-1, 8.970e-1, 8.680e-1, 8.240e-1, 7.930e-1, 7.690e-1, 7.540e-1, 7.440e-1],
    ]
)

# fmt: on

