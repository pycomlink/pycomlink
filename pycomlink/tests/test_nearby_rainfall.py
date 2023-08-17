import unittest
import pandas as pd
import numpy as np
import xarray as xr

import pycomlink.processing.nearby_rain_retrival as nearby_rain


class Test_nearby_wetdry_approach(unittest.TestCase):

    def test_nearby_reference_level(self):
        time = pd.date_range("2020-01-01 00:00", periods=200)

        wet = xr.DataArray(
            np.concatenate([np.repeat(1, 100), np.repeat(0, 100)]),
            coords=dict(time=time))
        pmin1 = xr.DataArray(np.linspace(0, -10, len(wet)),
                             coords=dict(time=time))
        pmin2 = xr.DataArray(np.linspace(0, -30, len(wet)),
                             coords=dict(time=time))
        pmax = xr.DataArray(np.linspace(0, 10, len(wet)),
                            coords=dict(time=time))

        res1 = nearby_rain.nearby_determine_reference_level(wet, pmin1)
        res2 = nearby_rain.nearby_determine_reference_level(wet, pmin2, pmax)

        result = np.array(
            [np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan,
             -5.02512563, -5.05025126, -5.07537688, -5.10050251, -5.12562814,
             -5.15075377, -5.1758794, -5.20100503, -5.22613065, -5.25125628,
             -5.27638191, -5.30150754, -5.32663317, -5.35175879, -5.37688442,
             -5.40201005, -5.42713568, -5.45226131, -5.47738693, -5.50251256,
             -5.52763819, -5.55276382, -5.57788945, -5.60301508, -5.6281407,
             -5.65326633, -5.67839196, -5.70351759, -5.72864322, -5.75376884,
             -5.77889447, -5.8040201, -5.82914573, -5.85427136, -5.87939698,
             -5.90452261, -5.92964824, -5.95477387, -5.9798995, -6.00502513,
             -6.03015075, -6.05527638, -6.08040201, -6.10552764, -6.13065327,
             -6.15577889, -6.18090452, -6.20603015, -6.23115578, -6.25628141,
             -6.28140704, -6.30653266, -6.33165829, -6.35678392, -6.38190955,
             -6.40703518, -6.4321608, -6.45728643, -6.48241206, -6.50753769,
             -6.53266332, -6.55778894, -6.58291457, -6.6080402, -6.63316583,
             -6.65829146, -6.68341709, -6.70854271, -6.73366834, -6.75879397,
             -6.7839196, -6.80904523, -6.83417085, -6.85929648, -6.88442211,
             -6.90954774, -6.93467337, -6.95979899, -6.98492462, -7.01005025,
             -7.03517588, -7.06030151, -7.08542714, -7.11055276, -7.13567839,
             -7.16080402, -7.18592965, -7.21105528, -7.2361809, -7.26130653,
             -7.28643216, -7.31155779, -7.33668342, -7.36180905, -7.38693467,
             -7.4120603, -7.46231156, -7.51256281, -7.56281407, -7.61306533])

        np.testing.assert_array_almost_equal(
            res1.values,
            result
        )

        np.testing.assert_array_almost_equal(
            res2.values,
            result
        )

    def test_nearby_p_correction(self):
        time = pd.date_range("2020-01-01 00:00", periods=100)
        pmin = xr.DataArray(np.linspace(0, -10, 100),
                            coords=dict(time=time))
        wet = xr.DataArray(
            np.concatenate([np.repeat(0, 50), np.repeat(1, 50)]),
            coords=dict(time=time))
        pref = xr.DataArray(np.repeat(-8, 100),
                            coords=dict(time=time))
        res_pcmin, res_pcmax = nearby_rain.nearby_correct_recieved_signals(
            pmin, wet, pref)

        result = np.array([-8., -8., -8., -8., -8., -8., -8., -8.,
                           -8., -8., -8., -8., -8., -8., -8., -8.,
                           -8., -8., -8., -8., -8., -8., -8., -8.,
                           -8., -8., -8., -8., -8., -8., -8., -8.,
                           -8., -8., -8., -8., -8., -8., -8., -8.,
                           -8., -8., -8., -8., -8., -8., -8., -8.,
                           -8., -8., -8., -8., -8., -8., -8., -8.,
                           -8., -8., -8., -8., -8., -8., -8., -8.,
                           -8., -8., -8., -8., -8., -8., -8., -8.,
                           -8., -8., -8., -8., -8., -8., -8., -8.,
                           -8.08080808, -8.18181818, -8.28282828, -8.38383838,
                           -8.48484848, -8.58585859, -8.68686869, -8.78787879,
                           -8.88888889, -8.98989899, -9.09090909, -9.19191919,
                           -9.29292929, -9.39393939, -9.49494949, -9.5959596,
                           -9.6969697, -9.7979798, -9.8989899, -10.])

        np.testing.assert_array_almost_equal(
            res_pcmin.values,
            result
        )

        np.testing.assert_array_almost_equal(
            res_pcmax.values,
            result
        )

    def test_rainfall_calc(self):

        time = pd.date_range("2020-01-01 00:00", periods=100)
        cml_id = ["cml_1"]
        pmin = xr.DataArray(
            np.reshape((np.linspace(0, -10, 100)), (1, 100)),
            coords=dict(cml_id=cml_id, time=time),
        )
        wet = xr.DataArray(
            np.reshape(np.concatenate([np.repeat(0, 50), np.repeat(1, 50)]),
                       (1, 100)),
            coords=dict(cml_id=cml_id, time=time),
        )
        pref = xr.DataArray(
            np.reshape(np.repeat(-4, 100), (1, 100)),
            coords=dict(cml_id=cml_id, time=time)
        )
        F = xr.DataArray(
            np.reshape(np.repeat(-20, 100), (1, 100)),
            coords=dict(cml_id=cml_id, time=time)
        )
        F[0,20] = -44

        pcmin, pcmax = nearby_rain.nearby_correct_recieved_signals(
            pmin, wet, pref)
        length = 5
        f_GHz = xr.DataArray([25])
        pol = xr.DataArray(['Vertical'])

        R = nearby_rain.nearby_rainfall_retrival(
            pref=pref,
            p_c_min=pcmin,
            p_c_max=pcmax,
            F=F,
            length=length,
            f_GHz = f_GHz,
            pol = pol,
            a = None,
            b = None,
            a_b_approximation = "ITU_2005",
            waa_max = 2.3,
            alpha = 0.33,
            F_value_correction = True
        )

        result = np.array([[
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
                np.nan, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.10367216, 0.25556143,
            0.40242787, 0.54640596, 0.68835262, 0.82873479, 0.96784663,
            1.1058904 , 1.24301365, 1.37932866, 1.51492374, 1.64987012,
            1.78422644, 1.9180418 , 2.05135791, 2.18421063, 2.31663111,
            2.44864666, 2.5802814 , 2.71155681, 2.84249211, 2.97310463,
            3.10341004, 3.2334226 , 3.36315532, 3.49262013, 3.62182803,
            3.75078913, 3.87951283, 4.00800783, 4.13628224, 4.26434363,
            4.39219907, 4.51985519, 4.64731821, 4.774594  , 4.90168807]])

        np.testing.assert_array_almost_equal(
            R.values,
            result
        )




