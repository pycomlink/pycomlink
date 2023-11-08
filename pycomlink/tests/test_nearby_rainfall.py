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
        pmin = xr.DataArray(np.linspace(0, -30, len(wet)),
                            coords=dict(time=time))
        pmax = xr.DataArray(np.linspace(0, 10, len(wet)),
                            coords=dict(time=time))

        res = nearby_rain.nearby_determine_reference_level(pmin, pmax, wet)

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
            res.values,
            result
        )

    def test_nearby_p_correction(self):
        time = pd.date_range("2020-01-01 00:00", periods=100)
        pmin = xr.DataArray(np.linspace(0, -30, 100),
                            coords=dict(time=time))
        pmax = xr.DataArray(np.linspace(0, 10, 100),
                            coords=dict(time=time))
        wet = xr.DataArray(
            np.concatenate([np.repeat(0, 50), np.repeat(1, 50)]),
            coords=dict(time=time))
        pref = xr.DataArray(np.repeat(-8, 100),
                            coords=dict(time=time))
        pcmin, pcmax = nearby_rain.nearby_correct_received_signals(
            pmin, pmax, wet, pref)

        res_pcmin  =np.array([
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        ,  -8.        ,  -8.        ,
            -8.        ,  -8.        , -15.15151515, -15.45454545,
           -15.75757576, -16.06060606, -16.36363636, -16.66666667,
           -16.96969697, -17.27272727, -17.57575758, -17.87878788,
           -18.18181818, -18.48484848, -18.78787879, -19.09090909,
           -19.39393939, -19.6969697 , -20.        , -20.3030303 ,
           -20.60606061, -20.90909091, -21.21212121, -21.51515152,
           -21.81818182, -22.12121212, -22.42424242, -22.72727273,
           -23.03030303, -23.33333333, -23.63636364, -23.93939394,
           -24.24242424, -24.54545455, -24.84848485, -25.15151515,
           -25.45454545, -25.75757576, -26.06060606, -26.36363636,
           -26.66666667, -26.96969697, -27.27272727, -27.57575758,
           -27.87878788, -28.18181818, -28.48484848, -28.78787879,
           -29.09090909, -29.39393939, -29.6969697 , -30.        ])
        res_pcmax = np.array([
            -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8.,
            -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8.,
            -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8.,
            -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8.,
            -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8.,
            -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8.,
            -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8.,
            -8., -8., -8., -8., -8., -8., -8., -8., -8.])


        np.testing.assert_array_almost_equal(
            pcmin.values,
            res_pcmin
        )

        np.testing.assert_array_almost_equal(
            pcmax.values,
            res_pcmax
        )

    def test_rainfall_calc(self):

        time = pd.date_range("2020-01-01 00:00", periods=100)
        cml_id = ["cml_1"]
        pmin = xr.DataArray(
            np.reshape((np.linspace(0, -30, 100)), (1, 100)),
            coords=dict(cml_id=cml_id, time=time),
        )
        pmax = xr.DataArray(
            np.reshape((np.linspace(0, 10, 100)), (1, 100)),
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

        pcmin, pcmax = nearby_rain.nearby_correct_received_signals(
            pmin, pmax, wet, pref)
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
            F_value = -32.5
        )

        result = np.array([[
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            np.nan    ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            4.34511529,  4.50198968,  4.65914286,  4.81656635,  4.97425221,
            5.13219296,  5.29038154,  5.44881128,  5.60747586,  5.76636931,
            5.92548597,  6.08482042,  6.24436755,  6.40412247,  6.56408051,
            6.72423722,  6.88458835,  7.04512981,  7.20585772,  7.36676834,
            7.52785806,  7.68912345,  7.85056119,  8.01216809,  8.1739411 ,
            8.33587724,  8.49797367,  8.66022765,  8.82263651,  8.9851977 ,
            9.14790873,  9.31076722,  9.47377083,  9.63691733,  9.80020455,
            9.96363036, 10.12719274, 10.29088969, 10.45471929, 10.61867968,
            10.78276904, 10.9469856 , 11.11132765, 11.27579352, 11.44038158,
            11.60509026, 11.76991802, 11.93486336, 12.09992481, 12.26510096]])

        np.testing.assert_array_almost_equal(
            R.values,
            result
        )

        # with given a and b values and with a channel id

        time = pd.date_range("2020-01-01 00:00", periods=100)
        cml_id = ["cml_1"]
        channel_id = ["channel_1"]
        pmin = xr.DataArray(
            np.reshape((np.linspace(0, -30, 100)), (1, 100)),
            coords=dict(cml_id=cml_id, time=time),
        )
        pmax = xr.DataArray(
            np.reshape((np.linspace(0, 10, 100)), (1, 100)),
            coords=dict(cml_id=cml_id, time=time),
        )
        wet = xr.DataArray(
            np.reshape(np.concatenate([np.repeat(0, 50), np.repeat(1, 50)]),
                       (1, 1, 100)),
            coords=dict(cml_id=cml_id, channel_id=channel_id, time=time),
        )
        pref = xr.DataArray(
            np.reshape(np.repeat(-4, 100), (1, 1, 100)),
            coords=dict(cml_id=cml_id, channel_id=channel_id, time=time)
        )
        F = xr.DataArray(
            np.reshape(np.repeat(-20, 100), (1, 1, 100)),
            coords=dict(cml_id=cml_id, channel_id=channel_id, time=time)
        )
        F[0, 0, 20] = -44

        pcmin, pcmax = nearby_rain.nearby_correct_received_signals(
            pmin, pmax, wet, pref)
        length = 5


        R = nearby_rain.nearby_rainfall_retrival(
            pref=pref,
            p_c_min=pcmin,
            p_c_max=pcmax,
            F=F,
            length=length,
            a=1,
            b=1,
            a_b_approximation="ITU_2005",
            waa_max=2.3,
            alpha=0.33,
            F_value = -32.5
        )

        result = np.array([[[
            0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., np.nan, 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0.5842, 0.6042, 0.6242, 0.6442, 0.6642, 0.6842,
            0.7042, 0.7242, 0.7442, 0.7642, 0.7842, 0.8042, 0.8242, 0.8442,
            0.8642, 0.8842, 0.9042, 0.9242, 0.9442, 0.9642, 0.9842, 1.0042,
            1.0242, 1.0442, 1.0642, 1.0842, 1.1042, 1.1242, 1.1442, 1.1642,
            1.1842, 1.2042, 1.2242, 1.2442, 1.2642, 1.2842, 1.3042, 1.3242,
            1.3442, 1.3642, 1.3842, 1.4042, 1.4242, 1.4442, 1.4642, 1.4842,
            1.5042, 1.5242, 1.5442, 1.5642]]])

        np.testing.assert_array_almost_equal(
            R.values,
            result
        )

        # test if a and be are transformed to xarray datasets correctly when
        # channel_id is an additional dimension
        time = pd.date_range("2020-01-01 00:00", periods=100)
        cml_id = ["cml_1"]
        channel_id = ["channel_1"]
        pmin = xr.DataArray(
            [np.reshape((np.linspace(0, -30, 100)), (1, 100))],
            coords=dict(channel_id=channel_id, cml_id=cml_id, time=time),
        )
        pmax = xr.DataArray(
            [np.reshape((np.linspace(0, 10, 100)), (1, 100))],
            coords=dict(channel_id=channel_id, cml_id=cml_id, time=time),
        )
        wet = xr.DataArray(
            [np.reshape(np.concatenate([np.repeat(0, 50), np.repeat(1, 50)]),
                        (1, 100))],
            coords=dict(channel_id=channel_id, cml_id=cml_id, time=time),
        )
        pref = xr.DataArray(
            [np.reshape(np.repeat(-4, 100), (1, 100))],
            coords=dict(channel_id=channel_id, cml_id=cml_id, time=time)
        )
        F = xr.DataArray(
            [np.reshape(np.repeat(-20, 100), (1, 100))],
            coords=dict(channel_id=channel_id, cml_id=cml_id, time=time)
        )
        F[0, 0, 20] = -44

        pcmin, pcmax = nearby_rain.nearby_correct_received_signals(
            pmin, pmax, wet, pref)
        length = 5
        f_GHz = xr.DataArray([[25]])
        pol = xr.DataArray([['Vertical']])

        R = nearby_rain.nearby_rainfall_retrival(
            pref=pref,
            p_c_min=pcmin,
            p_c_max=pcmax,
            F=F,
            length=length,
            f_GHz=f_GHz,
            pol=pol,
            a=None,
            b=None,
            a_b_approximation="ITU_2005",
            waa_max=2.3,
            alpha=0.33,
            F_value = -32.5
        )

        result = np.array([[[
            0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,
            np.nan    ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  4.34511529,  4.50198968,
            4.65914286,  4.81656635,  4.97425221,  5.13219296,
            5.29038154,  5.44881128,  5.60747586,  5.76636931,
            5.92548597,  6.08482042,  6.24436755,  6.40412247,
            6.56408051,  6.72423722,  6.88458835,  7.04512981,
            7.20585772,  7.36676834,  7.52785806,  7.68912345,
            7.85056119,  8.01216809,  8.1739411 ,  8.33587724,
            8.49797367,  8.66022765,  8.82263651,  8.9851977 ,
            9.14790873,  9.31076722,  9.47377083,  9.63691733,
            9.80020455,  9.96363036, 10.12719274, 10.29088969,
            10.45471929, 10.61867968, 10.78276904, 10.9469856 ,
            11.11132765, 11.27579352, 11.44038158, 11.60509026,
            11.76991802, 11.93486336, 12.09992481, 12.26510096]]])

        np.testing.assert_array_almost_equal(
            R.values,
            result
        )

    def test_raises(self):
        time = pd.date_range("2020-01-01 00:00", periods=100)
        cml_id = ["cml_1"]
        pmin = xr.DataArray(
            np.reshape((np.linspace(0, -30, 100)), (1, 100)),
            coords=dict(cml_id=cml_id, time=time),
        )
        pmax = xr.DataArray(
            np.reshape((np.linspace(0, 10, 100)), (1, 100)),
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
        F[0, 20] = -44

        pcmin, pcmax = nearby_rain.nearby_correct_received_signals(
            pmin, pmax, wet, pref)
        length = 5
        f_GHz = xr.DataArray([25])
        pol = xr.DataArray(['Vertical'])


        with self.assertRaises(ValueError):
            nearby_rain.nearby_rainfall_retrival(
                pref=pref,
                p_c_min=pcmin,
                p_c_max=pcmax,
                F=F,
                length=length,
                f_GHz=f_GHz,
                pol=pol,
                a=None,
                b=1,
                a_b_approximation="ITU_2005",
                waa_max=2.3,
                alpha=0.33,
                F_value=-32.5
            )

        with self.assertRaises(IndexError):
            nearby_rain.nearby_rainfall_retrival(
                pref=pref,
                p_c_min=pcmin,
                p_c_max=pcmax,
                F=F,
                length=length,
                f_GHz = f_GHz,
                pol = xr.DataArray(['Vertical','Vertical']),
                a = None,
                b = None,
                a_b_approximation = "ITU_2005",
                waa_max = 2.3,
                alpha = 0.33,
                F_value=-32.5
            )
