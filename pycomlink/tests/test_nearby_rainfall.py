import unittest
import pandas as pd
import numpy as np
import xarray as xr

import pycomlink.processing.nearby_rain_retrival as nearby_rain
import pycomlink.processing.wet_dry.nearby_wetdry as nearby_wetdry
from pycomlink.io.examples import get_example_data_path

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
        pmax = xr.DataArray(np.linspace(0, -20, 100),
                            coords=dict(time=time))
        wet = xr.DataArray(
            np.concatenate([np.repeat(0, 50), np.repeat(1, 50)]),
            coords=dict(time=time))
        wet[85] = 0
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
            -25.45454545,           -8, -26.06060606, -26.36363636,
            -26.66666667, -26.96969697, -27.27272727, -27.57575758,
            -27.87878788, -28.18181818, -28.48484848, -28.78787879,
            -29.09090909, -29.39393939, -29.6969697 , -30.        ])
        res_pcmax = np.array([
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
            -8.        ,  -8.        , -10.1010101 , -10.3030303 ,
            -10.50505051, -10.70707071, -10.90909091, -11.11111111,
            -11.31313131, -11.51515152, -11.71717172, -11.91919192,
            -12.12121212, -12.32323232, -12.52525253, -12.72727273,
            -12.92929293, -13.13131313, -13.33333333, -13.53535354,
            -13.73737374, -13.93939394, -14.14141414, -14.34343434,
            -14.54545455, -14.74747475, -14.94949495, -15.15151515,
            -15.35353535, -15.55555556, -15.75757576, -15.95959596,
            -16.16161616, -16.36363636, -16.56565657, -16.76767677,
            -16.96969697,  -8.        , -17.37373737, -17.57575758,
            -17.77777778, -17.97979798, -18.18181818, -18.38383838,
            -18.58585859, -18.78787879, -18.98989899, -19.19191919,
            -19.39393939, -19.5959596 , -19.7979798 , -20.        ])


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
            np.reshape((np.linspace(0, -20, 100)), (1, 100)),
            coords=dict(cml_id=cml_id, time=time),
        )
        wet = xr.DataArray(
            np.reshape(np.concatenate([np.repeat(0, 50), np.repeat(1, 50)]),
                       (1, 100)),
            coords=dict(cml_id=cml_id, time=time),
        )
        wet.loc[dict(cml_id="cml_1")][85] = 0
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
        f_GHz = np.array(25)
        pol = np.array('v')

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
            F_value_threshold = -32.5
        )

        result = np.array([[
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                np.nan,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            7.96550311,  8.32540146,  8.68612906,  9.04765236,  9.40994058,
            9.77296534, 10.13670042, 10.50112148, 10.86620588, 11.23193249,
            11.59828156, 11.96523457, 12.3327741 , 12.70088379, 13.06954817,
            13.43875265, 13.80848339, 14.1787273 , 14.54947193, 14.92070547,
            15.29241665, 15.66459473, 16.03722948, 16.41031112, 16.78383027,
            17.15777799, 17.53214568, 17.90692511, 18.28210837, 18.65768786,
            19.03365627, 19.41000656, 19.78673197, 20.16382595, 20.54128222,
            0.         , 21.29725748, 21.67576492, 22.05461152, 22.43379198,
            22.81330115, 23.19313405, 23.57328585, 23.95375188, 24.33452759,
            24.71560858, 25.09699057, 25.4786694 , 25.86064103, 26.24290154]])

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
            np.reshape((np.linspace(0, -20, 100)), (1, 100)),
            coords=dict(cml_id=cml_id, time=time),
        )
        wet = xr.DataArray(
            np.reshape(np.concatenate([np.repeat(0, 50), np.repeat(1, 50)]),
                       (1, 1, 100)),
            coords=dict(cml_id=cml_id, channel_id=channel_id, time=time),
        )
        wet.loc[dict(cml_id="cml_1",channel_id="channel_1")][85] = 0
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
            F_value_threshold = -32.5
        )

        result = np.array([[[
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
            1.09353535, 1.14060606, 1.18767677, 1.23474747, 1.28181818,
            1.32888889, 1.3759596 , 1.4230303 , 1.47010101, 1.51717172,
            1.56424242, 1.61131313, 1.65838384, 1.70545455, 1.75252525,
            1.79959596, 1.84666667, 1.89373737, 1.94080808, 1.98787879,
            2.03494949, 2.0820202 , 2.12909091, 2.17616162, 2.22323232,
            2.27030303, 2.31737374, 2.36444444, 2.41151515, 2.45858586,
            2.50565657, 2.55272727, 2.59979798, 2.64686869, 2.69393939,
            0.        , 2.78808081, 2.83515152, 2.88222222, 2.92929293,
            2.97636364, 3.02343434, 3.07050505, 3.11757576, 3.16464646,
            3.21171717, 3.25878788, 3.30585859, 3.35292929, 3.4       ]]])

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
            [np.reshape((np.linspace(0, -20, 100)), (1, 100))],
            coords=dict(channel_id=channel_id, cml_id=cml_id, time=time),
        )
        wet = xr.DataArray(
            [np.reshape(np.concatenate([np.repeat(0, 50), np.repeat(1, 50)]),
                        (1, 100))],
            coords=dict(channel_id=channel_id, cml_id=cml_id, time=time),
        )
        wet.loc[dict(cml_id="cml_1",channel_id="channel_1")][85] = 0
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
        f_GHz = 25
        pol = 'v'

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
            F_value_threshold = -32.5
        )

        result = np.array([[[
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                np.nan,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            7.96550311,  8.32540146,  8.68612906,  9.04765236,  9.40994058,
            9.77296534, 10.13670042, 10.50112148, 10.86620588, 11.23193249,
            11.59828156, 11.96523457, 12.3327741 , 12.70088379, 13.06954817,
            13.43875265, 13.80848339, 14.1787273 , 14.54947193, 14.92070547,
            15.29241665, 15.66459473, 16.03722948, 16.41031112, 16.78383027,
            17.15777799, 17.53214568, 17.90692511, 18.28210837, 18.65768786,
            19.03365627, 19.41000656, 19.78673197, 20.16382595, 20.54128222,
            0.         , 21.29725748, 21.67576492, 22.05461152, 22.43379198,
            22.81330115, 23.19313405, 23.57328585, 23.95375188, 24.33452759,
            24.71560858, 25.09699057, 25.4786694 , 25.86064103, 26.24290154]]])

        np.testing.assert_array_almost_equal(
            R.values,
            result
        )

    def test_with_real_data(self):
        # get data
        data_path = get_example_data_path()
        cmls = xr.open_dataset(data_path + "/example_cml_data.nc")
        cmls = cmls.isel(cml_id=range(100)).sel(time=slice("2018-05-13 12:00", "2018-05-14 12:00"))
        # prepare data
        cmls["rsl"] = cmls["rsl"].where(cmls.rsl > -99.9)
        cmls["tsl"] = cmls["tsl"].where(cmls.tsl < 255.0)
        cmls["rsl"] = cmls.rsl.interpolate_na(dim="time", method="linear", max_gap="5min")
        cmls["tsl"] = cmls.tsl.interpolate_na(dim="time", method="linear", max_gap="5min")
        rstl = cmls.rsl - cmls.tsl
        pmin = rstl.resample(time="15min").min()
        pmax = rstl.resample(time="15min").max()
        # wet dry detection
        ds_dist = nearby_wetdry.calc_distance_between_cml_endpoints(
            cml_ids=cmls.cml_id.values,
            site_a_latitude=cmls.site_a_latitude,
            site_a_longitude=cmls.site_a_longitude,
            site_b_latitude=cmls.site_b_latitude,
            site_b_longitude=cmls.site_b_longitude,
        )
        r = 15  # radius in km
        ds_dist["within_r"] = (
            (ds_dist.a_to_all_a < r)
            & (ds_dist.a_to_all_b < r)
            & (ds_dist.b_to_all_a < r)
            & (ds_dist.b_to_all_b < r)
        )
        wet, F = nearby_wetdry.nearby_wetdry(
            pmin=pmin,
            ds_dist=ds_dist,
            radius=15,
            thresh_median_P=-1.4,
            thresh_median_PL=-0.7,
            min_links=3,
            interval=15,
            timeperiod=24,
        )
        # baseline
        pref = nearby_rain.nearby_determine_reference_level(pmin, pmax, wet, n_average_dry=96, min_periods=1)
        # p correction
        p_c_min, p_c_max = nearby_rain.nearby_correct_received_signals(pmin, pmax, wet, pref)
        # rainfall retrival
        R_calculated = nearby_rain.nearby_rainfall_retrival(
            pref,
            p_c_min,
            p_c_max,
            F,
            length=pmin.length,
            f_GHz=pmin.frequency / 1e9,
            pol=pmin.polarization,
            waa_max=2.3,
            alpha=0.33,
            F_value_threshold =-32.5,
        )
        np.testing.assert_array_almost_equal(
            R_calculated.shape,
            (2, 100, 97)
        )
        R_calculated.shape
        R_sum_time_expected=np.array([[
            41.90965929, 30.81230363, 42.27260361,  0.        , 76.03876161,
            66.08282092, 46.36603486, 62.87213183, 57.58080548, 51.27198303,
             0.        ,  0.        , 29.63752437, 38.56509215, 46.19928983,
             0.        ,  0.        , 58.52898827, 47.79559598,  0.        ,
             0.        , 44.40474887, 85.92727433,  0.        ,  0.        ,
            74.92346778,  0.        ,  0.        , 58.5559156 ,  0.        ,
            26.18794222, 29.4832387 , 39.05399955,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            36.4470238 , 52.96669035,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        , 85.77647073,  0.        ,  0.        ,
            64.25433464, 76.43354689, 83.69074307,  0.        ,  0.        ,
             0.        ,  0.        , 53.77075543,  0.        ,  0.        ,
             0.        ,  0.        , 15.82877295,  0.        ,  0.        ,
             6.51377856,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            18.66728697,  0.        ,  0.        , 43.46232702,  0.        ,
             0.        ,  0.        ,  0.        , 69.24545576, 50.77061707,
             0.        ,  0.        , 41.45100524,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [46.18260008, 36.28838243, 46.00223947,  0.        , 87.43020587,
            71.27052955, 44.93293929, 65.74056063, 55.28587772, 53.31173911,
             0.        ,  0.        , 35.35763306, 41.51196592, 46.56380559,
             0.        ,  0.        , 71.04911487, 50.2374038 ,  0.        ,
             0.        , 48.21782928, 83.95550245,  0.        ,  0.        ,
            82.38917753,  0.        ,  0.        , 48.97362551,  0.        ,
            25.75600752, 32.21408015, 36.54471132,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            34.57063941, 50.31299284,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        , 78.58860383,  0.        ,  0.        ,
            61.92381488, 83.28080822, 81.21570599,  0.        ,  0.        ,
             0.        ,  0.        , 52.32298068,  0.        ,  0.        ,
             0.        ,  0.        , 15.32280994,  0.        ,  0.        ,
             6.24706586,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            16.7221373 ,  0.        ,  0.        , 40.58670524,  0.        ,
             0.        ,  0.        ,  0.        , 69.59247934, 52.1466364 ,
             0.        ,  0.        , 43.21924184,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])

        np.testing.assert_array_almost_equal(
            R_calculated.sum(dim='time'),
            R_sum_time_expected,
            decimal=4
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
        pol = xr.DataArray(["Vertical"])


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
                F_value_threshold=-32.5
            )

        with self.assertRaises(ValueError):
            nearby_rain.nearby_rainfall_retrival(
                pref=pref,
                p_c_min=pcmin,
                p_c_max=pcmax,
                F=F,
                length=length,
                f_GHz = f_GHz,
                pol = xr.DataArray(["Vertical","Vertical"]),
                a = None,
                b = None,
                a_b_approximation = "ITU_2005",
                waa_max = 2.3,
                alpha = 0.33,
                F_value_threshold=-32.5
            )
