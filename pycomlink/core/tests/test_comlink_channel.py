from __future__ import division
from builtins import str
import unittest

import numpy as np
import pandas as pd

from pycomlink.core import ComlinkChannel


t_date_range = pd.date_range(start='2015-01-01', periods=500, freq='min')
t_list = [str(date) for date in t_date_range]
rx_list = list(np.sin(np.linspace(0, 10, len(t_list))))
tx_list = list(np.cos(np.linspace(0, 10, len(t_list))))

rx2_list = list(2*np.sin(np.linspace(0, 10, len(t_list))))
tx2_list = list(2*np.cos(np.linspace(0, 10, len(t_list))))

f = 18.9 * 1e9


class TestComlinkChannelInit(unittest.TestCase):

    def test_with_DataFrame_rx_and_tx(self):
        df = pd.DataFrame(index=t_date_range,
                          data={'rx': rx_list,
                                'tx': tx_list})
        cml_ch = ComlinkChannel(data=df, frequency=f)
        # Test content of columns
        np.testing.assert_almost_equal(cml_ch.rx.values, rx_list)
        np.testing.assert_almost_equal(cml_ch.tx.values, tx_list)

        # Test index
        pd.util.testing.assert_almost_equal(cml_ch.index,
                                            pd.DatetimeIndex(
                                                t_date_range,
                                                name='time'))

    def test_with_list_rx_tx_data(self):
        """ Test if the column name is set correctly """
        cml_ch = ComlinkChannel(rx=rx_list, tx=tx_list, t=t_list, frequency=f)
        np.testing.assert_almost_equal(cml_ch.rx.values, rx_list)
        np.testing.assert_almost_equal(cml_ch.tx.values, tx_list)

    def test_kwargs(self):
        cml_ch = ComlinkChannel(rx=rx_list, tx=tx_list, t=t_list, frequency=f)
        assert(cml_ch.f_GHz == f /1e9)


class TestComlinkChannelAttributes(unittest.TestCase):

    def test_len(self):
        df = pd.DataFrame(index=t_date_range,
                          data={'rx': rx_list,
                                'tx': tx_list})
        cml_ch = ComlinkChannel(data=df, frequency=f)
        assert (len(cml_ch) == len(df))


class TestComlinkChannelCopy(unittest.TestCase):

    def test_copy(self):
        df = pd.DataFrame(index=t_date_range,
                          data={'rx': rx_list,
                                'tx': tx_list})
        cml_ch = ComlinkChannel(data=df, frequency=f)
        cml_ch_copy = cml_ch.copy()

        assert(type(cml_ch_copy) == ComlinkChannel)

        # Test (at least one) metadata attribute
        assert(cml_ch_copy.f_GHz == cml_ch.f_GHz)

        # Test that DataFrames are equal
        pd.util.testing.assert_frame_equal(cml_ch_copy.data, cml_ch.data)

        # Test that the new DataFrame is a copy and not a view
        cml_ch.data.rx[1] = -9999
        assert(cml_ch.rx[1] != cml_ch_copy.rx[1])

        # Test that the new metadata is not a reference but a copy
        cml_ch.f_GHz = 999
        assert(cml_ch.f_GHz == 999)
        assert(cml_ch_copy.f_GHz == f /1e9)

    def test_deepcopy(self):
        from copy import deepcopy
        df = pd.DataFrame(index=t_date_range,
                          data={'rx': rx_list,
                                'tx': tx_list})
        cml_ch = ComlinkChannel(data=df, frequency=f)

        cml_ch_copy = deepcopy(cml_ch)

        assert(type(cml_ch_copy) == ComlinkChannel)

        # Test (at least one) metadata attribute
        assert(cml_ch_copy.f_GHz == cml_ch.f_GHz)

        # Test that DataFrames are equal
        pd.util.testing.assert_frame_equal(cml_ch_copy.data, cml_ch.data)

        # Test that the new DataFrame is a copy and not a view
        cml_ch.data.rx[1] = -9999
        assert(cml_ch.rx[1] != cml_ch_copy.rx[1])

        # Test that the new metadata is not a reference but a copy
        cml_ch.f_GHz = 999
        assert(cml_ch.f_GHz == 999)
        assert(cml_ch_copy.f_GHz == f /1e9)


class TestComlinkChannelTypeAfterManipulation(unittest.TestCase):

    def test_index_slicing(self):
        df = pd.DataFrame(index=t_date_range,
                          data={'rx': rx_list,
                                'tx': tx_list})
        cml_ch = ComlinkChannel(data=df, frequency=f)
        cml_ch_sliced = cml_ch[1:4]
        assert(type(cml_ch_sliced) == ComlinkChannel)
        pd.util.testing.assert_frame_equal(cml_ch_sliced.data, df[1:4])
        for key in cml_ch._metadata:
            assert(cml_ch_sliced[key] == cml_ch[key])

    def test_resampling(self):
        df = pd.DataFrame(index=t_date_range,
                          data={'rx': rx_list,
                                'tx': tx_list})
        cml_ch_1min = ComlinkChannel(data=df, frequency=f)
        cml_ch_5min = ComlinkChannel(data=df.resample('5min').apply(np.mean),
                                     frequency=f)

        cml_ch_5min_no_inplace_kwarg = cml_ch_1min.resample('5min')
        assert(type(cml_ch_5min_no_inplace_kwarg) == ComlinkChannel)
        assert_comlink_channel_equal(cml_ch_5min_no_inplace_kwarg,
                                     cml_ch_5min)

        cml_ch_5min_inplace_false = cml_ch_1min.resample('5min', inplace=False)
        assert(type(cml_ch_5min_inplace_false) == ComlinkChannel)
        assert_comlink_channel_equal(cml_ch_5min_inplace_false,
                                     cml_ch_5min)

        cml_ch_5min_inplace_true = cml_ch_1min.copy()
        cml_ch_5min_inplace_true.resample('5min', inplace=True)
        assert(type(cml_ch_5min_inplace_true) == ComlinkChannel)
        assert_comlink_channel_equal(cml_ch_5min_inplace_true,
                                     cml_ch_5min)


class TestComlinkChannelAppendData(unittest.TestCase):
    def test_append_no_kwargs(self):
        df = pd.DataFrame(index=t_date_range,
                          data={'rx': rx_list,
                                'tx': tx_list})
        cml_ch_full = ComlinkChannel(data=df, frequency=f)
        cml_ch_shortened = ComlinkChannel(data=df.iloc[:100, :], frequency=f)
        cml_ch_rest_of_data = ComlinkChannel(data=df.iloc[100:, :], frequency=f)

        assert(len(cml_ch_full.data) == len(t_date_range))
        assert(len(cml_ch_shortened.data) == len(t_date_range[:100]))
        assert(len(cml_ch_rest_of_data.data) == len(t_date_range[100:]))

        cml_ch_shortened.append_data(cml_ch_rest_of_data)

        assert_comlink_channel_equal(cml_ch_shortened, cml_ch_full)

    def test_append_max_length(self):
        df = pd.DataFrame(index=t_date_range,
                          data={'rx': rx_list,
                                'tx': tx_list})
        cml_ch_shortened = ComlinkChannel(data=df.iloc[:100, :], frequency=f)
        cml_ch_rest_of_data = ComlinkChannel(data=df.iloc[100:, :], frequency=f)

        cml_ch_shortened.append_data(cml_ch_rest_of_data, max_length=10)

        assert(len(cml_ch_shortened.data) == 10)
        assert(cml_ch_shortened.data.index[-1] == df.index[-1])
        assert(cml_ch_shortened.data.index[0] == df.index[-10])
        assert(cml_ch_shortened.data.tx[-1] == df.tx[-1])
        assert(cml_ch_shortened.data.tx[0] == df.tx[-10])

    def test_append_max_age(self):
        df = pd.DataFrame(index=t_date_range,
                          data={'rx': rx_list,
                                'tx': tx_list})
        cml_ch_shortened = ComlinkChannel(data=df.iloc[:100, :], frequency=f)
        cml_ch_rest_of_data = ComlinkChannel(data=df.iloc[100:, :], frequency=f)

        cml_ch_shortened.append_data(cml_ch_rest_of_data, max_age='60min')

        assert(len(cml_ch_shortened.data) ==
               len(df.iloc[df.index >
                           (df.index[-1] - pd.Timedelta('60min')), :]))
        assert(cml_ch_shortened.data.index[-1] == df.index[-1])

    def test_append_wrong_cml_ch(self):
        df = pd.DataFrame(index=t_date_range,
                          data={'rx': rx_list,
                                'tx': tx_list})
        cml_ch_shortened = ComlinkChannel(data=df.iloc[:100, :],
                                          frequency=f)
        cml_ch_rest_of_data = ComlinkChannel(data=df.iloc[100:, :],
                                             frequency=22.235 * 1e9)

        self.assertRaises(ValueError,
                          cml_ch_shortened.append_data,
                          cml_ch_rest_of_data)


def assert_comlink_channel_equal(cml_ch_1, cml_ch_2):

    assert(cml_ch_1.f_GHz == f / 1e9)

    assert(cml_ch_2.f_GHz == f / 1e9)

    for key in cml_ch_1.__dict__:
        if key == 'data':
            pd.util.testing.assert_frame_equal(cml_ch_1.data,
                                               cml_ch_2.data)
        else:
            assert(cml_ch_1.__dict__[key] ==
                   cml_ch_2.__dict__[key])

