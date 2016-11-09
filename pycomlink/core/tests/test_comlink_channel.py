import unittest

import numpy as np
import pandas as pd

from pycomlink.core import ComlinkChannel


t_date_range = pd.date_range(start='2015-01-01', periods=20, freq='min')
t_list = [str(date) for date in t_date_range]
rx_list = list(np.sin(np.linspace(0, 10, len(t_list))))
tx_list = list(np.cos(np.linspace(0, 10, len(t_list))))

rx2_list = list(2*np.sin(np.linspace(0, 10, len(t_list))))
tx2_list = list(2*np.cos(np.linspace(0, 10, len(t_list))))

f = 18.9

class TestComlinkChannelInit(unittest.TestCase):

    def test_with_DataFrame_only_rx_no_tx(self):
        df = pd.DataFrame(index=t_date_range, data={'rx': rx_list})
        cml_ch = ComlinkChannel(data=df, f_GHz=f)
        # Test content of rx column
        np.testing.assert_almost_equal(cml_ch.rx.values, rx_list)
        # Test index
        pd.util.testing.assert_almost_equal(cml_ch.index, pd.DatetimeIndex(t_date_range))

    def test_with_list_only_rx_data(self):
        """ Test if the column name is set correctly """
        cml_ch = ComlinkChannel(rx=rx_list, t=t_list, f_GHz=f)
        np.testing.assert_almost_equal(cml_ch.rx.values, rx_list)

    def test_kwargs(self):
        cml_ch = ComlinkChannel(rx=rx_list, t=t_list, f_GHz=f)
        assert(cml_ch.f_GHz == f)


class TestComlinkChannelAttributes(unittest.TestCase):

    def test_len(self):
        df = pd.DataFrame(index=t_date_range, data={'rx': rx_list})
        cml_ch = ComlinkChannel(data=df)
        assert (len(cml_ch) == len(df))


class TestComlinkChannelCopy(unittest.TestCase):

    def test_copy(self):
        df = pd.DataFrame(index=t_date_range, data={'rx': rx_list})
        cml_ch = ComlinkChannel(data=df, f_GHz=18.9)

        cml_ch_copy = cml_ch.copy()

        assert(type(cml_ch_copy) == ComlinkChannel)

        # Test (at least one) metadata attribute
        assert(cml_ch_copy.f_GHz == cml_ch.f_GHz)

        # Test that DataFrames are equal
        pd.util.testing.assert_frame_equal(cml_ch_copy._df, cml_ch._df)

        # Test that the new DataFrame is a copy and not a view
        cml_ch._df.rx[1] = -9999
        assert(cml_ch.rx[1] != cml_ch_copy.rx[1])

        # Test that the new metadata is not a reference but a copy
        cml_ch.f_GHz = 999
        assert(cml_ch.f_GHz == 999)
        assert(cml_ch_copy.f_GHz == f)

    def test_deepcopy(self):
        from copy import deepcopy
        df = pd.DataFrame(index=t_date_range, data={'rx': rx_list})
        cml_ch = ComlinkChannel(data=df, f_GHz=18.9)

        cml_ch_copy = deepcopy(cml_ch)

        assert(type(cml_ch_copy) == ComlinkChannel)

        # Test (at least one) metadata attribute
        assert(cml_ch_copy.f_GHz == cml_ch.f_GHz)

        # Test that DataFrames are equal
        pd.util.testing.assert_frame_equal(cml_ch_copy._df, cml_ch._df)

        # Test that the new DataFrame is a copy and not a view
        cml_ch._df.rx[1] = -9999
        assert(cml_ch.rx[1] != cml_ch_copy.rx[1])

        # Test that the new metadata is not a reference but a copy
        cml_ch.f_GHz = 999
        assert(cml_ch.f_GHz == 999)
        assert(cml_ch_copy.f_GHz == f)


class TestComlinkChannelTypeAfterManipulation(unittest.TestCase):

    def test_index_slicing(self):
        df = pd.DataFrame(index=t_date_range, data={'rx': rx_list})
        cml_ch = ComlinkChannel(data=df)
        cml_ch_sliced = cml_ch[1:4]
        assert(type(cml_ch_sliced) == ComlinkChannel)
        pd.util.testing.assert_frame_equal(cml_ch_sliced._df, df[1:4])
        for key in cml_ch._metadata:
            assert(cml_ch_sliced[key] == cml_ch[key])

    def test_resampling(self):
        df = pd.DataFrame(index=t_date_range, data={'rx': rx_list})
        cml_ch_1min = ComlinkChannel(data=df, f_GHz=f)
        cml_ch_5min = ComlinkChannel(data=df.resample('5min').apply(np.mean), f_GHz=f)

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


def assert_comlink_channel_equal(cml_ch_1, cml_ch_2):

    assert(cml_ch_1.f_GHz == f)

    assert(cml_ch_2.f_GHz == f)

    for key in cml_ch_1.__dict__:
        if key == '_df':
            pd.util.testing.assert_frame_equal(cml_ch_1._df,
                                               cml_ch_2._df)
        else:
            assert(cml_ch_1.__dict__[key] ==
                   cml_ch_2.__dict__[key])

