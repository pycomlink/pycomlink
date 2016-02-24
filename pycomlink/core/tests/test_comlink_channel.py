import unittest
import sys

import numpy as np
import pandas as pd

from pycomlink.core.comlink_channel import ComlinkChannel


t_date_range = pd.date_range(start='2015-01-01', periods=20, freq='min')
t_list = [str(date) for date in t_date_range]
rx_list = list(np.sin(np.linspace(0, 10, len(t_list))))
tx_list = list(np.cos(np.linspace(0, 10, len(t_list))))

rx2_list = list(2*np.sin(np.linspace(0, 10, len(t_list))))
tx2_list = list(2*np.cos(np.linspace(0, 10, len(t_list))))

class TestComlinkChannelInit(unittest.TestCase):

    def test_with_DataFrame_only_rx_and_tx(self):
        df = pd.DataFrame(index=t_date_range, data={'rx': rx_list})
        cml_ch = ComlinkChannel(data=df)
        # Test content of rx column
        np.testing.assert_almost_equal(cml_ch.rx.values, rx_list)
        # Test index
        pd._testing.assert_almost_equal(cml_ch.index, pd.DatetimeIndex(t_date_range))

    def test_with_list_only_rx_data(self):
        """ Test if the column name is set correctly """
        cml_ch = ComlinkChannel(rx=rx_list, t=t_list)
        pd._testing.assert_almost_equal(cml_ch.rx, rx_list)


class TestComlinkChannelTypeAfterManipulation(unittest.TestCase):

    def test_resampling(self):
        df = pd.DataFrame(index=t_date_range, data={'rx': rx_list})
        cml_ch_1min = ComlinkChannel(data=df)
        cml_ch_5min = cml_ch_1min.resample('5min')
        assert(type(cml_ch_1min) == type(cml_ch_5min))
        for key in cml_ch_1min._metadata:
            assert(cml_ch_1min[key] == cml_ch_5min[key])
