import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from pycomlink.core import ComlinkChannel, Comlink


t_date_range = pd.date_range(start='2015-01-01', periods=20, freq='min')
t_list = [str(date) for date in t_date_range]
rx_list = list(np.sin(np.linspace(0, 10, len(t_list))))
tx_list = list(np.cos(np.linspace(0, 10, len(t_list))))

f = 18.9

df = pd.DataFrame(index=t_date_range, data={'rx': rx_list})
cml_ch = ComlinkChannel(data=df, f_GHz=f)
cml_ch2 = ComlinkChannel(data=df*2, f_GHz=f*2)


class TestComlinkInit(unittest.TestCase):

    def testWithOneComlinkChannel(self):
        cml = generate_standard_cml()
        np.testing.assert_almost_equal(cml.channel_1.rx, rx_list)
        assert(cml.channel_1.f_GHz == f)


class TestComlinkCopy(unittest.TestCase):

    def test_copy(self):

        cml = generate_standard_cml()

        cml_copy = deepcopy(cml)

        assert(type(cml_copy) == Comlink)

        # Test (at least one) metadata attribute
        assert(cml_copy.metadata['site_a_latitude'] ==
               cml.metadata['site_a_latitude'])

        # Test that the new metadata is not a reference but a copy
        cml.metadata['site_a_latitude'] = 999
        assert(cml.metadata['site_a_latitude'] == 999)
        assert(cml_copy.metadata['site_a_latitude'] == 44.1)

        # Test that DataFrames of channels are equal
        for ch_name in cml.channels.keys():
            ch = cml.channels[ch_name]
            ch_copy = cml_copy.channels[ch_name]
            pd.util.testing.assert_frame_equal(ch._df, ch_copy._df)

        # Test that the new DataFrames of the channels are copies and not a views
        for ch_name in cml.channels.keys():
            ch = cml.channels[ch_name]
            ch_copy = cml_copy.channels[ch_name]
            ch._df.rx[1] = -9999
            assert(ch.rx[1] != ch_copy.rx[1])


def generate_standard_cml():
    cml = Comlink(channels=deepcopy(cml_ch),
                  f_GHz=f,
                  pol='V',
                  metadata={'site_a_latitude': 44.1,
                            'site_a_longitude': 11.1,
                            'site_b_latitude': 44.2,
                            'site_b_longitude': 11.2,
                            'cml_id': 'foo_bar_123'})
    return cml


def assert_comlink_equal(cml_1, cml_2):
    pass

