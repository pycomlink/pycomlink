import unittest

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
        cml = Comlink(channels=cml_ch,
                      f_GHz=f,
                      pol='V',
                      site_dict={'site_a_lat': 44.1,
                                 'site_a_lon': 11.1,
                                 'site_b_lat': 44.2,
                                 'site_b_lon': 11.2})
        np.testing.assert_almost_equal(cml.channel_1.rx, rx_list)
        assert(cml.channel_1.f_GHz == f)

def assert_comlink_equal(cml_1, cml_2):

    assert(cml_ch_1.f_GHz == f)

    assert(cml_ch_2.f_GHz == f)

    for key in cml_ch_1.__dict__:
        if key == '_df':
            pd.util.testing.assert_frame_equal(cml_ch_1._df,
                                               cml_ch_2._df)
        else:
            assert(cml_ch_1.__dict__[key] ==
                   cml_ch_2.__dict__[key])

