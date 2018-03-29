import unittest
import numpy as np
import pandas as pd
import pycomlink as pycml
from pycomlink.tests.utils import load_processed_cml_list

t_str_list = [
    '2017-06-27 21:50:00',
    '2017-06-27 22:50:00',
    '2017-06-27 23:50:00',
    '2017-06-28 00:50:00',
    '2017-06-28 01:50:00',
    '2017-06-28 02:50:00',
    '2017-06-28 03:50:00',
    '2017-06-28 04:50:00',
    '2017-06-28 05:50:00',
    '2017-06-28 06:50:00',
    '2017-06-28 07:50:00',
    '2017-06-28 08:50:00',
    '2017-06-28 09:50:00']


class TestAggregateDfToNewIndex(unittest.TestCase):
    def test_short_new_hourly_index_left_edge(self):
        cml_list = load_processed_cml_list()
        cml = cml_list[0]

        # Test with label 'right'
        df_resample = pycml.util.temporal.aggregate_df_onto_DatetimeIndex(
            df=cml.channel_1.data,
            new_index=pd.to_datetime(t_str_list),
            label='right',
            method=np.mean)

        pd.util.testing.assert_frame_equal(
            pd.DataFrame(df_resample.txrx),
            pd.DataFrame(
                index=pd.DatetimeIndex([
                    '2017-06-28 00:50:00+00:00', '2017-06-28 01:50:00+00:00',
                    '2017-06-28 02:50:00+00:00', '2017-06-28 03:50:00+00:00',
                    '2017-06-28 04:50:00+00:00', '2017-06-28 05:50:00+00:00',
                    '2017-06-28 06:50:00+00:00', '2017-06-28 07:50:00+00:00',
                    '2017-06-28 08:50:00+00:00', '2017-06-28 09:50:00+00:00'],
                    dtype='datetime64[ns, UTC]', name=u'time', freq=None),
                data={'txrx': [60.988,  60.755, 60.88813559, 60.87368421,
                               60.97241379, 70.48305085, 63.42807018, 61.2,
                               60.62166667, 60.92]})
            )

        # Test with label 'left'
        df_resample = pycml.util.temporal.aggregate_df_onto_DatetimeIndex(
            df=cml.channel_1.data,
            new_index=pd.to_datetime(t_str_list),
            label='left',
            method=np.mean)

        pd.util.testing.assert_frame_equal(
            pd.DataFrame(df_resample.txrx),
            pd.DataFrame(
                index=pd.DatetimeIndex([
                    '2017-06-28 00:50:00+00:00', '2017-06-28 01:50:00+00:00',
                    '2017-06-28 02:50:00+00:00', '2017-06-28 03:50:00+00:00',
                    '2017-06-28 04:50:00+00:00', '2017-06-28 05:50:00+00:00',
                    '2017-06-28 06:50:00+00:00', '2017-06-28 07:50:00+00:00',
                    '2017-06-28 08:50:00+00:00'],
                    dtype='datetime64[ns, UTC]', name=u'time', freq=None),
                data={'txrx': [60.755, 60.88813559, 60.87368421,
                               60.97241379, 70.48305085, 63.42807018, 61.2,
                               60.62166667, 60.92]})
            )