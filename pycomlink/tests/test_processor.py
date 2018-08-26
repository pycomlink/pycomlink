import unittest
import numpy as np
from pycomlink.tests.utils import load_and_clean_example_cml
import pycomlink as pycml


class TestWetDryStdDev(unittest.TestCase):
    def test_standrad_processing(self):
        cml = load_and_clean_example_cml()
        cml.process.wet_dry.std_dev(window_length=60, threshold=0.8)

        assert (cml.channel_1.data.wet.loc['2016-11-02 14:00'].values[0]
                == True)
        assert (cml.channel_1.data.wet.loc['2016-11-02 13:00'].values[0]
                == False)

        assert (cml.channel_2.data.wet.loc['2016-11-02 14:00'].values[0]
                == True)
        assert (cml.channel_2.data.wet.loc['2016-11-02 13:00'].values[0]
                == False)

        cml.process.wet_dry.std_dev(window_length=30, threshold=0.8)

        assert (cml.channel_1.data.wet.loc['2016-11-02 14:00'].values[0]
                == False)
        assert (cml.channel_1.data.wet.loc['2016-11-02 13:00'].values[0]
                == False)
        assert (cml.channel_1.data.wet.loc['2016-11-02 14:30'].values[0]
                == True)

        assert (cml.channel_2.data.wet.loc['2016-11-02 14:00'].values[0]
                == False)
        assert (cml.channel_2.data.wet.loc['2016-11-02 13:00'].values[0]
                == False)
        assert (cml.channel_2.data.wet.loc['2016-11-02 14:30'].values[0]
                == True)

        # Test if the end result, the rain rate, is the same when using the
        # Processor and the functions

        # This test only works correctly if the CML uses vertical polarization
        # since the default in the function is 'H'
        assert cml.channel_1.metadata['polarization'] == 'V'

        cml.process.wet_dry.std_dev(window_length=30, threshold=0.8)
        cml.process.baseline.linear()
        cml.process.baseline.calc_A()
        cml.process.A_R.calc_R()

        R_from_processor = cml.channel_1.data.R.copy()

        R_from_function = (
            pycml.processing.A_R_relation.A_R_relation
            .calc_R_from_A(
                A=cml.channel_1.data.A,
                L=cml.metadata['length'],
                f_GHz=cml.channel_1.metadata['frequency'] / 1e9,
                pol=cml.channel_1.metadata['polarization']))

        np.testing.assert_almost_equal(R_from_processor, R_from_function)

    def test_processing_for_selected_time_period(self):
        cml = load_and_clean_example_cml()

        # First as a comparisson the standard way
        cml.process.wet_dry.std_dev(window_length=60, threshold=0.8)

        assert (cml.channel_1.data.wet.loc['2016-11-02 18:30'].values[0]
                == True)
        assert (cml.channel_1.data.wet.loc['2016-11-02 14:00'].values[0]
                == True)
        assert (cml.channel_1.data.wet.loc['2016-11-02 13:00'].values[0]
                == False)

        # Then for a period from a starting point in time
        t_start = '2016-11-02 14:30'
        cml.channel_1.data.wet = False
        cml.process.wet_dry.std_dev(window_length=60,
                                    threshold=0.8,
                                    t_start=t_start)
        assert (cml.channel_1.data.wet.loc['2016-11-02 18:30'].values[0]
                == True)
        assert (cml.channel_1.data.wet.loc['2016-11-02 14:00'].values[0]
                == False)
        assert (cml.channel_1.data.wet.loc['2016-11-02 13:00'].values[0]
                == False)

        # Then for a period to a end point in time
        t_stop = '2016-11-02 15:00'
        cml.channel_1.data.wet = False
        cml.process.wet_dry.std_dev(window_length=60,
                                    threshold=0.8,
                                    t_stop=t_stop)
        assert (cml.channel_1.data.wet.loc['2016-11-02 18:30'].values[0]
                == False)
        assert (cml.channel_1.data.wet.loc['2016-11-02 14:00'].values[0]
                == True)
        assert (cml.channel_1.data.wet.loc['2016-11-02 13:00'].values[0]
                == False)
