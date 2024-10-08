import unittest
import numpy as np  
import pycomlink.processing.blackout_gap_detection as gap_detec


class TestBlackoutDetection(unittest.TestCase):
    def testBlackoutdetectionwithsimpletimeseries(self):
        rsl_list = [np.array(
            [-63, -66, -67, -69, -64, np.nan, -64, -66, -70, np.nan, -70, -68]),
            np.array([np.nan, -66, -67, -69, -64, np.nan, -64, -66, -70,
                      np.nan, -70, -68]),
            np.array([-63, -66, -67, -69, -64, np.nan, -64, -66, -70,
                      np.nan,-70, -np.nan])]
        ref = np.array([False, False, False, False, False, False,
                        False, False, False, True, False, False])

        for rsl in rsl_list:
            gap_start, gap_end = gap_detec.get_blackout_start_and_end(
                rsl=rsl, rsl_threshold=-65
            )

            mask = gap_detec.created_blackout_gap_mask_from_start_end_markers(
                rsl, gap_start, gap_end
            )
            mask_reverse = gap_detec.created_blackout_gap_mask_from_start_end_markers(
                rsl, gap_end[::-1], gap_start[::-1]
            )
            mask = mask | mask_reverse[::-1]
            np.testing.assert_array_almost_equal(mask, ref)




