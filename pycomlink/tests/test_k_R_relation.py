import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from pycomlink.processing import k_R_relation



class TestCalc_R_from_A(unittest.TestCase):
    
    def test_with_float(self):
        
        f_GHz = 30
        
        pol = "V"
        calculated_a, calculated_b = k_R_relation.a_b(f_GHz, pol, approx_type="ITU_2005")
        assert_almost_equal(0.2291, calculated_a)
        assert_almost_equal(0.9129, calculated_b)

        pol = "H"
        calculated_a, calculated_b = k_R_relation.a_b(f_GHz, pol, approx_type="ITU_2005")
        assert_almost_equal(0.2403, calculated_a)
        assert_almost_equal(0.9485, calculated_b)
    
    def test_with_array(self):
        
        f_GHz = np.arange(1,100,0.5)

        pol = "V"
        a, b = k_R_relation.a_b(f_GHz, pol, approx_type="ITU_2005")
        calculated_a = a[2::40]
        calculated_b = b[2::40]
        expected_a = np.array([0.0000998, 0.117, 0.4712, 0.8889, 1.1915])
        expected_b = np.array([0.949, 0.97, 0.8296, 0.7424, 0.6988])
        assert_almost_equal(
            expected_a,
            calculated_a,
        )
        assert_almost_equal(
            expected_b,
            calculated_b,
        )

        pol = "H"
        a, b = k_R_relation.a_b(f_GHz, pol, approx_type="ITU_2005")
        calculated_a = a[2::40]
        calculated_b = b[2::40]
        expected_a = np.array([0.0000847, 0.1155, 0.4865, 0.8974, 1.1946])
        expected_b = np.array([1.0664, 1.0329, 0.8539, 0.7586, 0.7077])
        assert_almost_equal(
            expected_a,
            calculated_a,
        )
        assert_almost_equal(
            expected_b,
            calculated_b,
        )


        