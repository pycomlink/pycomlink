import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from pycomlink.processing import k_R_relation


class Test_a_b(unittest.TestCase):
    def test_with_float(self):

        f_GHz = 30

        pol = "V"
        calculated_a, calculated_b = k_R_relation.a_b(
            f_GHz, pol, approx_type="ITU_2005"
        )
        assert_almost_equal(0.2291, calculated_a)
        assert_almost_equal(0.9129, calculated_b)

        pol = "H"
        calculated_a, calculated_b = k_R_relation.a_b(
            f_GHz, pol, approx_type="ITU_2005"
        )
        assert_almost_equal(0.2403, calculated_a)
        assert_almost_equal(0.9485, calculated_b)

    def test_with_array(self):

        f_GHz = np.arange(1, 100, 0.5)

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

    def test_interpolation(self):

        f_GHz = np.array([1.7, 28.9, 82.1])

        pol = "V"
        calculated_a, calculated_b = k_R_relation.a_b(
            f_GHz, pol, approx_type="ITU_2005"
        )
        expected_a = np.array([7.32461650e-05, 2.10765233e-01, 1.19270158e00])
        expected_b = np.array([0.91564363, 0.92104328, 0.69864681])
        assert_almost_equal(expected_a, calculated_a)
        assert_almost_equal(expected_b, calculated_b)

        pol = "H"
        calculated_a, calculated_b = k_R_relation.a_b(
            f_GHz, pol, approx_type="ITU_2005"
        )
        expected_a = np.array([5.79858748e-05, 2.20643479e-01, 1.19577832e00])
        expected_b = np.array([1.03804498, 0.95897267, 0.70750869])
        assert_almost_equal(expected_a, calculated_a)
        assert_almost_equal(expected_b, calculated_b)

    def test_2003_ITU_table(self):

        f_GHz = np.array([2, 35, 90])

        pol = "V"
        calculated_a, calculated_b = k_R_relation.a_b(
            f_GHz, pol, approx_type="ITU_2003"
        )
        expected_a = np.array([1.00e-04, 2.33e-01, 9.99e-01])
        expected_b = np.array([0.923, 0.963, 0.754])
        assert_almost_equal(expected_a, calculated_a)
        assert_almost_equal(expected_b, calculated_b)

        pol = "H"
        calculated_a, calculated_b = k_R_relation.a_b(
            f_GHz, pol, approx_type="ITU_2003"
        )
        expected_a = np.array([2.00e-04, 2.63e-01, 1.06e00])
        expected_b = np.array([0.963, 0.979, 0.753])
        assert_almost_equal(expected_a, calculated_a)
        assert_almost_equal(expected_b, calculated_b)

    def test_raises(self):

        with self.assertRaises(ValueError):
            k_R_relation.a_b(-2, "H", approx_type="ITU_2005")

        with self.assertRaises(ValueError):
            k_R_relation.a_b(30, "b", approx_type="ITU_2005")

        with self.assertRaises(ValueError):
            k_R_relation.a_b(30, "H", approx_type="ITU_2000")


class Test_calc_R_from_A(unittest.TestCase):
    def test_with_int(self):
        A = 5
        L_km = 5
        f_GHz = 30
        pol = 'H'
        calculated_R = k_R_relation.calc_R_from_A(A, L_km, f_GHz, pol)
        assert_almost_equal(4.49644185, calculated_R)

    def test_with_arg_a_b(self):
        A = 5
        L_km = 3
        f_GHz = 23
        pol = 'V'
        a, b = k_R_relation.a_b(f_GHz=f_GHz, pol=pol)

        calculated_R_a_b = k_R_relation.calc_R_from_A(
                A=A, L_km=L_km, a=a, b=b,
                )
        calculated_R_f_pol = k_R_relation.calc_R_from_A(
                A=A, L_km=L_km, f_GHz=f_GHz, pol=pol,
                )
        assert_almost_equal(calculated_R_a_b, calculated_R_f_pol)

    def test_with_array(self):

        A = np.linspace(0, 30, 5)
        A[3] = np.nan

        pol = 'H'

        L_km = 0.1
        f_GHz = 80
        expected_R = np.array([0.0, 346.1952323, 917.0924868, np.nan, 2429.43446609])
        calculated_R = k_R_relation.calc_R_from_A(A, L_km, f_GHz, pol)
        assert_almost_equal(expected_R, calculated_R)

        L_km = 5
        f_GHz = 30
        expected_R = np.array([0.0, 6.89479467, 14.31845421, np.nan, 29.7352047])
        calculated_R = k_R_relation.calc_R_from_A(A, L_km, f_GHz, pol)
        assert_almost_equal(expected_R, calculated_R)

        L_km = 30
        f_GHz = 12
        expected_R = np.array([0.0, 7.29133826, 13.10322105, np.nan, 23.54772138])
        calculated_R = k_R_relation.calc_R_from_A(A, L_km, f_GHz, pol)
        assert_almost_equal(expected_R, calculated_R)

        L_km = 50
        f_GHz = 5
        expected_R = np.array([0.0, 47.24635411, 71.08341151, np.nan, 106.946906])
        calculated_R = k_R_relation.calc_R_from_A(A, L_km, f_GHz, pol)
        assert_almost_equal(expected_R, calculated_R)

        # without NaN
        expected_R = np.array([0.0, 47.24635411, 71.08341151])
        calculated_R = k_R_relation.calc_R_from_A(A[:3], L_km, f_GHz, pol)
        assert_almost_equal(expected_R, calculated_R)
