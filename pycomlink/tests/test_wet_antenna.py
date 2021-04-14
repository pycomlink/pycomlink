import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from pycomlink.processing import wet_antenna


class TestWaaSchleiss2013(unittest.TestCase):
    def test_with_synthetic_data(self):
        trsl = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            + [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            + [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            + [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            + [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
            + [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        wet = trsl > 1
        baseline = np.zeros_like(trsl)

        # Test one set of parameters
        waa = wet_antenna.waa_schleiss_2013(
            rsl=trsl, baseline=baseline, wet=wet, waa_max=1.5, delta_t=1, tau=15
        )
        assert_almost_equal(
            waa[[5, 15, 30, 40, 50, 65]],
            np.array([0.0, 1.106784, 1.48616494, 0.8, 0.8, 0.0]),
        )

        # Test another set of parameters
        waa = wet_antenna.waa_schleiss_2013(
            rsl=trsl, baseline=baseline, wet=wet, waa_max=1.8, delta_t=5, tau=30
        )
        assert_almost_equal(
            waa[[5, 15, 30, 40, 50, 65]],
            np.array([0.0, 1.771875, 1.79999914, 0.8, 0.8, 0.0]),
        )


class TestWaaLeijnse2008(unittest.TestCase):
    def test_with_R_array(self):
        R = np.logspace(-3, 2, 6)

        expected_dict = {
            "10": np.array(
                [0.14634379, 0.25478827, 0.44387539, 0.77317254, 1.34234051, 2.30324428]
            ),
            "20": np.array(
                [0.25401763, 0.4375012, 0.74861952, 1.26680469, 2.10422909, 3.39294479]
            ),
            "27": np.array(
                [0.25118968, 0.4319954, 0.7374494, 1.24336903, 2.05456625, 3.29028424]
            ),
            "50": np.array(
                [0.19204487, 0.33241259, 0.57352774, 0.9835691, 1.66754941, 2.76780894]
            ),
            "70": np.array(
                [0.29750292, 0.51574491, 0.89129894, 1.52905179, 2.58071428, 4.21794048]
            ),
        }

        for f in expected_dict.keys():
            f_Hz = float(f) * 1e9
            expected = expected_dict[f]

            result = wet_antenna.waa_leijnse_2008(R=R, f_Hz=f_Hz)

            assert_almost_equal(expected, result)


class TestWaaLeijnse2008FromAobs(unittest.TestCase):
    def test_with_R_array(self):
        expected = np.array(
            [
                0.0,
                0.85081045,
                1.2087577,
                1.39943582,
                1.53099038,
                1.63112499,
                1.7142827,
                1.78122755,
                1.84301522,
                1.89539232,
            ]
        )
        result = wet_antenna.waa_leijnse_2008_from_A_obs(
            A_obs=np.arange(10), L_km=10, f_Hz=23e9
        )
        assert_almost_equal(expected, result)


class TestEpswater(unittest.TestCase):
    def test_with_f_array(self):
        f = np.arange(1, 100, 10) * 1e9

        expected = np.array(
            [
                77.48287775 + 3.57991987j,
                61.08942285 + 30.4521563j,
                40.02809328 + 36.24134086j,
                26.80177866 + 33.20334358j,
                19.35764296 + 28.80842726j,
                15.02863615 + 24.91108738j,
                12.36086346 + 21.74801145j,
                10.62309316 + 19.21692908j,
                9.43591151 + 17.17832787j,
                8.59182335 + 15.51584998j,
            ]
        )

        result = wet_antenna.eps_water(f_Hz=f, T_K=300)

        assert_almost_equal(expected, result)
