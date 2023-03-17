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
                0.82241041,
                1.16192413,
                1.34918332,
                1.47911528,
                1.57976159,
                1.66444932,
                1.73214808,
                1.79472864,
                1.84852106,
            ]
        )
        result = wet_antenna.waa_leijnse_2008_from_A_obs(
            A_obs=np.arange(10),
            L_km=10,
            f_Hz=23e9,
            pol="H",
        )
        assert_almost_equal(expected, result)

    def test_error_with_negative_R(self):
        with self.assertRaises(ValueError) as cm:
            wet_antenna.waa_leijnse_2008_from_A_obs(
                A_obs=np.array([0, 2, -1]),
                L_km=10,
                f_Hz=23e9,
                pol="H",
            )


class TestWaaPastorek2021(unittest.TestCase):
    def test_with_R_array(self):
        R = np.logspace(-3, 2, 6)

        # some values for A_max
        expected_dict = {
            "5": np.array(
                [0.01118109, 0.03955909, 0.13895185, 0.47581291, 1.49347849, 3.58020499]
            ),
            "14": np.array(
                [
                    0.03130704,
                    0.11076545,
                    0.38906518,
                    1.33227615,
                    4.18173977,
                    10.02457398,
                ]
            ),
            "20": np.array(
                [
                    0.04472434,
                    0.15823636,
                    0.55580741,
                    1.90325164,
                    5.97391396,
                    14.32081997,
                ]
            ),
        }

        for k in expected_dict.keys():
            A_max = float(k)
            expected = expected_dict[k]
            result = wet_antenna.waa_pastorek_2021(R=R, A_max=A_max, zeta=0.55, d=0.1)
            assert_almost_equal(expected, result)

        # some values for zeta
        expected_dict = {
            "0.3": np.array(
                [0.17514477, 0.34728415, 0.68436903, 1.33227615, 2.53233773, 4.59773931]
            ),
            "0.55": np.array(
                [
                    0.03130704,
                    0.11076545,
                    0.38906518,
                    1.33227615,
                    4.18173977,
                    10.02457398,
                ]
            ),
            "0.65": np.array(
                [
                    0.01569945,
                    0.06999067,
                    0.30993868,
                    1.33227615,
                    5.04355184,
                    12.09630827,
                ]
            ),
        }

        for k in expected_dict.keys():
            zeta = float(k)
            expected = expected_dict[k]
            result = wet_antenna.waa_pastorek_2021(R=R, A_max=14, zeta=zeta, d=0.1)
            assert_almost_equal(expected, result)


class TestWaaPastorek2021FromAobs(unittest.TestCase):
    def test_with_R_array(self):
        expected = np.array(
            [
                0.0,
                0.66274715,
                1.10407043,
                1.45760649,
                1.76189926,
                2.0291218,
                2.26924516,
                2.49339808,
                2.69506481,
                2.88842441,
            ]
        )
        result = wet_antenna.waa_pastorek_2021_from_A_obs(
            A_obs=np.arange(10), f_Hz=23e9, pol="H", L_km=10, A_max=14, zeta=0.55, d=0.1
        )
        assert_almost_equal(expected, result)

    def test_error_with_negative_R(self):
        with self.assertRaises(ValueError) as cm:
            wet_antenna.waa_pastorek_2021_from_A_obs(
                A_obs=np.array([0, 2, -1]),
                f_Hz=23e9,
                pol="H",
                L_km=10,
                A_max=14,
                zeta=0.55,
                d=0.1,
            )


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
