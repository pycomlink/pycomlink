import pycomlink as pycml
import numpy as np


def init_data():
    # TODO Do not use example data here
    cml = pycml.io.examples.get_one_cml()
    cml["tsl"] = cml.tsl.where(cml.tsl != 255.0)
    cml["rsl"] = cml.rsl.where(cml.rsl != -99.9)
    cml["trsl"] = cml.tsl - cml.rsl
    return cml


def test_baseline_constant_kwarg():
    cml = init_data()
    # first do simple wet-dry classification
    # TODO add other methods, maybe sftf when it works with new xarray interface
    cml["wet"] = cml.trsl.rolling(time=60, center=True).std()

    # call baseline function using kwargs
    baseline_da = pycml.processing.baseline.baseline_constant(
        trsl=cml.trsl,
        wet=cml.wet,
    )

    for channel_id in range(len(cml.channel_id)):
        baseline_np = pycml.processing.baseline.baseline_constant(
            trsl=cml.trsl.isel(channel_id=channel_id).values,
            wet=cml.wet.isel(channel_id=channel_id).values,
        )
    np.testing.assert_almost_equal(
        baseline_np, baseline_da.isel(channel_id=channel_id).values
    )


def test_waa_schleiss_kwarg():
    cml = init_data()
    # first do simple wet-dry classification
    # TODO add other methods, maybe sftf when it works with new xarray interface
    cml["wet"] = cml.trsl.rolling(time=60, center=True).std()

    # call baseline function using kwargs
    cml["baseline"] = pycml.processing.baseline.baseline_constant(
        trsl=cml.trsl,
        wet=cml.wet,
    )

    waa_da = pycml.processing.wet_antenna.waa_schleiss_2013(
        rsl=cml.trsl,
        baseline=cml.baseline,
        wet=cml.wet,
        waa_max=2.3,
        delta_t=1,
        tau=15,
    )

    for channel_id in range(len(cml.channel_id)):
        waa_np = pycml.processing.wet_antenna.waa_schleiss_2013(
            rsl=cml.trsl.isel(channel_id=channel_id).values,
            baseline=cml.baseline.isel(channel_id=channel_id).values,
            wet=cml.wet.isel(channel_id=channel_id).values,
            waa_max=2.3,
            delta_t=1,
            tau=15,
        )
        np.testing.assert_almost_equal(
            waa_np, waa_da.isel(channel_id=channel_id).values
        )


def test_waa_leijnse_kwarg():
    cml = init_data()
    cml["wet"] = cml.trsl.rolling(time=60, center=True).std()

    # call baseline function using kwargs
    cml["baseline"] = pycml.processing.baseline.baseline_constant(
        trsl=cml.trsl,
        wet=cml.wet,
    )

    cml["A"] = cml.trsl - cml.baseline
    cml["A"] = cml.A.where((cml.A.isnull().values | (cml.A.values >= 0)), 0)

    waa_da = pycml.processing.wet_antenna.waa_leijnse_2008_from_A_obs(
        A_obs=cml.A,
        f_Hz=cml.frequency * 1e9,
        pol=cml.polarization,
        L_km=cml.length,
        T_K=293.0,
        gamma=2.06e-05,
        delta=0.24,
        n_antenna=(1.73 + 0.014j),
        l_antenna=0.001,
    )

    for channel_id in range(len(cml.channel_id)):
        waa_np = pycml.processing.wet_antenna.waa_leijnse_2008_from_A_obs(
            A_obs=cml.A.isel(channel_id=channel_id).values,
            f_Hz=cml.frequency.isel(channel_id=channel_id).values * 1e9,
            pol=cml.isel(channel_id=channel_id).polarization,
            L_km=cml.length,
            T_K=293.0,
            gamma=2.06e-05,
            delta=0.24,
            n_antenna=(1.73 + 0.014j),
            l_antenna=0.001,
        )
        np.testing.assert_almost_equal(
            waa_np, waa_da.isel(channel_id=channel_id).values
        )


def test_calc_R_from_A():
    cml = init_data()
    cml["wet"] = cml.trsl.rolling(time=60, center=True).std()

    # call baseline function using kwargs
    cml["baseline"] = pycml.processing.baseline.baseline_constant(
        trsl=cml.trsl,
        wet=cml.wet,
    )

    cml["A"] = cml.trsl - cml.baseline
    cml["A"] = cml.A.where((cml.A.isnull().values | (cml.A.values >= 0)), 0)

    cml["R"] = pycml.processing.k_R_relation.calc_R_from_A(
        A=cml.A,
        L_km=cml.length,
        f_GHz=cml.frequency,
        pol=cml.polarization,
    )

    for channel_id in range(len(cml.channel_id)):
        R_np = pycml.processing.k_R_relation.calc_R_from_A(
            A=cml.isel(channel_id=channel_id).A.values,
            L_km=cml.isel(channel_id=channel_id).length.values,
            f_GHz=cml.isel(channel_id=channel_id).frequency,
            pol=cml.polarization,
        )
        np.testing.assert_almost_equal(R_np, cml.R.isel(channel_id=channel_id).values)


# TODO Add test for using positional args
