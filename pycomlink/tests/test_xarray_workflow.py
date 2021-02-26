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
    np.testing.assert_almost_equal(waa_np, waa_da.isel(channel_id=channel_id).values)


# TODO Add test for using positional args
