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
        rsl=cml.trsl,
        wet=cml.wet,
    )
    baseline_ch0 = pycml.processing.baseline.baseline_constant(
        rsl=cml.trsl.isel(channel_id=0).values,
        wet=cml.wet.isel(channel_id=0).values,
    )
    baseline_ch1 = pycml.processing.baseline.baseline_constant(
        rsl=cml.trsl.isel(channel_id=1).values,
        wet=cml.wet.isel(channel_id=1).values,
    )

    np.testing.assert_almost_equal(baseline_ch0, baseline_da.isel(channel_id=0).values)
    np.testing.assert_almost_equal(baseline_ch1, baseline_da.isel(channel_id=1).values)
