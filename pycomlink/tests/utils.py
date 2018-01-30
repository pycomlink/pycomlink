import pycomlink as pycml


def load_and_clean_example_cml():
    cml = pycml.io.examples.read_one_cml()
    cml.process.quality_control.set_to_nan_if('tx', '>=', 100)
    cml.process.quality_control.set_to_nan_if('rx', '==', -99.9)
    for cml_ch in cml.channels.values():
        cml_ch.data.interpolate(limit=3)
    return cml


def load_and_clean_example_cml_list():
    cml_list = pycml.io.examples.get_75_cmls()
    for cml in cml_list:
        cml.process.quality_control.set_to_nan_if('tx', '>=', 100)
        cml.process.quality_control.set_to_nan_if('rx', '==', -99.9)
        for cml_ch in cml.channels.values():
            cml_ch.data.interpolate(limit=3)
    return cml_list
