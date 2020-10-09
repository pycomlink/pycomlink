import pkg_resources
import os.path
import pycomlink as pycml


def get_test_data_path():
    return pkg_resources.resource_filename("pycomlink", "tests/test_data")


def load_and_clean_example_cml():
    cml = pycml.io.examples.read_one_cml()
    cml.process.quality_control.set_to_nan_if("tx", ">=", 100)
    cml.process.quality_control.set_to_nan_if("rx", "==", -99.9)
    for cml_ch in cml.channels.values():
        cml_ch.data.interpolate(limit=3)
    return cml


def load_processed_cml_list():
    data_path = get_test_data_path()
    fn = "75_cmls_processed.h5"
    return pycml.io.read_from_cmlh5(os.path.join(data_path, fn), read_all_data=True)
