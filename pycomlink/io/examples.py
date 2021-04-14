import pkg_resources
from os import path


from pycomlink.io.cmlh5_to_xarray import read_cmlh5_file_to_xarray


def get_example_data_path():
    return pkg_resources.resource_filename("pycomlink", "io/example_data")


def get_one_cml():
    data_path = get_example_data_path()
    fn = "one_cml.h5"
    return read_cmlh5_file_to_xarray(path.join(data_path, fn))[0]


def get_75_cmls():
    data_path = get_example_data_path()
    fn = "75_cmls.h5"
    return read_cmlh5_file_to_xarray(path.join(data_path, fn))


def get_75_cmls_min_max():
    data_path = get_example_data_path()
    fn = "75_cmls_min_max.h5"
    return read_cmlh5_file_to_xarray(path.join(data_path, fn))
