import pkg_resources
from os import path

from ..util.maintenance import deprecated


from pycomlink.io.cmlh5 import read_from_cmlh5


def get_example_data_path():
    return pkg_resources.resource_filename('pycomlink', 'io/example_data')


def read_one_cml():
    data_path = get_example_data_path()
    fn = 'one_cml.h5'
    return read_from_cmlh5(path.join(data_path, fn))[0]


@deprecated('This function will be depreciated. Use `get_75_cmls() instead.')
def read_cml_list():
    data_path = get_example_data_path()
    fn = '75_cmls.h5'
    return read_from_cmlh5(path.join(data_path, fn))


def get_75_cmls():
    data_path = get_example_data_path()
    fn = '75_cmls.h5'
    return read_from_cmlh5(path.join(data_path, fn))


def get_75_cmls_min_max():
    data_path = get_example_data_path()
    fn = '75_cmls_min_max.h5'
    return read_from_cmlh5(path.join(data_path, fn))
