import pkg_resources
from os import path

from pycomlink.io.cmlh5 import read_from_cmlh5


def get_example_data_path():
    return pkg_resources.resource_filename('pycomlink',
                                           '../notebooks/example_data')


def read_one_cml():
    data_path = get_example_data_path()
    fn = 'one_cml.h5'
    return read_from_cmlh5(path.join(data_path, fn))[0]


def read_cml_list():
    data_path = get_example_data_path()
    fn = '75_cmls.h5'
    return read_from_cmlh5(path.join(data_path, fn))
