import unittest
import tempfile
import os
import shutil
from glob import glob

import pandas as pd

import pycomlink as pycml

cml_list_instantaneous = pycml.io.examples.get_75_cmls()[0:11]
cml_list_min_max = pycml.io.examples.get_75_cmls_min_max()[0:11]


class TemporaryDirectory(object):
    """ Context manager for tempfile.mkdtemp() to use it with "with" statement.
    Taken from
    https://stackoverflow.com/questions/6884991/how-to-delete-dir-created-by-python-tempfile-mkdtemp
    """
    def __enter__(self):
        self.name = tempfile.mkdtemp()
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.name)


# Round trip testing for cmlh5 write/read
class TestRoundTripCmlh5(unittest.TestCase):
    def setUp(self):
        self.cml_list = cml_list_instantaneous

    def testWithOneFileOneCml(self):
        cml = self.cml_list[0]

        with TemporaryDirectory() as temp_dir:
            temp_fn = os.path.join(temp_dir, 'test_cmlh5.h5')
            pycml.io.write_to_cmlh5([cml, ], fn=temp_fn)

            cml_loaded = pycml.io.read_from_cmlh5(fn=temp_fn)[0]

            assert(cml.metadata['cml_id'] == cml_loaded.metadata['cml_id'])
            for ch_name in cml.channels.keys():
                assert(cml.channels[ch_name].metadata['frequency'] ==
                       cml_loaded.channels[ch_name].metadata['frequency'])
                pd.util.testing.assert_frame_equal(
                    cml.channels[ch_name].data,
                    cml_loaded.channels[ch_name].data)

    def testWithOneFileThreeCmls(self):
        cml_list = self.cml_list
        with TemporaryDirectory() as temp_dir:
            temp_fn = os.path.join(temp_dir, 'test_cmlh5.h5')
            pycml.io.write_to_cmlh5(cml_list, fn=temp_fn)

            cml_list_loaded = pycml.io.read_from_cmlh5(fn=temp_fn)

            for cml, cml_loaded in zip(cml_list, cml_list_loaded):
                assert(cml.metadata['cml_id'] == cml_loaded.metadata['cml_id'])
                for ch_name in cml.channels.keys():
                    pd.util.testing.assert_frame_equal(
                        cml.channels[ch_name].data,
                        cml_loaded.channels[ch_name].data)

    def testWithMultiFilesTwoCmls(self):
        cml_list = self.cml_list
        with TemporaryDirectory() as temp_dir:
            temp_fn = os.path.join(temp_dir, 'test_cmlh5.h5')
            pycml.io.write_to_cmlh5(cml_list,
                                    fn=temp_fn,
                                    splitting_period='D',
                                    split_to_multiple_files=True)

            temp_fn_list = glob(temp_fn.split('.')[0] + '*')
            assert(len(temp_fn_list) ==
                   len(cml_list[0].channel_1.data.resample('D').mean().index))
            cml_list_loaded = pycml.io.cmlh5.read_from_multiple_cmlh5(
                fn_list=temp_fn_list)

            for cml, cml_loaded in zip(cml_list, cml_list_loaded):
                assert(cml.metadata['cml_id'] == cml_loaded.metadata['cml_id'])
                for ch_name in cml.channels.keys():
                    pd.util.testing.assert_frame_equal(
                        cml.channels[ch_name].data,
                        cml_loaded.channels[ch_name].data)


class TestRoundTripCmlh5MinMax(TestRoundTripCmlh5):
    def setUp(self):
        self.cml_list = cml_list_min_max
