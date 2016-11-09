#----------------------------------------------------------------------------
# Name:         
# Purpose:      
#
# Authors:      
#
# Created:      
# Copyright:    (c) Christian Chwala 2014
# Licence:      The MIT License
#----------------------------------------------------------------------------

from functools import wraps
from copy import deepcopy

from pycomlink.wet_dry import std_dev


class Processor(object):
    def __init__(self, cml):
        self.cml = deepcopy(cml)

        self.wet_dry = WetDry(self.cml)


class WetDry(object):
    def __init__(self, cml):
        self.std_dev = cml_to_txrx_to_cml(cml,
                                          std_dev.std_dev_classification,
                                          'wet')


def cml_to_txrx_to_cml(cml, func, variable_name):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        for name, cml_ch in cml.channels.iteritems():
            ts, temp_result_dict = func(cml_ch.rx, *args, **kwargs)
            cml_ch._df[variable_name] = ts
            try:
                cml_ch.intermediate_results.keys()
            except AttributeError:
                cml_ch.intermediate_results = {}
            for key, value in temp_result_dict.iteritems():
                cml_ch.intermediate_results[key] = value
        return None
    return func_wrapper
