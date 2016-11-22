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

from pycomlink.processing.wet_dry import std_dev, stft
from pycomlink.processing.baseline.baseline import \
    baseline_linear, baseline_constant
from pycomlink.processing.A_R_relation.A_R_relation import calc_R_from_A


class Processor(object):
    def __init__(self, cml):
        self.cml = deepcopy(cml)

        self.wet_dry = WetDry(self.cml)

        self.baseline = Baseline(self.cml)

        self.A_R = A_R(self.cml)


class WetDry(object):
    def __init__(self, cml):
        self.std_dev = cml_wrapper(cml,
                                   std_dev.std_dev_classification,
                                   'rx',
                                   'wet',
                                   returns_temp_results=True)

        self.stft = cml_wrapper(cml,
                                stft.stft_classification,
                                'rx',
                                'wet',
                                returns_temp_results=True)


class Baseline(object):
    def __init__(self, cml):
        self.cml = cml

        self.linear = cml_wrapper(cml,
                                  baseline_linear,
                                  ['rx', 'wet'],
                                  'baseline')
        self.constant = cml_wrapper(cml,
                                    baseline_constant,
                                    ['rx', 'wet'],
                                    'baseline')

    # TODO: Integarte this somewhere else, since this
    #       sould be carried out after every baseline determination
    def calc_A(self):
        for ch_name, cml_ch in self.cml.channels.iteritems():
            cml_ch._df['A'] = cml_ch._df['baseline'] - cml_ch._df['rx']


class A_R(object):
    def __init__(self, cml):
        self.calc_R = cml_wrapper(cml,
                                  calc_R_from_A,
                                  ['A'],
                                  'R',
                                  L=4.6,
                                  f=18.9) # TODO: Add metadata for length and frequency to Comlink


def cml_wrapper(cml, func, vars_in, var_out, returns_temp_results=False, **additional_kwargs):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        # Make sure that we have a list of vars_in
        if type(vars_in) == str:
            vars_in_list = [vars_in, ]
        else:
            vars_in_list = vars_in
        # Reverse list to have the correct order when appending to args
        vars_in_list.reverse()

        # Iterate over channels
        args_initial = deepcopy(args)
        for name, cml_ch in cml.channels.iteritems():
            # Add var_in variables to args-list
            args = list(deepcopy(args_initial))
            for var_in in vars_in_list:
                args.insert(0, cml_ch._df[var_in].values)

            # Add additional kwargs
            for k, v in additional_kwargs.iteritems():
                kwargs[k] = v

            # Call processing function
            temp = func(*args, **kwargs)

            # Parse results
            if returns_temp_results:
                ts, temp_result_dict = temp
                try:
                    cml_ch.intermediate_results.keys()
                except AttributeError:
                    cml_ch.intermediate_results = {}
                for key, value in temp_result_dict.iteritems():
                    cml_ch.intermediate_results[key] = value
            else:
                ts = temp
            cml_ch._df[var_out] = ts

        return None
    return func_wrapper
