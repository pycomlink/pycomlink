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

from builtins import object
from functools import wraps
from copy import deepcopy

from pycomlink.processing.wet_dry import std_dev, stft
from pycomlink.processing.baseline.baseline import \
    baseline_linear, baseline_constant
from pycomlink.processing.wet_antenna.wet_antenna import waa_adjust_baseline
from pycomlink.processing.A_R_relation.A_R_relation import calc_R_from_A
from pycomlink.processing.quality_control.simple import set_to_nan_if


class Processor(object):
    def __init__(self, cml):
        #self.cml = deepcopy(cml)
        self.cml = cml

        self.quality_control = QualityControl(self.cml)

        self.wet_dry = WetDry(self.cml)

        self.baseline = Baseline(self.cml)

        self.A_R = A_R(self.cml)

    def __copy__(self):
        cls = self.__class__
        new_proc = cls.__new__(cls)
        new_proc.__dict__.update(self.__dict__)
        return new_proc

    def __deepcopy__(self, memo=None):
        # TODO: Check the logic of this function. It seems to work, but why?

        if memo is None:
            memo = {}

        cml_from_last_copy = memo.get(self, False)

        copy_again = True
        if cml_from_last_copy:
            copy_again = False

        if not copy_again:
            #print '### NOT COPYING'
            return self
        else:
            #print '### COPYING'
            return Processor(deepcopy(self.cml, memo=memo))


class QualityControl(object):
    def __init__(self, cml):
        self.set_to_nan_if = pass_cml_wrapper(cml, set_to_nan_if)

        
class WetDry(object):
    def __init__(self, cml):
        self.std_dev = cml_wrapper(cml,
                                   std_dev.std_dev_classification,
                                   'txrx',
                                   'wet',
                                   returns_temp_results=True)

        self.stft = cml_wrapper(cml,
                                stft.stft_classification,
                                'txrx',
                                'wet',
                                returns_temp_results=True)


class Baseline(object):
    def __init__(self, cml):
        self._cml = cml
        self.linear = cml_wrapper(cml,
                                  baseline_linear,
                                  ['txrx', 'wet'],
                                  'baseline')
        self.constant = cml_wrapper(cml,
                                    baseline_constant,
                                    ['txrx', 'wet'],
                                    'baseline')

        self.waa_schleiss = cml_wrapper(cml,
                                        waa_adjust_baseline,
                                        ['txrx', 'baseline', 'wet'],
                                        'baseline')

        self.calc_A = cml_wrapper(cml,
                                  _calc_A,
                                  ['txrx', 'baseline'],
                                  'A')


# TODO: Integarte this somewhere else, since this
#       sould be carried out after every baseline determination
def _calc_A(txrx, baseline):
    return txrx - baseline


class A_R(object):
    def __init__(self, cml):
        # TODO: Make it possible to use individual f_GHz from each channel
        self.calc_R = cml_wrapper(cml,
                                  calc_R_from_A,
                                  ['A'],
                                  'R',
                                  L=cml.get_length(),
                                  f_GHz=cml.channel_1.f_GHz)


def cml_wrapper(cml, func,
                vars_in, var_out,
                returns_temp_results=False,
                **additional_kwargs):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        # Make sure that we have a list of vars_in
        if type(vars_in) == str:
            vars_in_list = [vars_in, ]
        else:
            vars_in_list = vars_in

        t_start = kwargs.pop('t_start', None)
        t_stop = kwargs.pop('t_stop', None)

        # Iterate over channels
        args_initial = deepcopy(args)
        for name, cml_ch in cml.channels.items():
            # Add var_in variables to args-list
            args = list(deepcopy(args_initial))

            # Go through list in reverse to have the correct order
            for var_in in reversed(vars_in_list):
                if (t_start is not None) and (t_stop is not None):
                    t_ix = ((cml_ch.data.index > t_start) &
                            (cml_ch.data.index < t_stop))
                elif t_start is not None:
                    t_ix = cml_ch.data.index > t_start
                elif t_stop is not None:
                    t_ix = cml_ch.data.index < t_stop

                if (t_start is not None) or (t_stop is not None):
                    args.insert(0, cml_ch.data.loc[t_ix, var_in].values)
                else:
                    args.insert(0, cml_ch.data.loc[:, var_in].values)

            # Add additional kwargs
            for k, v in additional_kwargs.items():
                kwargs[k] = v

            # Call processing function
            temp = func(*args, **kwargs)

            # Parse results
            if returns_temp_results:
                ts, temp_result_dict = temp
                try:
                    list(cml_ch.intermediate_results.keys())
                except AttributeError:
                    cml_ch.intermediate_results = {}
                for key, value in temp_result_dict.items():
                    cml_ch.intermediate_results[key] = value
            else:
                ts = temp

            if (t_start is not None) or (t_stop is not None):
                cml_ch.data.loc[t_ix, var_out] = ts
            else:
                cml_ch.data.loc[:, var_out] = ts
        
        return cml
    return func_wrapper


def pass_cml_wrapper(cml, func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        func(cml, *args, **kwargs)
        return cml
    return func_wrapper
