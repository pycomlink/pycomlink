#----------------------------------------------------------------------------
# Name:         
# Purpose:      
#
# Authors:      
#
# Created:      
# Copyright:    (c) Christian Chwala 2016
# Licence:      The MIT License
#----------------------------------------------------------------------------

import numpy as np


def set_to_nan_if(cml, ts_name, expression, value):
    if expression == '==':
        def func(a, b): return a == b
    elif expression == '>=':
        def func(a, b): return a >= b
    elif expression == '<=':
        def func(a, b): return a <= b
    else:
        raise ValueError('expresison `%s` not supported' % expression)

    for ch_name, ch in cml.channels.items():
        index = func(ch.data[ts_name], value)
        ch.data.loc[index, ts_name] = np.nan
        ch.data['txrx'] = ch.data['tx'] - ch.data['rx']
    return cml
