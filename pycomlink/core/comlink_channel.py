#----------------------------------------------------------------------------
# Name:         comlink_channel
# Purpose:      Class that represents one channel of a CML, holding the
#               TX and RX data as well as info on frequency
#               and polarization.
#
# Authors:      Christian Chwala
#
# Created:      24.02.2016
# Copyright:    (c) Christian Chwala 2016
# Licence:      The MIT License
#----------------------------------------------------------------------------


import numpy as np
import pandas as pd

class ComlinkChannel(pd.DataFrame):
    """A class for holding CML channel data and metadata"""

    #
    # !!! Many of the subclassing things need pandas >=0.17 !!!
    #

    # According to the pandas docs this is necessary to have
    # additional attributes
    _metadata = ['f_GHz', 'pol', 'atpc', 'id',
                 'sampling_type', 'temporal_resolution',]

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------

        data: pandas.DataFrame, or everything that DataFrame.__init__() digests

        index: list, np.array, or everything that DataFrame.__init__() digest

        rx: list or np.array

        tx: list or np.array

        rx_min: list or np.array

        rx_max: list or np.array

        f_GHz: float

        pol: str {'h', 'v', 'H', 'V'}

        atpc: boolean

        """
        f_GHz = kwargs.pop('f_GHz', None)
        pol = kwargs.pop('pol', None)
        atpc = kwargs.pop('atpc', None)

        data = kwargs.pop('data', None)

        t = kwargs.pop('t', None)
        rx = kwargs.pop('rx', None)
        tx = kwargs.pop('tx', None)
        rx_min = kwargs.pop('rx_min', None)
        rx_max = kwargs.pop('rx_max', None)

        # TODO: If this is not supplied we should maybe derive it somehow
        self.sampling_type = None

        # Parse the different data relevant kwargs to a DataFrame
        # and add it back to the kwargs so that we can pass this
        # then on to the pandas.DataFrame.__init__() below
        kwargs['data'] = self._parse_kwargs_to_dataframe(
            data=data, t=t, rx=rx, tx=tx, rx_min=rx_min, rx_max=rx_max)

        super(ComlinkChannel, self).__init__(*args, **kwargs)

        # TODO: Sanely parse metadata
        self.f_GHz = f_GHz

    def _infer_column_names(self):
        pass

    def __eq__(self):
        pass

#    def __getitem__(self, key):
#        result = super(ComlinkChannel, self).__getitem__(key)
#        return result

    @property
    def _constructor(self):
        return ComlinkChannel

    def _parse_kwargs_to_dataframe(self, data, t, rx, tx, rx_min, rx_max):
        # The case where only `t` and `rx` are supplied
        if ((data is None) and
            (tx is None) and
            (rx is not None) and
            (t is not None)):
            if (rx_min is not None) or (rx_max is not None):
                raise ValueError('`rx_min` and `rx_max` must not be supplied ' \
                                 'if `rx` is supplied')
            return pd.DataFrame(index=t, data={'rx': rx})

        # The case where `data` has been supplied.
        # We check that `data` is a DataFrame below.
        elif data is not None:
            if ((tx is not None) or (rx is not None) or
                (rx_min is not None) or (rx_max is not None) or
                (t is not None)):
                raise ValueError('`rx`, `tx`, `rx_min`, `rx_max`  and  `t` '\
                                 'must not be supplied if `data` is supplied')
            if isinstance(data, pd.DataFrame):
                # `data` is what we want, so put it back to the kwargs
                return data

            else:
                raise ValueError('`data` must be a pandas.DataFrame')

        else:
            raise ValueError('Could not parse the supplied arguments')