# ----------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import copy
import warnings


class ComlinkChannel(object):
    """A class for holding CML channel data and metadata"""

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------

        data: pandas.DataFrame, or everything that DataFrame.__init__() digests

        t: list, np.array, or everything that DataFrame.__init__() digest

        rx: list or np.array

        tx: list, np.array, float or int
            TX power. If only a scalar value is supplied, it is interpreted
            as the constant TX power.

        f_GHz: float

        pol: str {'h', 'v', 'H', 'V'}

        atpc: str {'on', 'off'}

        channel_id: str

        metadata: dict
            Default is None.

        """
        f_GHz = kwargs.pop('f_GHz', None)
        pol = kwargs.pop('pol', None)
        atpc = kwargs.pop('atpc', None)
        channel_id = kwargs.pop('channel_id', None)
        metadata = kwargs.pop('metadata', None)

        data = kwargs.pop('data', None)

        t = kwargs.pop('t', None)
        rx = kwargs.pop('rx', None)
        tx = kwargs.pop('tx', None)

        # TODO: If this is not supplied we should maybe derive it somehow
        self.sampling_type = None

        # Parse the different data relevant kwargs to a DataFrame
        # and add it back to the kwargs so that we can pass this
        # then on to the pandas.DataFrame.__init__() below
        kwargs['data'] = _parse_kwargs_to_dataframe(
            data=data, t=t, rx=rx, tx=tx)

        # super(ComlinkChannel, self).__init__(*args, **kwargs)
        self.data = kwargs.pop('data')

        # TODO: Sanely parse metadata
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = {
                'frequency': f_GHz * 1e9,
                'polarization': pol,
                'channel_id': channel_id,
                'atpc': atpc}

        # TODO: Remove this
        # Keeping this for now for backwards compatibility
        self.f_GHz = self.metadata['frequency'] / 1e9

    def __eq__(self):
        pass

    def __getitem__(self, key):
        new_cml_ch = self.__copy__()
        new_cml_ch.data = self.data.__getitem__(key)
        return new_cml_ch

    def __len__(self):
        return len(self.data)

    def __str__(self, *args, **kwargs):
        print 'f_GHz: ', self.f_GHz
        print self.data.__str__()

    def __getattr__(self, item):
        try:
            return self.data.__getattr__(item)
        except:
            raise AttributeError('Neither \'ComlinkChannel\' nor its '
                                 '\'DataFrame\' have the attribute \'%s\''
                                 % item)

    def __copy__(self):
        cls = self.__class__
        new_cml_ch = cls.__new__(cls)
        new_cml_ch.__dict__.update(self.__dict__)
        return new_cml_ch

    def __deepcopy__(self, memo=None):
        new_cml_ch = self.__copy__()
        if memo is None:
            memo = {}
        memo[id(self)] = new_cml_ch
        new_cml_ch.data = copy.deepcopy(self.data, memo)
        return new_cml_ch

    def _repr_html_(self):
        metadata_str = ''
        for key, value in self.metadata.iteritems():
            if key == 'frequency':
                metadata_str += (str(key) + ': ' + str(value/1e9) + ' GHz<br/>')
            else:
                metadata_str += (str(key) + ': ' + str(value) + '<br/>')
        return metadata_str + self.data._repr_html_()

    def copy(self):
        return self.__deepcopy__()
        #return copy.deepcopy(self)

    def resample(self, *args, **kwargs):
        inplace = kwargs.pop('inplace', False)
        how = kwargs.pop('how', np.mean)

        if inplace:
            self.data = self.data.resample(*args, **kwargs).apply(how)
        elif not inplace:
            new_cml_ch = copy.copy(self)
            new_cml_ch._df = self.data.resample(*args, **kwargs).apply(how)
            return new_cml_ch
        else:
            raise ValueError('`inplace` must be either True or False')


def _parse_kwargs_to_dataframe(data, t, rx, tx):
    # The case where only `t`, `rx` and `tx` are supplied
    if data is None:
        df = pd.DataFrame(index=t, data={'rx': rx})
        df['tx'] = tx

    # The case where `data` has been supplied.
    # We check that `data` is a DataFrame and that the DataFrame has the
    # columns `tx` and `rx`.
    elif data is not None:
        if isinstance(data, pd.DataFrame):
            # `data` is what we want, so return it
            df = data
            try:
                df.tx
            except AttributeError:
                raise AttributeError('DataFrame `data` must have a column `tx`')
            try:
                df.rx
            except AttributeError:
                raise AttributeError('DataFrame `data` must have a column `tx`')
        else:
            raise ValueError('type of `data` is %s, '
                             'but must be pandas.DataFrame' % type(data))

    else:
        raise ValueError('Could not parse the supplied arguments')

    df['txrx'] = df.tx - df.rx
    df.index.name = 'time'
    return df
