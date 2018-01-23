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

from __future__ import print_function
from __future__ import division

from builtins import str
from builtins import object
import numpy as np
import pandas as pd
import copy


class ComlinkChannel(object):
    """A class for holding CML channel data and metadata"""

    def __init__(self, data=None, metadata=None,
                 t=None, rx=None, tx=None,
                 frequency=None, polarization=None,
                 atpc=None, channel_id=None):
        """

        Parameters
        ----------

        data: pandas.DataFrame
            DataFrame with the columns `tx` and `rx` holding the time series
            of the TX and RX level, respectively. The index of the DataFrame
            must contain the time stamps. If the TX level is constant,
            please still supply of full time series for it. You can specify
            that the TX level is constant by passing `atpc = 'off'`.

        t: list, np.array, or everything that DataFrame.__init__() digest

        rx: list or np.array, or everything that DataFrame.__init__() digest
            Timer series of RX power.

        tx: list, np.array, float or int
            Timer series of TX power. If only a scalar value is supplied,
            it is interpreted as the constant TX power.

        frequency: float
            Frequency in Hz.

        polarization: str {'h', 'v', 'H', 'V'}
            Polarization

        atpc: str {'on', 'off'}
            Flag to specifiy if ATPC (Automatic Transmission Power Control),
            i.e. a variable TX level, is active or not

        channel_id: str
            The ID of this channel.

        metadata: dict
           Dictionary with metadata, where this is an example of the minimum
           of information that has to be supplied in the dict, if it is not
           supplied seperately

           {'frequency': 20 * 1e9,
            'polarization': 'V',
            'channel_id': 'channel_xy'
            'atpc': 'off'}

        """

        # TODO: If this is not supplied we should maybe derive it somehow
        self.sampling_type = None

        # Handle the different arguments and build a DataFrame from them
        # if it has not been supplied as `data`
        self.data = copy.deepcopy(
            _parse_kwargs_to_dataframe(data=data, t=t, rx=rx, tx=tx))

        # TODO: Sanely parse metadata
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = {
                'frequency': frequency,
                'polarization': polarization,
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
        print('f_GHz: ', self.f_GHz)
        print(self.data.__str__())

    def __getattr__(self, item):
        try:
            return self.data.__getattr__(item)
        except:
            raise AttributeError('Neither \'ComlinkChannel\' nor its '
                                 '\'DataFrame\' have the attribute \'%s\''
                                 % item)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

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
        for key, value in self.metadata.items():
            if key == 'frequency':
                metadata_str += (str(key) + ': ' + str(value / 1e9) + ' GHz<br/>')
            else:
                metadata_str += (str(key) + ': ' + str(value) + '<br/>')
        return metadata_str + self.data._repr_html_()

    def copy(self):
        """ Return a deepcopy of this channel object """
        return self.__deepcopy__()

    def resample(self, resampling_time, how=np.mean, inplace=False):
        """ Resample channel data

        Parameters
        ----------

        resampling_time : str
            The frequency to which you want to resample. Use the pandas
            notation, e.g. '5min' for 5 minutes or '3H' for three hours.

        how : function, optional
            The function to be applied for resampling. Defaults to `np.mean`,
            but e.g. also `np.max`, `np.min` or `np.sum` can make sense,
            depending on what you want

        inplace : bool, optional
            If set to `True` the resampling will be carried out directly on
            this `ComlinkChannel`. If set to `False`  a copy of the current
            `ComlinkChannel` with the resampled data will be returned. The
            original channel and its data will not be altered.

        Example
        -------

        # Resample an existing channel to 5 minutes
        cml_ch_5min = cml_ch_1min.resample('5min', inplace=False, how=np.mean)

        """

        if inplace:
            self.data = self.data.resample(resampling_time).apply(how)
        elif not inplace:
            new_cml_ch = copy.copy(self)
            new_cml_ch.data = self.data.resample(resampling_time).apply(how)
            return new_cml_ch
        else:
            raise ValueError('`inplace` must be either True or False')

    def append_data(self, cml_ch, max_length=None, max_age=None):
        """ Append data to the current channel

        Parameters
        ----------
        cml_ch
        max_length
        max_age

        Returns
        -------

        """

        for key in self.metadata.keys():
            if self.metadata[key] != cml_ch.metadata[key]:
                raise ValueError('ComlinkChannel metadata `%s` is different'
                                 'for the two channels: %s vs. %s' %
                                 (key,
                                  self.metadata[key],
                                  cml_ch.metadata[key]))

        self.data = self.data.append(cml_ch.data)
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        self.data = self.data.sort_index()

        if max_length is not None:
            if max_length < len(self.data):
                self.data = self.data.iloc[-max_length:, :]

        if max_age is not None:
            min_allowed_timestamp = (self.data.index[-1] -
                                     pd.Timedelta(max_age))
            self.data = self.data.loc[self.data.index > min_allowed_timestamp]


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
            #try:
            #    df.tx
            #except AttributeError:
            #    raise AttributeError('DataFrame `data` must have a column `tx`')
            #try:
            #    df.rx
            #except AttributeError:
            #    raise AttributeError('DataFrame `data` must have a column `tx`')
        else:
            raise ValueError('type of `data` is %s, '
                             'but must be pandas.DataFrame' % type(data))

    else:
        raise ValueError('Could not parse the supplied arguments')

    # Quick fix to make this work for instantaneous and min-max data
    try:
        df['txrx'] = df.tx - df.rx
    except AttributeError:
        pass
    try:
        df['txrx_max'] = df.tx_max - df.rx_min
        df['txrx_min'] = df.tx_min - df.rx_max
    except AttributeError:
        pass

    df.index.name = 'time'
    return df
