# ----------------------------------------------------------------------------
# Name:         comlink
# Purpose:      Class that represents one CML, which consists of several
#               ComlinkChannels and CML-specific metadata like coordinates
#               of the TX- and RX-sites
#
# Authors:      Christian Chwala
#
# Created:      21.04.2016
# Copyright:    (c) Christian Chwala 2016
# Licence:      The MIT License
# ----------------------------------------------------------------------------

import warnings

from comlink_channel import ComlinkChannel


class Comlink(object):
    """ A class representing a CML with its channels and metadata"""

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------

        channels : ComlinkChannel or list of those

        site_metadata : dict

        """

        channels = kwargs.pop('channels', None)

        # If no channels are supplied, there must be at least `t`, `rx` and
        # the necessary channel metadata to automatically build a ComlinkChannel
        if channels is None:
               t = kwargs.pop('t')
               rx = kwargs.pop('rx')
               tx = kwargs.pop('tx')

        elif type(channels) == ComlinkChannel:
            channels = [channels]
        elif type(channels) == list:
            for channel in channels:
                if not type(channel) == ComlinkChannel:
                    raise ValueError\
                        ('`channels` must be of type ComlinkChannel')
        else:
            raise AttributeError('`channels` is %s must be either a ComlinkChannel or a list of ComlinkChannels' %
                                 type(channels))

        # if channels are supplied, the channel metadata should not be supplied since
        # it will have no effect, because they are already part of the individual ComlinkChannels
        if channels is not None:
            if ((kwargs.has_key('t')) or
                    (kwargs.has_key('rx')) or
                    (kwargs.has_key('tx')) or
                    (kwargs.has_key('f_GHz')) or
                    (kwargs.has_key('pol'))):
                warnings.warn('All supplied channel metadata (e.g. f_GHz) has no effect, '
                              'since they are already contained in the supplied ComlinkChannel')

        self._channel_dict = _channels_list_to_dict(channels)

    def __getattr__(self, item):
        if ((item.split('_')[0] == 'channel') and
                (type(int(item.split('_')[1])) == int)):
            channel_n = int(item.split('_')[1])-1
            if channel_n < 0:
                raise AttributeError('The channel number must be >= 1')
            return self._channel_dict[item]
        else:
            raise AttributeError('`Comlink` has no attribute %s' % item)

    def _repr_html_(self):
        html_str = '<table> <tr> '
        for channel_name in self._channel_dict:
            cml_ch = self._channel_dict[channel_name]
            html_str = (html_str + '<td> ' +
                        channel_name + '<br/>' +
                        cml_ch._repr_html_() + '</td>')
        html_str = html_str + '</tr>' + '</table>'
        return html_str

    def __dir__(self):
        return self.__dict__.keys() + self._channel_dict.keys()


def _channels_list_to_dict(channels):
    channel_dict = {}
    for i, channel in enumerate(channels):
        channel_dict['channel_' + str(i+1)] = channel
    return channel_dict

