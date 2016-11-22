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
from copy import deepcopy
from collections import namedtuple

import matplotlib.pyplot as plt
import folium

from comlink_channel import ComlinkChannel


class Comlink(object):
    """ A class representing a CML with its channels and metadata"""

    def __init__(self, channels=None, metadata=None, **kwargs):
        """

        WIP

        Parameters
        ----------

        channels : ComlinkChannel or list of those

        metadata : dict

        """

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
                # Duck-type to see if it behaves like a ComlinkChannel
                try:
                    channel._df['rx']
                except Exception as e:
                    raise e('`channels` must behave like a ComlinkChannel')
        else:
            raise AttributeError('`channels` is %s must be either a '
                                 'ComlinkChannel or a list of ComlinkChannels' %
                                 type(channels))

        # if channels are supplied, the channel metadata should not be
        # supplied since it will have no effect, because they are already
        # part of the individual ComlinkChannels
        if channels is not None:
            if ((kwargs.has_key('t')) or
                    (kwargs.has_key('rx')) or
                    (kwargs.has_key('tx')) or
                    (kwargs.has_key('f_GHz')) or
                    (kwargs.has_key('pol'))):
                warnings.warn('All supplied channel metadata (e.g. f_GHz) '
                              'has no effect, since they are already '
                              'contained in the supplied ComlinkChannel')

        self.channels = _channels_list_to_dict(channels)

        self.metadata = {'site_a_latitude': metadata['site_a_latitude'],
                         'site_a_longitude': metadata['site_a_longitude'],
                         'site_b_latitude': metadata['site_b_latitude'],
                         'site_b_longitude': metadata['site_b_longitude'],
                         'cml_id': metadata['cml_id']}

    def __getattr__(self, item):
        if ((item.split('_')[0] == 'channel') and
                (type(int(item.split('_')[1])) == int)):
            channel_n = int(item.split('_')[1])-1
            if channel_n < 0:
                raise AttributeError('The channel number must be >= 1')
            return self.channels[item]
        else:
            raise AttributeError('`Comlink` has no attribute %s' % item)

    def _repr_html_(self):
        html_str = '<table> <tr> '
        for channel_name in self.channels:
            cml_ch = self.channels[channel_name]
            html_str = (html_str + '<td> ' +
                        '<b>' + channel_name + '</b><br/>' +
                        cml_ch._repr_html_() + '</td>')
        html_str = html_str + '</tr>' + '</table>'
        return html_str

    def __dir__(self):
        return self.__dict__.keys() + self.channels.keys()

    def __copy__(self):
        cls = self.__class__
        new_cml = cls.__new__(cls)
        new_cml.__dict__.update(self.__dict__)
        return new_cml

    def __deepcopy__(self, memo=None):
        new_cml = self.__copy__()
        if memo is None:
            memo = {}
        memo[id(self)] = new_cml
        #for name, channel in self.channels.iteritems():
        #    new_cml.channels[name] = deepcopy(channel, memo)
        new_cml.metadata = deepcopy(self.metadata, memo)
        new_cml.channels = deepcopy(self.channels, memo)
        return new_cml

    def get_coordinates(self):
        Coords = namedtuple('coords', ['lon_a', 'lon_b', 'lat_a', 'lat_b'])
        coords = Coords(lon_a=self.metadata['site_a_longitude'],
                        lon_b=self.metadata['site_b_longitude'],
                        lat_a=self.metadata['site_a_latitude'],
                        lat_b=self.metadata['site_b_latitude'])
        return coords

    def plot_map(self, tiles='OpenStreetMap', fol_map=None):
        coords = self.get_coordinates()

        if fol_map is None:
            fol_map = folium.Map(location=[(coords.lat_a + coords.lat_b)/2,
                                           (coords.lon_a + coords.lon_b)/2],
                                 tiles=tiles,
                                 zoom_start=11)
        fol_map.add_children(folium.PolyLine([(coords.lat_a, coords.lon_a),
                                              (coords.lat_b, coords.lon_b)]))
        return fol_map

    def plot_line(self, ax=None, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        coords = self.get_coordinates()
        ax.plot([coords.lon_a, coords.lon_b],
                [coords.lat_a, coords.lat_b],
                *args, **kwargs)
        return ax

    def plot_data(self, columns=['rx',], ax=None):
        if ax is None:
            fig, ax = plt.subplots(len(columns),
                                   1,
                                   figsize=(12, 2*len(columns) + 1),
                                   sharex=True)
        try:
            ax[0].get_alpha()
        except TypeError:
            ax = [ax, ]

        for ax_i, column in zip(ax, columns):
            for i, (name, cml_ch) in enumerate(self.channels.iteritems()):
                if column == 'wet':
                    ax_i.fill_between(
                        cml_ch._df[column].index,
                        i,
                        i + cml_ch._df[column].values,
                        alpha=0.9,
                        linewidth=0.0,
                        label=name)
                else:
                    ax_i.plot(
                        cml_ch._df[column].index,
                        cml_ch._df[column].values,
                        label=name)
            ax_i.set_ylabel(column)

        return ax

    def get_center_lon_lat(self):
        coords = self.get_coordinates()
        center_lon = (coords.lon_a + coords.lon_b) / 2.0
        center_lat = (coords.lat_a + coords.lat_b) / 2.0
        return center_lon, center_lat


def _channels_list_to_dict(channels):
    channel_dict = {}
    for i, channel in enumerate(channels):
        channel_dict['channel_' + str(i+1)] = channel
    return channel_dict

