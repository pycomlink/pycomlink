from __future__ import absolute_import
from __future__ import division
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

from builtins import zip
from builtins import str
from builtins import object
import warnings
from copy import deepcopy
from collections import namedtuple, OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import folium

from .comlink_channel import ComlinkChannel
from ..processing import Processor
from ..spatial.helper import distance


# Assure that the pandas matplotlib converters are registered,
# as long as a new matplotlib release does not handle pandas
# time data (or np.datetime64) automatically
# TODO: Remove this when solved via new matplotlib, maybe in 2.2.something...
# here: https://github.com/matplotlib/matplotlib/pull/9779
import pandas.plotting
pandas.plotting.register_matplotlib_converters()

Coords = namedtuple('coords', ['lon_a', 'lon_b', 'lat_a', 'lat_b'])


class Comlink(object):
    """ A class representing a CML with its channels and metadata"""

    def __init__(self, channels=None, metadata=None, **kwargs):
        """
        Comlink object representing one physical (commercial) microwave link,
        abbreviated as CML. One CML can contain several CommlinkChannels,
        typically two, for both directions of communication.

        The preferred way to initialize a Comlink object is to initialize the
        CommlinkChannels first and pass them as argument here.

        Parameters
        ----------

        channels : ComlinkChannel or list of those

        metadata : dict
            Dictionary with basic CML metadata of the form
                {'site_a_latitude': 12.34,
                'site_a_longitude': 12.34,
                'site_b_latitude': 56.78,
                'site_b_longitude': 56.78,
                'cml_id': 'XY1234'}


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
                    channel.data
                except Exception:
                    raise AttributeError('`channels` must behave like a '
                                         'ComlinkChannel object')
        else:
            raise AttributeError('`channels` is %s must be either a '
                                 'ComlinkChannel or a list of ComlinkChannels' %
                                 type(channels))

        # if channels are supplied, channel metadata or separate data for
        # `t`, `tx` or `rx` should not be supplied since they will have no
        # effect, because they are already part of the individual
        # ComlinkChannels
        if channels is not None:
            if (('t' in kwargs) or
                    ('rx' in kwargs) or
                    ('tx' in kwargs) or
                    ('f_GHz' in kwargs) or
                    ('pol' in kwargs)):
                warnings.warn('All supplied channel metadata (e.g. f_GHz) '
                              'has no effect, since they are already '
                              'contained in the supplied ComlinkChannel')

        self.channels = _channels_list_to_ordered_dict(channels)

        self.metadata = {'site_a_latitude': metadata['site_a_latitude'],
                         'site_a_longitude': metadata['site_a_longitude'],
                         'site_b_latitude': metadata['site_b_latitude'],
                         'site_b_longitude': metadata['site_b_longitude'],
                         'cml_id': metadata['cml_id']}

        calculated_length = self.calc_length()

        if 'length' in list(metadata.keys()):
            length_diff = calculated_length - metadata['length']
            if abs(length_diff) > 0.5:
                warnings.warn('Calculated length = %2.2f and supplied length '
                              '= %2.2f differ more than 0.5 km' %
                              (calculated_length, self.metadata['length']))

        if kwargs.pop('calculate_length', True):
            self.metadata['length'] = calculated_length

        self.process = Processor(self)

    def __getattr__(self, item):
        """ Makes channels available via, e.g. `comlink.channel_1` """
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
        attr_list = (list(Comlink.__dict__.keys()) +
                     list(self.__dict__.keys()) +
                     list(self.channels.keys()))
        return attr_list

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
        new_cml.process = deepcopy(self.process, memo)
        return new_cml

    def get_coordinates(self):
        """ Return the coordinates of site_a and site_b

        Returns
        -------

        coords : namedtuple
            Named tuple of coordinates with the names 'lon_a', 'lon_b',
            'lat_a', 'lat_b'.

        """
        coords = Coords(lon_a=self.metadata['site_a_longitude'],
                        lon_b=self.metadata['site_b_longitude'],
                        lat_a=self.metadata['site_a_latitude'],
                        lat_b=self.metadata['site_b_latitude'])
        return coords

    def calc_length(self):
        """ Calculate and return length of CML km """

        coords = self.get_coordinates()
        d_km = distance((coords.lat_a, coords.lon_a),
                        (coords.lat_b, coords.lon_b))
        return d_km

    def get_length(self):
        """ Return length of CML in km """
        return self.metadata['length']

    def plot_map(self, tiles='OpenStreetMap', fol_map=None):
        """ Plot a dynamic map in Jupyter notebook using folium

        Parameters
        ----------

        tiles: str
            Name of tile to be used by folium, default is 'OpenStreetMap'
        fol_map: folium map instance
            An existing folium map instance can be passed here

        Returns
        -------

        fol_map : folium map object

        """

        coords = self.get_coordinates()

        if fol_map is None:
            fol_map = folium.Map(location=[(coords.lat_a + coords.lat_b) / 2,
                                           (coords.lon_a + coords.lon_b) / 2],
                                 tiles=tiles,
                                 zoom_start=11)
        fol_map.add_children(folium.PolyLine([(coords.lat_a, coords.lon_a),
                                              (coords.lat_b, coords.lon_b)]))
        return fol_map

    def plot_line(self, ax=None, *args, **kwargs):
        """ Plot the CML path using matplotlib

        `args` and `kwargs` will be passed to `matplotlib.pyplot.plot`

        Parameters
        ----------

        ax : matplotlib.axes
            Matplotlib axes handle, defaults to None. A figure is created in
            the default case

        Returns
        -------

        ax : matplotib.axes

        """

        if ax is None:
            fig, ax = plt.subplots()
        coords = self.get_coordinates()
        ax.plot([coords.lon_a, coords.lon_b],
                [coords.lat_a, coords.lat_b],
                *args, **kwargs)
        return ax

    def plot_data(self, columns=['rx', ], channels=None, ax=None):
        """ Plot time series of data from the different channels

        Linked subplots will be created for the different specified columns
        of the DataFrames of the different channels.

        Parameters
        ----------

        columns : list, optional
            List of DataFrame columns to plot for each channel.
            Defaults to ['rx', ]

        channels : list, optional
            List of channel names, i.e. the keys of the Comlink.channels
            dictionary, to specify which channel to plot. Defaults to None,
            which plots for all channels

        ax : matplotlib.axes, optional
            Axes handle, defaults to None, which plots into a new figure

        Returns
        -------

        ax : matplotlib.axes

        """
        if ax is None:
            fig, ax = plt.subplots(len(columns),
                                   1,
                                   figsize=(10, 1.5*len(columns) + 1),
                                   sharex=True)
        try:
            ax[0].get_alpha()
        except TypeError:
            ax = [ax, ]

        if channels is None:
            channels_to_plot = self.channels
        else:
            channels_to_plot = {ch_key: self.channels[ch_key]
                                for ch_key in channels}

        for ax_i, column in zip(ax, columns):
            for i, (name, cml_ch) in enumerate(channels_to_plot.items()):
                if column == 'wet':
                    ax_i.fill_between(
                        cml_ch.data[column].index,
                        i,
                        i + cml_ch.data[column].values,
                        alpha=0.9,
                        linewidth=0.0,
                        label=name)
                else:
                    ax_i.plot(
                        cml_ch.data[column].index,
                        cml_ch.data[column].values,
                        label=name)
            ax_i.set_ylabel(column)

        return ax

    def get_center_lon_lat(self):
        """ Calculate and return longitude and latitude of the CML path center

        Returns
        -------

        (center_lon, center_lat)

        """
        coords = self.get_coordinates()
        center_lon = (coords.lon_a + coords.lon_b) / 2
        center_lat = (coords.lat_a + coords.lat_b) / 2
        return center_lon, center_lat

    def append_data(self, cml, max_length=None, max_age=None):
        """ Append the data from the same CML stored in another Comlink object

        Parameters
        ----------
        cml
        max_length
        max_age

        Returns
        -------

        """

        for key in self.metadata.keys():
            if self.metadata[key] != cml.metadata[key]:
                raise ValueError('Comlink metadata `%s` is different'
                                 'for the two CMLs: %s vs. %s' %
                                 (key, self.metadata[key], cml.metadata[key]))

        for ch_name in self.channels.keys():
            self.channels[ch_name].append_data(
                cml_ch=cml.channels[ch_name],
                max_length=max_length,
                max_age=max_age)


def _channels_list_to_ordered_dict(channels):
    """ Helper function to parse a list of channels to a dict of channels

    The keys will be `channel_(i+1)`, where i is the index of the list of
    channels. These keys will be used to make the different channels
    available in the Comlink object via e.g. `comlink.channel_1`.

    Parameters
    ----------

    channels : list
        List of ComlinkChannel objects

    Returns
    -------

    channel_dict : dict

    """
    channel_dict = OrderedDict()
    for i, channel in enumerate(channels):
        channel_dict['channel_' + str(i+1)] = channel
    return channel_dict

