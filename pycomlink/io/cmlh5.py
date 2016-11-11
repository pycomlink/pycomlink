# ----------------------------------------------------------------------------
# Name: Read/Write/Helper functions for HDF5 based CML data format cmlH5
# Purpose:      
#
# Authors:      
#
# Created:      
# Copyright:    (c) Christian Chwala 2016
# Licence:      The MIT License
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import h5py

from warnings import warn

from pycomlink import Comlink, ComlinkChannel



CMLH5_VERSION = 0.2

cml_metadata_dict = {
    'cml_id': {'mandatory': True, 'type': str},
    'site_a_latitude': {'mandatory': True, 'type': float},
    'site_a_longitude': {'mandatory': True, 'type': float},
    'site_b_latitude': {'mandatory': True, 'type': float},
    'site_b_longitude': {'mandatory': True, 'type': float},
}

cml_ch_metadata_dict = {
    'frequency': {'mandatory': True, 'type': float,
                  'min': 0.1e9, 'max': 100e9},
    'polarization': {'mandatory': True, 'type': str,
                     'options': ['H', 'V', 'h', 'v']},
    'channel_id': {'mandatory': True, 'type': str},
    'atpc': {'mandatory': False, 'type': str,
             'options': ['on', 'off']}
}

cml_ch_data_names_dict = {
    'rx': {'mandatory': True,
           'quantity': 'Received signal level',
           'units': 'dBm'},
    'tx': {'mandatory': False,
           'quantity': 'Trasmitted signal level',
           'units': 'dBm'},
    'time': {'mandatory': True,
             'quantity': 'Timestamp',
             'units': 'seconds since 1970-01-01 00:00:00',
             'calendar': 'proleptic_gregorian'}
}


def write_to_cmlh5(cml_list, fn,
                   product_keys=None, product_names=None, product_units=None,
                   compression='gzip', compression_opts=4):
    """

    Parameters
    ----------

    cml_list:
    fn:
    product_keys:
    product_names:
    product_units:
    compression:
    compression_opts:

    """

    # Check and prepare `product_keys`, `product_names` and `product_units`
    if product_keys is not None:
        if type(product_keys) == str:
            product_keys = [product_keys]
            strings_are_supplied = True
        else:
            strings_are_supplied = False

        if product_names is None:
            product_names = product_keys
        else:
            if type(product_names) == str:
                if not strings_are_supplied:
                    raise AttributeError('`product_keys` was supplied as list,'
                                         ' so must be `product_names`')
                product_names = [product_names]

        if product_units is None:
            raise AttributeError('Units must be supplied for the products')
        else:
            if type(product_units) == str:
                if not strings_are_supplied:
                    raise AttributeError('`product_keys` was supplied as list,'
                                         ' so must be `product_units`')
                product_units = [product_units]

    # Write to file
    with h5py.File(fn, mode='w') as h5file:
        h5file.attrs['file_format'] = 'cmlH5'
        h5file.attrs['file_format_version'] = CMLH5_VERSION

        for i_cml, cml in enumerate(cml_list):
            # Create CML HDF5 group
            cml_g = h5file.create_group('cml_%d' % i_cml)
            # Write CML attributes
            _write_cml_attributes(cml_g, cml)

            # Write CML channels
            for i_channel, channel_id in enumerate(cml.channels.keys()):
                cml_ch = cml.channels[channel_id]
                chan_g = cml_g.create_group('channel_%d' % (i_channel + 1))
                _write_channel_attributes(chan_g, cml_ch)
                _write_channel_data(chan_g,
                                    cml_ch,
                                    compression,
                                    compression_opts)

            # Write CML derived products like rain rate for each CML
            if product_keys is not None:
                for i_prod, (product_key, product_name, product_unit) in \
                        enumerate(zip(
                            product_keys,
                            product_names,
                            product_units)):
                    prod_g = cml_g.create_group('product_%d' % i_prod)
                    _write_product(prod_g, cml, product_key, product_name,
                                   product_unit, compression, compression_opts)


def _write_cml_attributes(cml_g, cml):
    """
    cml_g : HDF5 group at CML level
    cml : pycomlink.Comlink object

    """

    for attr_name, attr_options in cml_metadata_dict.iteritems():
        cml_g.attrs[attr_name] = cml.metadata[attr_name]


def _write_channel_attributes(chan_g, cml_ch):
    """
    chan_g : HDF5 group at CML-channel level
    cml : pycomlink.Comlink object

    """

    for attr_name, attr_options in cml_ch_metadata_dict.iteritems():
        attr_value = cml_ch.metadata[attr_name]
        if attr_value is None:
            if attr_options['mandatory']:
                warn('\n The mandatory attribute `%s` is `None`'
                     '\n Using fill_value instead' % attr_name)
            chan_g.attrs[attr_name] = _missing_attribute(attr_options['type'])
        else:
            chan_g.attrs[attr_name] = attr_value


def _write_channel_data(chan_g, cml_ch, compression, compression_opts):
    """

    @param chan_g:
    @param cml_ch:
    @param compression:
    @param compression_opts:
    @return:
    """

    # write variables
    for name, attrs in cml_ch_data_names_dict.iteritems():
        if name == 'time':
            t_vec = cml_ch._df.index.astype('int64') / 1e9
            chan_g.create_dataset(name, data=t_vec,
                                  compression=compression,
                                  compression_opts=compression_opts)
        else:
            chan_g.create_dataset(name, data=cml_ch._df[name].values,
                                  compression=compression,
                                  compression_opts=compression_opts)

        for attr_name, attr_value in attrs.iteritems():
            chan_g[name].attrs[attr_name] = attr_value

    # Create time scale
    chan_g['time'].dims.create_scale(chan_g['time'], 'time')

    # Link all other datasets to the time scale
    for name in cml_ch_data_names_dict.keys():
        if not name == 'time':
            chan_g[name].dims[0].attach_scale(chan_g['time'])


def _missing_attribute(attr_type):
    if attr_type == float:
        fill_value = np.nan
    elif attr_type == int:
        fill_value = -9999
    elif attr_type == str:
        fill_value = 'NA'
    else:
        raise AttributeError('Could not infer `missing_value` for '
                             '`attr_type` %s' % attr_type)
    return fill_value


#!!!!!!!!!!!!!!!!!!!!!!#
# OBSOLETE STUFF BELOW #
#!!!!!!!!!!!!!!!!!!!!!!#


def _get_cml_channel_attrs(cml, channel_id):
    ch_metadata = cml.tx_rx_pairs[channel_id]
    attrs = {'frequency': ch_metadata['f_GHz'],
             'polarisation': ch_metadata['pol'],
             'ID': ch_metadata['name'],
             'ATPC': 'Not sure...',
             'sampling_type': 'instantaneous',
             'temporal_resolution': '1min',
             'TX_quantization': 1.0,
             'RX_quantization': 0.3}
    try:
        attrs['TX_site'] = ch_metadata['tx_site']
        attrs['RX_site'] = ch_metadata['rx_site']
    except KeyError:
        pass
    return attrs


def _get_cml_channel_data(cml, channel_id):
    # Get UNIX time form pandas.DatetimeIndex (which is UNIX time in ns)
    t_vec = cml.data.index.astype('int64') / 1e9

    tx_column_name = cml.tx_rx_pairs[channel_id]['tx']
    rx_column_name = cml.tx_rx_pairs[channel_id]['rx']

    tx_vec = cml.data[tx_column_name].values
    rx_vec = cml.data[rx_column_name].values

    return t_vec, tx_vec, rx_vec


def _write_product(prod_g, cml, product_key, product_name, product_unit,
                   compression, compression_opts):
    # Get UNIX time form pandas.DatetimeIndex (which is UNIX time in ns)
    t_vec = cml.data.index.astype('int64') / 1e9

    product_vec = cml.data[product_key].values

    prod_g.create_dataset(product_name, data=product_vec,
                          compression=compression,
                          compression_opts=compression_opts)
    prod_g[product_name].attrs['units'] = product_unit

    # write time dimension
    prod_g.create_dataset('time', data=t_vec,
                          compression=compression,
                          compression_opts=compression_opts)
    prod_g['time'].attrs['units'] = 'POSIX time UTC'
    prod_g['time'].attrs['calendar'] = 'proleptic_gregorian'
    prod_g[product_name].dims.create_scale(prod_g['time'], 'time')
    prod_g[product_name].dims[0].attach_scale(prod_g['time'])


def _read_cml_metadata(cml_g):
    metadata = {}
    metadata['link_id'] = cml_g.attrs['id']
    metadata['length_km'] = cml_g.attrs['length']
    metadata['site_A'] = {'lat': cml_g.attrs['site_a_latitude'],
                          'lon': cml_g.attrs['site_a_longitude']}
    metadata['site_B'] = {'lat': cml_g.attrs['site_b_latitude'],
                          'lon': cml_g.attrs['site_b_longitude']}
    return metadata


def _read_channels_metadata(cml_g):
    tx_rx_pairs = {}
    for chan_g_name, chan_g in cml_g.items():
        tx_rx_pairs[chan_g_name] = {'name': chan_g_name,
                                    'tx': 'tx_' + chan_g_name,
                                    'rx': 'rx_' + chan_g_name,
                                    'f_GHz': chan_g.attrs['frequency'],
                                    'pol': chan_g.attrs['polarisation']}
    return tx_rx_pairs


def _read_channels_data(cml_g):
    data_dict = {}
    for chan_g_name, chan_g in cml_g.items():
        data_dict['rx_' + chan_g_name] = chan_g['RX'][:]
        data_dict['tx_' + chan_g_name] = chan_g['TX'][:]
    data = pd.DataFrame(data=data_dict,
                        index=pd.DatetimeIndex(chan_g['time'][:] * 1e9,
                                               tz='UTC'))

    return data


def _read_one_cml(cml_g):
    metadata = _read_cml_metadata(cml_g)
    tx_rx_pairs = _read_channels_metadata(cml_g)
    df_data = _read_channels_data(cml_g)
    cml = Comlink(data=df_data,
                  tx_rx_pairs=tx_rx_pairs,
                  metadata=metadata)
    return cml


def read_from_cmlh5(fn):
    h5_reader = h5py.File(fn, mode='r')
    cml_list = []
    for cml_g_name in h5_reader['/']:
        cml_g = h5_reader['/' + cml_g_name]
        cml = _read_one_cml(cml_g)
        cml_list.append(cml)
    print '%d CMLs read in' % len(cml_list)
    return cml_list
