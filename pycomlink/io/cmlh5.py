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

from copy import deepcopy
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


#########################
# Functions for writing #
#########################

def write_to_cmlh5(cml_list, fn,
                   write_all_data=False,
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
                _write_channel_data(chan_g=chan_g,
                                    cml_ch=cml_ch,
                                    compression=compression,
                                    compression_opts=compression_opts,
                                    write_all_data=write_all_data)

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


def _write_channel_data(chan_g,
                        cml_ch,
                        compression,
                        compression_opts,
                        write_all_data=False):
    """

    @param chan_g:
    @param cml_ch:
    @param compression:
    @param compression_opts:
    @param write_all_data:
    @return:
    """

    if write_all_data:
        # If all channel data shall be written, build a dict with the
        # columns names and additional metadata. Start with the dict
        # with the default channel data definition
        _cml_ch_data_names_dict = deepcopy(cml_ch_data_names_dict)
        # Attach all other column names of the channel's DataFrame
        for column_name in cml_ch.data.columns:
            if not column_name in _cml_ch_data_names_dict.keys():
                _cml_ch_data_names_dict[column_name] = {}
    else:
        # If only standard data shall be written use the dict defined on top
        # of this file
        _cml_ch_data_names_dict = cml_ch_data_names_dict

    # write variables
    for name, attrs in _cml_ch_data_names_dict.iteritems():
        if name == 'time':
            # Get the time index in UTC
            ts_t = cml_ch.data.index.tz_convert('UTC')
            # Transform the pandas (np.datetime64) which is in ns to seconds
            t_vec = ts_t.astype('int64') / 1e9
            chan_g.create_dataset(name, data=t_vec,
                                  compression=compression,
                                  compression_opts=compression_opts)
        else:
            chan_g.create_dataset(name, data=cml_ch.data[name].values,
                                  compression=compression,
                                  compression_opts=compression_opts)

        for attr_name, attr_value in attrs.iteritems():
            chan_g[name].attrs[attr_name] = attr_value

    # Create time scale
    chan_g['time'].dims.create_scale(chan_g['time'], 'time')

    # Link all other datasets to the time scale
    for name in _cml_ch_data_names_dict.keys():
        if not name == 'time':
            chan_g[name].dims[0].attach_scale(chan_g['time'])


def _write_product(prod_g, cml,
                   product_key, product_name, product_unit,
                   compression, compression_opts):
    """

    @param prod_g:
    @param cml:
    @param product_key:
    @param product_name:
    @param product_unit:
    @param compression:
    @param compression_opts:
    @return:
    """

    # TODO: Make it possible to save product from different channels
    # Choose the first channel (since there is no other solution now for how
    # to deal with the "products" for each channel of one CML
    cml_ch = cml.channel_1

    ts_t = cml_ch.data.index.tz_convert('UTC')
    # Transform the pandas (np.datetime64) which is in ns to seconds
    t_vec = ts_t.astype('int64') / 1e9
    prod_g.create_dataset('time',
                          data=t_vec,
                          compression=compression,
                          compression_opts=compression_opts)

    prod_g.create_dataset(product_name,
                          data=cml_ch.data[product_key].values,
                          compression=compression,
                          compression_opts=compression_opts)

    # Create time scale
    prod_g['time'].dims.create_scale(prod_g['time'], 'time')
    
    prod_g['time'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
    prod_g['time'].attrs['calendar'] = 'proleptic_gregorian'
    prod_g['time'].attrs['quantity'] = 'Timestamp'

    prod_g[product_name].attrs['units'] = product_unit
    prod_g[product_name].attrs['quantity'] = product_name

    # Link all other datasets to the time scale
    if not product_name == 'time':
            prod_g[product_name].dims[0].attach_scale(prod_g['time'])

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


#########################
# Functions for reading #
#########################


def read_from_cmlh5(fn):
    """

    @param fn:
    @return:
    """
    h5_reader = h5py.File(fn, mode='r')
    cml_list = []
    for cml_g_name in h5_reader['/']:
        cml_g = h5_reader['/' + cml_g_name]
        cml = _read_one_cml(cml_g)
        cml_list.append(cml)
    print '%d CMLs read in' % len(cml_list)
    return cml_list


def _read_one_cml(cml_g):
    """

    @param cml_g:
    @return:
    """
    metadata = _read_cml_metadata(cml_g)

    cml_ch_list = []
    for cml_ch_name, cml_ch_g in cml_g.items():
        if 'channel_' in cml_ch_name:
            cml_ch_list.append(_read_cml_channel(cml_ch_g))

    # TODO: Handle `auxiliary_N` and `product_N` cml_g-subgroups

    return Comlink(channels=cml_ch_list, metadata=metadata)


def _read_cml_metadata(cml_g):
    """

    @param cml_g:
    @return:
    """

    metadata = {}
    for attr_name, attr_options in cml_metadata_dict.iteritems():
        value = cml_g.attrs[attr_name]
        # TODO: Handle NaN values
        metadata[attr_name] = value
    return metadata


def _read_cml_channel_metadata(cml_ch_g):
    """

    @param cml_ch_g:
    @return:
    """

    metadata = {}
    for attr_name, attr_options in cml_ch_metadata_dict.iteritems():
        value = cml_ch_g.attrs[attr_name]
        # TODO: Handle NaN values
        metadata[attr_name] = value
    return metadata


def _read_cml_channel_data(cml_ch_g):
    """

    @param cml_ch_g:
    @return:

    """

    data_dict = {}
    for name, attrs in cml_ch_data_names_dict.iteritems():
        data_dict[name] = cml_ch_g[name]

    # Time is stored in seconds since epoch and is represented in pandas by
    # np.datetime64 in nanoseconds
    t = (data_dict.pop('time')[:] * 1e9).astype('datetime64[ns]')

    df = pd.DataFrame(index=t, data=data_dict)

    # Time must always be saved as UTC in cmlH5
    df.index = df.index.tz_localize('UTC')

    return df


def _read_cml_channel(cml_ch_g):
    """

    @param cml_ch_g:
    @return:
    """
    metadata = _read_cml_channel_metadata(cml_ch_g)
    df = _read_cml_channel_data(cml_ch_g)
    return ComlinkChannel(data=df, metadata=metadata)




