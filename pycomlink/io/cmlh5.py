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

from __future__ import print_function
from __future__ import division

from builtins import zip
import numpy as np
import pandas as pd
import h5py

from copy import deepcopy
from warnings import warn
from collections import OrderedDict

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

cml_ch_time_dict = {
    'mandatory': True,
    'quantity': 'Timestamp',
    'units': 'seconds since 1970-01-01 00:00:00',
    'calendar': 'proleptic_gregorian'
}

cml_ch_data_names_dict_tx_rx = {
    'rx': {'mandatory': False,
           'quantity': 'Received signal level',
           'units': 'dBm'},
    'tx': {'mandatory': False,
           'quantity': 'Trasmitted signal level',
           'units': 'dBm'},
    'time': cml_ch_time_dict
}

cml_ch_data_names_dict_tx_rx_min_max = {
    'rx_min': {'mandatory': False,
               'quantity': 'Minimum received signal level',
               'units': 'dBm'},
    'rx_max': {'mandatory': False,
               'quantity': 'Maximum received signal level',
               'units': 'dBm'},
    'tx_min': {'mandatory': False,
               'quantity': 'Minimum trasmitted signal level',
               'units': 'dBm'},
    'tx_max': {'mandatory': False,
               'quantity': 'Maximum trasmitted signal level',
               'units': 'dBm'},
    'time': cml_ch_time_dict
}


#########################
# Functions for writing #
#########################

def write_to_cmlh5(cml_list, fn,
                   t_start=None, t_stop=None,
                   column_names_to_write=None,
                   split_to_multiple_files=False, splitting_period='D',
                   append_date_str_to_fn='_%Y%m%d',
                   write_all_data=False,
                   product_keys=None, product_names=None, product_units=None,
                   compression='gzip', compression_opts=4):
    """

    Parameters
    ----------
    cml_list
    fn
    t_start
    t_stop
    column_names_to_write
    split_to_multiple_files
    splitting_period
    append_date_str_to_fn
    write_all_data
    product_keys
    product_names
    product_units
    compression
    compression_opts

    Returns
    -------

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

    if (t_start is None) and (t_stop is None):
            t_start, t_stop = (
                _get_first_and_last_timestamp_in_cml_list(cml_list))

    if split_to_multiple_files:
        t_in_file_start_list = pd.date_range(
            start=t_start,
            end=t_stop,
            freq=splitting_period,
            normalize=True)
        t_in_file_stop_list = pd.date_range(
            start=t_start + pd.Timedelta(1, splitting_period),
            end=t_stop + pd.Timedelta(1, splitting_period),
            freq=splitting_period,
            normalize=True)
        include_t_stop_in_file = False
    else:
        t_in_file_start_list = [t_start, ]
        t_in_file_stop_list = [t_stop, ]
        include_t_stop_in_file = True

    # Write to file(s)
    for i, (t_in_file_start, t_in_file_stop) in \
            enumerate(zip(t_in_file_start_list, t_in_file_stop_list)):
        if t_start > t_in_file_start:
            t_in_file_start = t_start
        if t_stop < t_in_file_stop:
            t_in_file_stop = t_stop
            include_t_stop_in_file = True

        if split_to_multiple_files:
            try:
                fn_body, fn_ending = fn.split('.')
            except:
                raise ValueError('file name must contain a `.`, '
                                 'e.g. `my_cml_file.h5`')
            if append_date_str_to_fn:
                fn_i = (fn_body +
                        t_in_file_start.strftime(append_date_str_to_fn) +
                        '.' + fn_ending)
            else:
                fn_i = '%s_%d.%s' % (fn_body, i, fn_ending)
        else:
            fn_i = fn

        with h5py.File(fn_i, mode='w') as h5file:
            h5file.attrs['file_format'] = 'cmlH5'
            h5file.attrs['file_format_version'] = CMLH5_VERSION
            h5file.attrs['time_coverage_start'] = t_in_file_start.strftime(
                '%Y-%m-%dT%H:%M:%SZ')
            h5file.attrs['time_coverage_stop'] = t_in_file_stop.strftime(
                '%Y-%m-%dT%H:%M:%SZ')

            for i_cml, cml in enumerate(cml_list):
                # Create CML HDF5 group
                cml_g = h5file.create_group(cml.metadata['cml_id'])
                # Write CML attributes
                _write_cml_attributes(cml_g, cml)

                # Write CML channels
                for i_channel, channel_id in enumerate(cml.channels.keys()):
                    cml_ch = cml.channels[channel_id]
                    chan_g = cml_g.create_group('channel_%d' % (i_channel + 1))
                    _write_channel_attributes(chan_g, cml_ch)
                    _write_channel_data(
                        chan_g=chan_g,
                        cml_ch=cml_ch,
                        t_start=t_in_file_start,
                        t_stop=t_in_file_stop,
                        column_names_to_write=column_names_to_write,
                        include_t_stop=include_t_stop_in_file,
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
                        _write_product(prod_g, cml, product_key,
                                       product_name, product_unit,
                                       compression, compression_opts)


def _get_first_and_last_timestamp_in_cml_list(cml_list):
    """

    Parameters
    ----------
    cml_list

    Returns
    -------

    """
    t_min = (
        min([min([cml_ch.data.index.min()
                  for cml_ch in cml.channels.values()])
             for cml in cml_list]))
    t_max = (
        max([max([cml_ch.data.index.max()
                  for cml_ch in cml.channels.values()])
             for cml in cml_list]))
    return t_min, t_max


def _write_cml_attributes(cml_g, cml):
    """
    cml_g : HDF5 group at CML level
    cml : pycomlink.Comlink object

    """

    for attr_name, attr_options in cml_metadata_dict.items():
        cml_g.attrs[attr_name] = cml.metadata[attr_name]


def _write_channel_attributes(chan_g, cml_ch):
    """
    chan_g : HDF5 group at CML-channel level
    cml : pycomlink.Comlink object

    """

    for attr_name, attr_options in cml_ch_metadata_dict.items():
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
                        t_start,
                        t_stop,
                        column_names_to_write,
                        compression,
                        compression_opts,
                        include_t_stop=True,
                        write_all_data=False):
    """

    Parameters
    ----------
    chan_g
    cml_ch
    t_start
    t_stop
    column_names_to_write
    compression
    compression_opts
    include_t_stop
    write_all_data

    Returns
    -------

    """

    # Build a dictionary with the info for the standard data that is to be
    # written to a cmlh5 file, i.e. `tx` and `rx`, or `rx_min`, `rx_max`,...
    _cml_ch_data_names_dict = {'time': cml_ch_time_dict}
    for column_name in cml_ch.data.columns:
        try:
            _cml_ch_data_names_dict[column_name] = (
                cml_ch_data_names_dict_tx_rx[column_name])
        except KeyError:
            pass
        try:
            _cml_ch_data_names_dict[column_name] = (
                cml_ch_data_names_dict_tx_rx_min_max[column_name])
        except KeyError:
            pass

    if (write_all_data is False) and (column_names_to_write is None):
        pass
    # If desired, add the rest of the columns, but without specific info
    # on the data like units or a descriptive name
    elif (write_all_data is True) and (column_names_to_write is None):
        for column_name in cml_ch.data.columns:
            if column_name not in list(_cml_ch_data_names_dict.keys()):
                _cml_ch_data_names_dict[column_name] = {}
    # Or add additional columns according to the info passed as argument
    elif (write_all_data is False) and (column_names_to_write is not None):
        if isinstance(column_names_to_write, dict):
            for key, value in column_names_to_write.iteritems():
                _cml_ch_data_names_dict[key] = value
        elif isinstance(column_names_to_write, list):
            for column_name in column_names_to_write:
                _cml_ch_data_names_dict[column_name] = {}
        else:
            raise AttributeError('`column_names_to_write` must be either a list'
                                 'or a dict')
    else:
        raise AttributeError('`write_all_data cannot be True if '
                             '`columns_names_to_write` is provided '
                             'and not None')

    # Get the time index in UTC
    ts_t = cml_ch.data.index.tz_convert('UTC')

    if include_t_stop:
        t_slice_ix = (ts_t >= t_start) & (ts_t <= t_stop)
    else:
        t_slice_ix = (ts_t >= t_start) & (ts_t < t_stop)

    # write variables
    for name, attrs in _cml_ch_data_names_dict.items():
        if name == 'time':
            # Transform the pandas (np.datetime64) which is in ns to seconds
            t_vec = ts_t.astype('int64') / 1e9
            chan_g.create_dataset(name,
                                  data=t_vec[t_slice_ix],
                                  compression=compression,
                                  compression_opts=compression_opts)
        elif name in cml_ch.data.columns:
            chan_g.create_dataset(name,
                                  data=cml_ch.data[name].values[t_slice_ix],
                                  compression=compression,
                                  compression_opts=compression_opts)
        else:
            print('`%s` not found in ComlinkChannel.data.columns' % name)

        for attr_name, attr_value in attrs.items():
            chan_g[name].attrs[attr_name] = attr_value

    # Create time scale
    chan_g['time'].dims.create_scale(chan_g['time'], 'time')

    # Link all other datasets to the time scale
    for name in list(_cml_ch_data_names_dict.keys()):
        if name in cml_ch.data.columns:
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
    t_vec = ts_t.astype('int64'), 1e9
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


def read_from_cmlh5(fn,
                    cml_id_list=None,
                    t_start=None,
                    t_stop=None,
                    column_names_to_read=None,
                    read_all_data=False):
    """

    Parameters
    ----------
    fn
    cml_id_list
    t_start
    t_stop
    column_names_to_read
    read_all_data

    Returns
    -------

    """
    h5_reader = h5py.File(fn, mode='r')
    cml_list = []
    for cml_g_name in h5_reader['/']:
        cml_g = h5_reader['/' + cml_g_name]
        cml = _read_one_cml(cml_g=cml_g,
                            cml_id_list=cml_id_list,
                            t_start=t_start,
                            t_stop=t_stop,
                            column_names_to_read=column_names_to_read,
                            read_all_data=read_all_data)
        if cml is not None:
            cml_list.append(cml)
    print('%d CMLs read in' % len(cml_list))
    return cml_list


def read_from_multiple_cmlh5(fn_list,
                             cml_id_list=None,
                             t_start=None,
                             t_stop=None,
                             sort_fn_list=True):
    """

    Parameters
    ----------
    fn_list
    cml_id_list
    t_start
    t_stop
    sort_fn_list

    Returns
    -------

    """

    if sort_fn_list:
        fn_list.sort()

    fn_list_selected = []

    # Find the files where data is stored for the specified period
    if (t_start is not None) and (t_stop is not None):
        # loop through all files to find their temporal coverage

        t_start = pd.to_datetime(t_start)
        t_stop = pd.to_datetime(t_stop)

        for fn in fn_list:
            with h5py.File(fn, mode='r') as h5_reader:
                # update fn_list so that only necessary files are contained
                time_coverage_start = pd.to_datetime(
                    h5_reader.attrs['time_coverage_start'])
                time_coverage_stop = pd.to_datetime(
                    h5_reader.attrs['time_coverage_stop'])
                if ((time_coverage_start < t_stop) and
                        (time_coverage_stop > t_start)):
                    fn_list_selected.append(fn)
    # If no start and stop data has been provided, just use fn_list
    elif (t_start is None) and (t_stop is None):
        fn_list_selected = fn_list
    else:
        raise ValueError('`t_start` and `t_stop` must both be either `None` '
                         'or some timestamp information.')

    # Loop over cmlh5 files and read them in
    cml_lists = []
    for fn in fn_list_selected:
        cml_lists.append(read_from_cmlh5(fn=fn,
                                         cml_id_list=cml_id_list,
                                         t_start=t_start,
                                         t_stop=t_stop))

    # Concat data for the Comlink objects
    cml_dict = OrderedDict()
    for cml_list in cml_lists:
        for cml in cml_list:
            cml_id = cml.metadata['cml_id']
            if cml_id in list(cml_dict.keys()):
                cml_dict[cml_id].append_data(cml)
            else:
                cml_dict[cml_id] = cml

    return list(cml_dict.values())


def _read_one_cml(cml_g,
                  cml_id_list=None,
                  t_start=None,
                  t_stop=None,
                  column_names_to_read=None,
                  read_all_data=False):
    """

    Parameters
    ----------
    cml_g
    cml_id_list
    t_start
    t_stop
    column_names_to_read
    read_all_data

    Returns
    -------

    """
    metadata = _read_cml_metadata(cml_g)

    if cml_id_list is not None:
        if metadata['cml_id'] not in cml_id_list:
            return None

    cml_ch_list = []
    for cml_ch_name, cml_ch_g in list(cml_g.items()):
        if 'channel_' in cml_ch_name:
            cml_ch_list.append(
                _read_cml_channel(
                    cml_ch_g=cml_ch_g,
                    t_start=t_start,
                    t_stop=t_stop,
                    column_names_to_read=column_names_to_read,
                    read_all_data=read_all_data))

    # TODO: Handle `auxiliary_N` and `product_N` cml_g-subgroups

    return Comlink(channels=cml_ch_list, metadata=metadata)


def _read_cml_metadata(cml_g):
    """

    @param cml_g:
    @return:
    """

    metadata = {}
    for attr_name, attr_options in cml_metadata_dict.items():
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
    for attr_name, attr_options in cml_ch_metadata_dict.items():
        value = cml_ch_g.attrs[attr_name]
        # This is necessary because of this h5py issue
        # https://github.com/h5py/h5py/issues/379
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        # TODO: Handle NaN values
        metadata[attr_name] = value
    return metadata


def _read_cml_channel_data(cml_ch_g,
                           t_start=None,
                           t_stop=None,
                           column_names_to_read=None,
                           read_all_data=None):
    """

    Parameters
    ----------
    cml_ch_g
    t_start
    t_stop
    column_names_to_read
    read_all_data

    Returns
    -------

    """

    if (read_all_data is False) and (column_names_to_read is None):
        _cml_ch_data_names_list = list(set(
            list(cml_ch_data_names_dict_tx_rx.keys()) +
            list(cml_ch_data_names_dict_tx_rx_min_max.keys())))
    elif (read_all_data is True) and (column_names_to_read is None):
        _cml_ch_data_names_list = list(cml_ch_g.keys())
    elif (read_all_data is False) and (column_names_to_read is not None):
        if isinstance(column_names_to_read, list):
            _cml_ch_data_names_list = column_names_to_read
        else:
            raise AttributeError('`column_names_to_write` must be either a list'
                                 'or a dict')
    else:
        raise AttributeError('`read_all_data cannot be True if '
                             '`columns_names_to_read` is provided '
                             'and not None')

    data_dict = {}
    for name in _cml_ch_data_names_list:
        try:
            data_dict[name] = cml_ch_g[name]
        except KeyError:
            pass
    if len(list(data_dict.keys())) == 0:
        print('Warning: No dataset matching the column names %s '
              'found in cml_ch_g' % str(_cml_ch_data_names_list))

    # Time is stored in seconds since epoch and is represented in pandas by
    # np.datetime64 in nanoseconds
    t = (data_dict.pop('time')[:] * 1e9).astype('datetime64[ns]')

    df = pd.DataFrame(index=t, data=data_dict)

    # Time must always be saved as UTC in cmlH5
    df.index = df.index.tz_localize('UTC')

    return df


def _read_cml_channel(cml_ch_g,
                      t_start=None,
                      t_stop=None,
                      column_names_to_read=None,
                      read_all_data=None):
    """

    Parameters
    ----------
    cml_ch_g
    t_start
    t_stop
    column_names_to_read
    read_all_data

    Returns
    -------

    """
    metadata = _read_cml_channel_metadata(cml_ch_g=cml_ch_g)
    df = _read_cml_channel_data(
        cml_ch_g=cml_ch_g,
        t_start=t_start,
        t_stop=t_stop,
        column_names_to_read=column_names_to_read,
        read_all_data=read_all_data)
    return ComlinkChannel(data=df, metadata=metadata)




