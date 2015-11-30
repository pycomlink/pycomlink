#----------------------------------------------------------------------------
# Name:         io
# Purpose:      Initialization of comlink object
#               Input and output functions for commercial MW link (CML) data
#
# Authors:      Christian Chwala, Felix Keis
#
# Created:      01.12.2014
# Copyright:    (c) Christian Chwala 2014
# Licence:      The MIT License
#----------------------------------------------------------------------------

import os
import scipy.io
import numpy as np
import pandas as pd

import h5py

from collections import namedtuple

from comlink import Comlink
from comlinkset import ComlinkSet

from math import radians, cos, sin, asin, sqrt


####################################################################
# Read/Write/Helper functions for HDF5 based CML data format CMLh5 #
####################################################################

def _get_cml_attrs(cml):
    attrs = {'id': cml.metadata['link_id'],
             'site_a_latitude': cml.metadata['site_A']['lat'],
             'site_a_longitude': cml.metadata['site_A']['lon'],
             'site_b_latitude': cml.metadata['site_B']['lat'],
             'site_b_longitude': cml.metadata['site_B']['lon'],
             'length': _haversine(cml.metadata['site_A']['lon'], cml.metadata['site_A']['lat'],
                                 cml.metadata['site_B']['lon'], cml.metadata['site_B']['lat']),
             'system_manufacturer': 'Ericsson',
             'system_model': 'MINI LINK Traffic Node'}
    return attrs


def _get_cml_channel_ids(cml):
    return cml.tx_rx_pairs.keys()


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

def _write_cml_attributes(cml_g, cml):
    '''
    cml_g : HDF5 group at CML level
    cml : pycomlink.Comlink object

    '''

    cml_attrs = _get_cml_attrs(cml)
    for key in cml_attrs.iterkeys():
        attr = cml_attrs[key]
        if attr == None:
            cml_g.attrs[key] = np.nan
        else:
            cml_g.attrs[key] = attr


def _write_channel_attributes(chan_g, cml, channel_id):
    chan_attrs = _get_cml_channel_attrs(cml, channel_id)
    for key in chan_attrs.iterkeys():
        attr = chan_attrs[key]
        if attr == None:
            chan_g.attrs[key] = np.nan
        else:
            chan_g.attrs[key] = attr


def _write_channel_data(chan_g, cml, channel_id, compression, compression_opts):
    t_vec, tx_vec, rx_vec = _get_cml_channel_data(cml, channel_id)

    # write variables
    chan_g.create_dataset('RX', data=rx_vec,
                          compression=compression,
                          compression_opts=compression_opts)
    chan_g['RX'].attrs['units'] = 'dBm'
    chan_g.create_dataset('TX', data=tx_vec,
                          compression=compression,
                          compression_opts=compression_opts)
    chan_g['TX'].attrs['units'] = 'dBm'

    # write time dimension
    chan_g.create_dataset('time', data=t_vec,
                          compression=compression,
                          compression_opts=compression_opts)
    chan_g['time'].attrs['units'] = 'POSIX time UTC'
    chan_g['time'].attrs['calendar'] = 'proleptic_gregorian'
    chan_g['RX'].dims.create_scale(chan_g['time'], 'time')
    chan_g['RX'].dims[0].attach_scale(chan_g['time'])
    chan_g['TX'].dims[0].attach_scale(chan_g['time'])


def write_to_cmlh5(cml_list, fn, compression='gzip', compression_opts=4):
    with h5py.File(fn, mode='w') as h5file:
        h5file.attrs['cmlH5_version'] = '0.2'

        for i_cml, cml in enumerate(cml_list):
            # Create CML HDF5 group
            cml_g = h5file.create_group('cml_%d' % i_cml)
            # Write CML attributes
            _write_cml_attributes(cml_g, cml)

            # Get and write CML channels
            channel_ids = _get_cml_channel_ids(cml)
            for i_channel, channel_id in enumerate(channel_ids):
                chan_g = cml_g.create_group('channel_%d' % (i_channel + 1))
                _write_channel_attributes(chan_g, cml, channel_id)
                _write_channel_data(chan_g, cml, channel_id, compression, compression_opts)


def _read_cml_metadata(cml_g):
    metadata = {}
    metadata['link_id'] = cml_g.attrs['id']
    metadata['length_km'] = cml_g.attrs['length']
    metadata['site_A'] = {'lat':cml_g.attrs['site_a_latitude'],
                          'lon': cml_g.attrs['site_a_longitude']}
    metadata['site_B'] = {'lat':cml_g.attrs['site_b_latitude'],
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
    data = pd.DataFrame(data=data_dict, index=pd.DatetimeIndex(chan_g['time'][:] * 1e9, tz='UTC'))

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


#########################################
# Obsolete old HDF5 read/write function #
#########################################

def _old_write_hdf5(fn, cml, cml_id=None):
    """ Write Comlink or Comlink list to HDF5
    
    Parameters
    ----------    
    fn : str
        Absolute filename
    cml : Comlink or list of Comlink objects
        Comlink object or list of these
    cml_id : str, optional
        Name of the CML which will be used to identify it in the HDF storer.
        Default is to use cml.metadata.link_id
        
    """
    
    store = pd.HDFStore(fn)

    if isinstance(cml, Comlink):
        cmls = [cml,]
    elif type(cml) == list:
        cmls = cml
    else:
        raise TypeError('Type of variable `cml` not understood')
        
    for cml in cmls:
        if cml_id == None:
            # Get cml_if from metadata
            cml_id = cml.metadata['link_id']
        elif cml_id != None and type(cml) == list:
            # For a list of cmls no fixed cml_id should be used
            raise TypeError('Do not supply a link_id when passing a list of cmls')
        else:
            # Take cml_id from function arguments
            pass
        
        # Store the TX and RX data
        store[cml_id] = cml.data
        # Store metadata
        store.get_storer(cml_id).attrs.metadata = cml.metadata
        # Store tx_rx_pairs
        store.get_storer(cml_id).attrs.tx_rx_pairs = cml.tx_rx_pairs
    
    store.close()
    
def _old_read_hdf5(fn, force_list_return=False):
    """Read Comlink or list of Comlinks from HDF5
    
    Parameters
    ----------
    fn : str
        Absolute filename to a pycomlink-HDF5 file
    force_list_return : bool, optional
        Set this to True if you always want a list in return even if there
        is only one cml in the HDF5 file. Default is False
        
    Returns
    -------    
    cml : Comlink
        If only one data set was found, the Comlink object that was stored in
        the HDF5 file is returned (except if forec_list_return == True).
    cmls : List of Comlink objects
        If several data sets were found, a list of all Comlink objects
        is returned
        
    """
    
    store = pd.HDFStore(fn, 'r')
    
    # Get all cml_id value (wich must be used as keys on pycomlink HDF5 files)
    cml_id_list = store.keys()

    cml_list = []
    
    for cml_id in cml_id_list:    
        # Get data from HDF store
        data = store[cml_id]
        metadata = store.get_storer(cml_id).attrs.metadata
        tx_rx_pairs = store.get_storer(cml_id).attrs.tx_rx_pairs
        # Generate Comlink object from it and append to cml list
        cml_list.append(Comlink(data=data, 
                                tx_rx_pairs=tx_rx_pairs,
                                metadata=metadata))
    
    store.close()

    # If there is only one cml element in the list    
    if len(cml_list) == 1 and force_list_return == False:
        return cml_list[0]
    else:
        return ComlinkSet(cml_list)

def read_PROCEMA_raw_data(fn):
    """ Read in PROCEMA data for one MW link stored as CSV or MATLAB binary
    
    Parameters
    ----------   
    fn : str
        Absolute filename. File can be a PROCEMA MATLAB file or a PROCEMA
        CSV file as exported by the old PROCEMA database
        
    Returns
    -------  
    cml : Comlink Object
    
    """
 
    fname, fext = os.path.splitext(fn)
    
    if fext=='.mat':
        dat = scipy.io.loadmat(fn)
        param = namedtuple('param', 'name mV_clear_sky RSL_clear_sky dB_per_V')
        
        param.name = dat['name'][0]
        param.mV_clear_sky = dat['equivalent_voltage_at_adc'][0][0]
        param.RSL_clear_sky = dat['received_power_clear_sky'][0][0]
        param.dB_per_V = dat['attenuation_per_volt'][0][0]
              
        rx = _mV2RSL(dat['values'][0],
                             param.dB_per_V,
                             param.RSL_clear_sky,
                             param.mV_clear_sky)
        index = _matlab_datenum_2_datetime(dat['time'][0])
                             
        data = pd.DataFrame({'rx' : pd.Series(rx, index=index)})
        # Add column with constant TX power
        data['tx'] = float(dat['ouput_power'][0][0])
        
        if dat['polarization'][0][0] == 0:
            pol = 'H'
        elif dat['polarization'][0][0] == 1:
            pol = 'V'
        else:
            raise ValueError('Polarization value in raw data should be 0 or 1')
        
        tx_rx_pairs =  {'fn': {'name': 'far-near', 
                               'tx': 'tx',
                               'rx': 'rx',
                               'tx_site': 'site_B',
                               'rx_site': 'site_A',
                               'f_GHz': float(dat['frequency'][0][0]),
                               'pol': pol,
                               'linecolor': 'b'}}
                               
        latA = float(dat['site1'][0][0][0])
        lonA = float(dat['site1'][0][0][1])
        latB = float(dat['site2'][0][0][0])
        lonB = float(dat['site2'][0][0][1])
        metadata = {'site_A': {'lat': latA,
                               'lon': lonA},
                    'site_B': {'lat': latB,
                               'lon': lonB},
                    'link_id': dat['name'][0],
                    'length_km': _haversine(lonA, latA, lonB, latB)}
    
    # !! This is only a quick hack and works correctly only
    # !! For data exported to csv from PROCEM database for link
    # !! gap0-oagau1 (which unfortunately does not export to .mat, 
    # !! probably because values like received_power_clear_sky are not
    # !! set) 
    if fext=='.csv':
        import re
               
        metadata_str = ''
        with open(fn) as f:
            line = f.readline()
            if line=='******************************METADATA******************************\n':
                for i in range(22):
                    metadata_str += f.readline()
            if not f.readline()=='********************************************************************\n':
                raise ValueError('Metadata format seems to be wrong')
        
        regex = re.compile('name: ([a-z0-9\-]*)')
        linkname = regex.findall(metadata_str)[0]  
        if linkname=='gap0-oagau1':
            param = namedtuple('param', 'name mV_clear_sky RSL_clear_sky dB_per_V')
            param.name = linkname
            param.mV_clear_sky = 1580
            param.RSL_clear_sky = -40
            param.dB_per_V = 29

        dat = np.recfromcsv(fn, 
                            skip_header=24, 
                            names=('name', 'ts', 'values'))
        data = pd.DataFrame(index=pd.to_datetime(pd.Series(dat['ts'])))
            
        # Convert RX power from mV recordings from data logger
        data['rx'] = _mV2RSL(dat['values'],
                             param.dB_per_V,
                             param.RSL_clear_sky,
                             param.mV_clear_sky)
        # Add column with constant TX power
        data['tx'] = 18.0
                             
        tx_rx_pairs =  {'fn': {'name': 'far-near', 
                               'tx': 'tx',
                               'rx': 'rx',
                               'tx_site': 'site_B',
                               'rx_site': 'site_A',
                               'f_GHz': 23.0,
                               'pol': 'V',
                               'linecolor': 'b'}}
        metadata = {'site_A': {'lat': 47.493,
                               'lon': 11.0971},
                    'site_B': {'lat': 47.5861,
                               'lon': 11.1028},
                    'link_id': linkname,
                    'length_km': 10.4}

    cml = Comlink(data=data, 
                  tx_rx_pairs=tx_rx_pairs,
                  metadata=metadata)
                             
    return cml

#############################################
# Helper functions for PROCEMA data parsing #
#############################################

def _mV2RSL(mV, dB_per_V, RSL_clear_sky, mV_clear_sky):
    """Helper function for PROCEMA data parsing
    """
    mV = _clean_mV_RSL_record(mV)
    RSL = RSL_clear_sky - (mV_clear_sky - mV) * 1e-3 * dB_per_V
    return RSL

def _clean_mV_RSL_record(mV_raw):
    """Helper function for PROCEMA data parsing
    """    
    import numpy as np
    mV = np.array(mV_raw, dtype=float)
    mV[mV==0] = np.NaN
    return mV
    
def _matlab_datenum_2_datetime(ts_datenum, round_to='seconds'):
    """Helper function for PROCEMA data parsing
    """    
    from datetime import datetime, timedelta
    ts = []
    for t in ts_datenum:
        ts_not_rounded = datetime.fromordinal(int(t)) \
                       + timedelta(days=t%1) \
                       - timedelta(days = 366)
        if round_to == 'seconds':
            ts.append(_round_datetime(ts_not_rounded, round_to='seconds'))
        elif round_to == 'None':
            ts.append(ts_not_rounded)
        else:
            print 'round_to value not supported'
    return ts
    
def _round_datetime(ts_not_rounded, round_to='seconds'):
    """Helper function for PROCEMA data parsing
    """    
    from datetime import timedelta
    seconds_not_rounded = ts_not_rounded.second \
                        + ts_not_rounded.microsecond * 1e-6
    ts_rounded = ts_not_rounded \
               + timedelta(seconds=round(seconds_not_rounded) \
                          -seconds_not_rounded)
    return ts_rounded
    
from math import radians, cos, sin, asin, sqrt
def _haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km
