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

from collections import namedtuple

from comlink import Comlink
from comlinkset import ComlinkSet


def write_hdf5(fn, cml, cml_id=None):
    """ 
    Write Comlink or Comlink list to HDF5
    
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
    
def read_hdf5(fn, force_list_return=False):
    """
    Read Comlink or list of Comlinks from HDF5
    
    Parameters
    ----------

    fn : str
        Absolute filename to a pycomlink-HDF5 file
    force_list_return : Bool, optional
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
    mV = _clean_mV_RSL_record(mV)
    RSL = RSL_clear_sky - (mV_clear_sky - mV) * 1e-3 * dB_per_V
    return RSL

def _clean_mV_RSL_record(mV_raw):
    import numpy as np
    mV = np.array(mV_raw, dtype=float)
    mV[mV==0] = np.NaN
    return mV
    
def _matlab_datenum_2_datetime(ts_datenum, round_to='seconds'):
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
    from datetime import timedelta
    seconds_not_rounded = ts_not_rounded.second \
                        + ts_not_rounded.microsecond * 1e-6
    ts_rounded = ts_not_rounded \
               + timedelta(seconds=round(seconds_not_rounded) \
                          -seconds_not_rounded)
    return ts_rounded
    
from math import radians, cos, sin, asin, sqrt
def _haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
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