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

def to_hdf5(cml, fn, complib=None, complevel=9):
    """Write Comlink object to HDF5 

    WIP....
    
    TODO: Also write processing info (and its timeseries to hdf5)
    
    Parameters
    ----------
    
    cml : Comlink
        A Comlink object
    fn : str
        Filename
        
    """
    
    # Open pandas HDF5 file interface object
    store = pd.HDFStore(fn, complib=complib, complevel=complevel)
    # Store MW link data from data DataFrame
    store['mw_df'] = cml.data
    
    # Quick hack to store metadata dict: Dict to DataFrame --> HDF5
    temp_metadata_df = pd.DataFrame.from_dict(cml.metadata, orient='index')
    store['metadata'] = temp_metadata_df
    
    store.close()
    
def from_hdf5(fn):
    """Read Comlink object frmo HDF5
    
    WIP...
    
    TODO: Also read processing info (when `to_hdf()` is able to store it...)

    Parameters
    ----------
    
    fn : str
        Filename
        
    """
    
    store = pd.HDFStore(fn, 'r')    
    data_df = store['mw_df']
    metadata_dict = pd.DataFrame.to_dict(store['metadata'])[0]
    
    cml = Comlink(metadata=metadata_dict, TXRX_df=data_df)
    
    return cml
    
def from_PROCEMA_raw_data(fn):
    """ Read in PROCEMA data for one MW link stored as CSV or MATLAB binary
    
    Parameters
    ----------
    
    fn : str
        Filename
        
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
        
        data = pd.DataFrame({'mV' : pd.Series(
                                        dat['values'][0],
                                        index=matlab_datenum_2_datetime(
                                              dat['time'][0]))})
        
        data['rx'] = mV2RSL(data.mV,
                             param.dB_per_V,
                             param.RSL_clear_sky,
                             param.mV_clear_sky)
    
    # !! This is only a quick hack and works correctly only
    # !! For data exported to csv from PROCEM database for link
    # !! gap0-oagau1 (which unfortunately does not export to .mat, 
    # !! probably because values like received_power_clear_sky are not
    # !! set) 
    if fext=='.csv':
        import re
        
        dat = np.recfromcsv(fn, 
                            skip_header=24, 
                            names=('name', 'ts', 'values'))
        data = pd.DataFrame(index=pd.to_datetime(pd.Series(dat['ts'])))
        data['mV'] = dat['values']
        
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
            
        data['rx'] = mV2RSL(data.mV,
                             param.dB_per_V,
                             param.RSL_clear_sky,
                             param.mV_clear_sky)
                             
    return data


# Helper functions for PROCEMA data parsing

def mV2RSL(mV, dB_per_V, RSL_clear_sky, mV_clear_sky):
    mV = clean_mV_RSL_record(mV)
    RSL = RSL_clear_sky - (mV_clear_sky - mV) * 1e-3 * dB_per_V
    return RSL

def clean_mV_RSL_record(mV_raw):
    import numpy as np
    mV = np.array(mV_raw, dtype=float)
    mV[mV==0] = np.NaN
    return mV
    
def matlab_datenum_2_datetime(ts_datenum, round_to='seconds'):
    from datetime import datetime, timedelta
    ts = []
    for t in ts_datenum:
        ts_not_rounded = datetime.fromordinal(int(t)) \
                       + timedelta(days=t%1) \
                       - timedelta(days = 366)
        if round_to == 'seconds':
            ts.append(round_datetime(ts_not_rounded, round_to='seconds'))
        elif round_to == 'None':
            ts.append(ts_not_rounded)
        else:
            print 'round_to value not supported'
    return ts
    
def round_datetime(ts_not_rounded, round_to='seconds'):
    from datetime import timedelta
    seconds_not_rounded = ts_not_rounded.second \
                        + ts_not_rounded.microsecond * 1e-6
    ts_rounded = ts_not_rounded \
               + timedelta(seconds=round(seconds_not_rounded) \
                          -seconds_not_rounded)
    return ts_rounded