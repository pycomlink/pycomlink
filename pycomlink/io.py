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



import pandas as pd
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
    