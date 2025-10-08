import numpy as np
from numpy.lib.stride_tricks import sliding_window_view 
import pickle
from importlib.resources import files

# Open pickled scikit-learn models
def get_model_file_path():
    return files("pycomlink").joinpath("processing/wet_dry/mlp_model_files")

with open(get_model_file_path() / "model_rg.pkl", 'rb') as f:
    model_rg = pickle.load(f)
    
with open(get_model_file_path() / "model_rad.pkl", 'rb') as f:
    model_rad = pickle.load(f)
    
with open(get_model_file_path() / "model_rg_online.pkl", 'rb') as f:
    model_rg_online = pickle.load(f)
    
with open(get_model_file_path() / "model_rad_online.pkl", 'rb') as f:
    model_rad_online = pickle.load(f)
    
    
def mlp_wet_dry(
    trsl_channel_1, 
    trsl_channel_2,
    model_sel = 'rad_online',
):
    """
    Wet dry classification using a simple neural network (MLP):
    
    Calculates wet and dry periods using a 40 minutes rolling window 
    for the CML signal loss from two sublinks (trsl_channel_1 and 
    trsl_channel_2) with temporal resolution equal to 1 minute. See notebook
    under ./pycomlink/notebooks/Rain event detection methods.ibynp for examples
    on how to run.
    
    This module contains 4 MLP models: 
        - rg: Model that was trained on rain gauge data using 6 hour rolling 
        median for detrending. 
        - rad: Model that was trained on radar data using 6 hour rolling 
        median for detrending. 
        - rg_online: Model that was trained on rain gauge data using the first
        order derivative for detrending. 
        - rad_online: Model that was trained on radar data using the first 
        order derivative for detrending. 
        
    The models that utilize a 6-hour rolling median for detrending are 
    documented in this publication: https://doi.org/10.5194/egusphere-2024-647. 
    Please note that the MLPs provided in this module are retrained versions 
    of the MLPs used in the publication. The classifications might thus differ
    slightly. Also note that the models which employ the 1st order derivative 
    for detrending have not yet been published or undergone extensive testing.
    
    For access to training data see: https://github.com/eoydvin/cml_wd_mlp. 
    This data can be used to train new MLPs, for instance using CMLs with 5
    minute resolution, or entirely different models. 
    
    Parameters
    ----------
    trsl_channel_1: xarray.DataArray 
         Time series of received signal level of sublink 1
    trsl_channel_2: xarray.DataArray
         Time series of received signal level of sublink 2
    model_sel : str
        MLP model to use. Set to 'rg', 'rad', 'rg_online' or 'rad_online'. 
        
    Returns
    -------
    iterable of float
        Time series of wet/dry probability or (if threshold is provided) 
        wet dry classification. Run np.argmax(mlp_out, axis = 1) on the 
        probability output to get optimal wet dry classification. 
        
    References
    ----------

    """
    
    # Select MLP model
    if model_sel == 'rg':
        model = model_rg    
        
    elif model_sel == 'rad':
        model = model_rad        
  
    elif model_sel == 'rad_online':
        model = model_rad_online        
        
    elif model_sel == 'rg_online':
        model = model_rg_online        
        
    else:
        raise ValueError('Did not recognize model')
        
    # Detrending of CML time series
    if (model_sel == 'rad') | (model_sel == 'rg'):
        # Detrending channel 1 using rolling median        
        trsl1 =  trsl_channel_1 - trsl_channel_1.rolling(
            time = 12*60, 
            min_periods=2 * 60, 
            center = True
        ).median().data
        
        # Detrending channel 2 using rolling median   
        trsl2 =  trsl_channel_2 - trsl_channel_2.rolling(
            time = 12*60, 
            min_periods=2 * 60, 
            center = True
        ).median().data
        
        # add nan to start and end of design matrix
        windowsize = 40 # use two channels 
        x_start = np.ones([int(windowsize/2), windowsize*2])*np.nan
        x_end = np.ones([int(windowsize/2)- 1, windowsize*2])*np.nan
        
        # sliding window
        sw_ch1 = sliding_window_view(trsl1, window_shape = windowsize)
        sw_ch2 = sliding_window_view(trsl2, window_shape = windowsize)
    
        # Create design matrix
        x_fts = np.vstack([x_start, np.hstack([sw_ch1, sw_ch2]), x_end])        
        
    elif (model_sel == 'rg_online') | (model_sel == 'rad_online'):
        # Detrending using 1st order derivative    
        trsl1 = trsl_channel_1.diff(dim = 'time', n = 1).data
        trsl2 = trsl_channel_2.diff(dim = 'time', n = 1).data
    
        # add nan to start of design matrix
        windowsize = 40 # use two channels 
        x_start = np.ones([int(windowsize), windowsize*2])*np.nan 
        
        # sliding window
        sw_ch1 = sliding_window_view(trsl1, window_shape = windowsize)
        sw_ch2 = sliding_window_view(trsl2, window_shape = windowsize)
    
        # Create design matrix
        x_fts = np.vstack([x_start, np.hstack([sw_ch1, sw_ch2])])        
    
    # Create matrix for storing estimates
    mlp_pred = np.full([x_fts.shape[0], 2], np.nan)
    
    # Get indices of timesteps to do prediction
    indices = np.argwhere(~np.isnan(x_fts).any(axis = 1)).ravel()
    
    # predic using MLP model
    if indices.size > 0: # else: predictions are kept as nan
        mlp_pred[indices] = model.predict_proba(x_fts[indices])   
    
    return mlp_pred