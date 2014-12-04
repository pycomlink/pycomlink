
        
###############################################################            
# Functions for the wet/dry classification of RSL time series #          
###############################################################
        
#-------------------------------------#        
# Rolling std deviation window method #
#-------------------------------------#
                                                                                                    
def wet_dry_std_dev(rsl, window_length, threshold):
    roll_std_dev = rolling_std_dev(rsl, window_length)
    wet = roll_std_dev > threshold
    return wet, roll_std_dev

def rolling_window(a, window):
    import numpy as np
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_std_dev(x, window_length):
    import numpy as np
    roll_std_dev = np.std(rolling_window(x, window_length), 1)
    pad_nan = np.zeros(window_length-1)
    pad_nan[:] = np.NaN
    roll_std_dev = np.concatenate((pad_nan, roll_std_dev))
    return roll_std_dev
        
