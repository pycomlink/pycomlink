#----------------------------------------------------------------------------
# Name:         comlink
# Purpose:      Commercial MW link Class to handle all processing steps
#
# Authors:      Christian Chwala
#
# Created:      01.12.2014
# Copyright:    (c) Christian Chwala 2014
# Licence:      The MIT License
#----------------------------------------------------------------------------


from __future__ import division
import matplotlib.pyplot as plt


class Comlink():
    """
    Commercial microwave link (CML) class for all data processing 
    
    Attributes
    ----------

    data : pandas.DataFrame
        DataFrame which holds at minimum the TX- and RX-levels. For each,
        far end and near end entries can exists. Furthermore, for protection
        links additional TX- and RX-level may exists. The naming convention 
        is:
         'tx_far'         = TX level far end 
         'tx_near'        = TX level near end
         'rx_far'         = RX level far end
         'rx_near'        = RX level near end
         'tx_far_protect' = TX level far end of protection link
         ....
         ...
         ..
         .
        Further columns can be present in the DataFrame, 
        e.g. RTT (the round trip time of a SNMP data acquisition request).
    param : 
        Metadata for the CML. Important are the site locations and the 
        CML frequency.
    
    """
    def __init__(self, metadata, TXRX_df):
        self.metadata = metadata
        self.data = TXRX_df

        # Calculate TX-RX
        self.data['txrx_nf'] = self.data.tx_near - self.data.rx_far
        self.data['txrx_fn'] = self.data.tx_far - self.data.rx_near
    
    def plot_txrx(self, resampling_time=None, **kwargs):
        """Plot TX- minus RX-level
        
        Parameters
        ----------
        
        resampling_time : str
            Resampling time according to Pandas resampling time options,
            e.g. ('min, '5min, 'H', 'D', ...)
        kwargs : 
            kwargs for Pandas plotting                        
            
        """
        
        if resampling_time != None:
            df_temp = self.data.resample(resampling_time)
        else:
            df_temp = self.data
        df_temp.txrx_nf.plot(label='near-far', **kwargs)
        df_temp.txrx_fn.plot(label='far-near', **kwargs)
        plt.legend(loc='best')
        #plt.title(self.metadata.)
    
    def plot_tx_rx_seperate(self, resampling_time=None, **kwargs):
        """Plot two linked plots for TX- and TX- minus RX-level
        
        Parameters
        ----------
        
        resampling_time : str
            Resampling time according to Pandas resampling time options,
            e.g. ('min, '5min, 'H', 'D', ...)
        kwargs : 
            kwargs for Pandas plotting                        
            
        """
        if resampling_time != None:
            df_temp = self.data.resample(resampling_time)
        else:
            df_temp = self.data
        fig, ax = plt.subplots(2,1, sharex=True, **kwargs)
        df_temp.tx_far.plot(label='near-far', ax=ax[0])
        df_temp.tx_near.plot(label='far-near', ax=ax[0])
        df_temp.txrx_nf.plot(label='near-far', ax=ax[1])
        df_temp.txrx_fn.plot(label='far-near', ax=ax[1])
        plt.legend(loc='best')
        #plt.title(self.metadata.)
        
    def do_wet_dry_classification(self, method='std_dev', 
                                        window_length=128,
                                        threshold=1,
                                        f_divide=1e-3,
                                        t_dry_start=None,
                                        t_dry_stop=None,
                                        reuse_last_Pxx=False,
                                        print_info=False):
        if method == 'std_dev':
            if print_info:
                print 'Performing wet/dry classification'
                print ' Method = std_dev'
                print ' window_length = ' + str(window_length)
                print ' threshold = ' + str(threshold)
            (self.data['wet'], 
             roll_std_dev) = wet_dry_std_dev(self.data.rsl.values, 
                                           window_length, 
                                           threshold)
            self.processing_info['wet_dry_method'] = 'std_dev'
            self.processing_info['wet_dry_window_length'] = window_length
            self.processing_info['wet_dry_threshold'] = threshold
            self.processing_info['wet_dry_roll_std_dev'] = roll_std_dev
            return self.data.wet
        
        
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
        

    
    
    