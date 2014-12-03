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
        a far end and near end entry must exists. The naming convention is
        'TX_far', 'TX_near', 'RX_far', 'RX_near'. Further columns can be
        present in the DataFrame, e.g. RTT (the round trip time of a SNMP
        data acquisition request).
    param : 
        Metadata for the CML. Important are the site locations and the 
        CML frequency.
    
    """
    def __init__(self, metadata, TXRX_df):
        self.metadata = metadata
        self.data = TXRX_df
        
        self.data.rename(columns={'txf': 'tx_far',
                                  'txn': 'tx_near',
                                  'rxf': 'rx_far',
                                  'rxn': 'rx_near'},
                                  inplace=True)

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
        
        

    
    
    