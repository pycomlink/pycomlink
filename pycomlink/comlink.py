#-------------------------------------------------------------------------------
# Name:         comlink
# Purpose:
#
# Authors:      Christian Chwala
#
# Created:      01.12.2014
# Copyright:    (c) Christian Chwala 2014
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python


from __future__ import division
import matplotlib.pyplot as plt


class Comlink():
    ''' 
    Commercial microwave link class for all data processing 
    '''
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
        if resampling_time != None:
            df_temp = self.data.resample(resampling_time)
        else:
            df_temp = self.data
        df_temp.txrx_nf.plot(label='near-far', **kwargs)
        df_temp.txrx_fn.plot(label='far-near', **kwargs)
        plt.legend(loc='best')
        #plt.title(self.metadata.)
    
    def plot_tx_rx_seperate(self, resampling_time=None, **kwargs):
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
        
        

    
    
    