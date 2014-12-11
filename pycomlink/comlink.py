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

from . import wet_dry
from . import baseline
from . import A_R_relation

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
        self.processing_info = {}

        # TODO Check column names for the available TX and RX values
        
        # TODO resolve protection link data in DataFrame

        tx_rx_pairs = {'fn': {'tx': 'tx_far', 'rx': 'rx_near'},
                       'nf': {'tx': 'tx_near', 'rx': 'rx_far'}}

        # Calculate TX-RX
        for pair_name, column_names in tx_rx_pairs.iteritems():
            self.data['txrx_' + pair_name] = self.data[column_names['tx']] \
                                           - self.data[column_names['rx']]
        self.processing_info['tx_rx_pairs'] = tx_rx_pairs
    
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
        plt.ylabel('TX-RX level in dB')
        #plt.title(self.metadata.)
    
    def plot_tx_txrx_seperate(self, resampling_time=None, **kwargs):
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
        df_temp.tx_near.plot(label='near-far', ax=ax[0])
        df_temp.tx_far.plot(label='far-near', ax=ax[0])
        df_temp.txrx_nf.plot(label='near-far', ax=ax[1])
        df_temp.txrx_fn.plot(label='far-near', ax=ax[1])
        plt.legend(loc='best')
        #plt.title(self.metadata.)
        
    def plot_tx_rx_txrx_seperate(self, resampling_time=None, **kwargs):
        """Plot three linked plots for TX-, RX- and TX- minus RX-level
        
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
        fig, ax = plt.subplots(3,1, sharex=True, **kwargs)
        df_temp.tx_near.plot(label='near-far', ax=ax[0])
        df_temp.tx_far.plot(label='far-near', ax=ax[0])
        df_temp.rx_far.plot(label='near-far', ax=ax[1])
        df_temp.rx_near.plot(label='far-near', ax=ax[1])
        df_temp.txrx_nf.plot(label='near-far', ax=ax[2])
        df_temp.txrx_fn.plot(label='far-near', ax=ax[2])
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
            for pair_name in self.processing_info['tx_rx_pairs']:
                (self.data['wet_' + pair_name], 
                 roll_std_dev) = wet_dry.wet_dry_std_dev(
                                    self.data['txrx_' + pair_name].values, 
                                    window_length, 
                                    threshold)
                self.processing_info['wet_dry_roll_std_dev_' 
                                    + pair_name] = roll_std_dev
            self.processing_info['wet_dry_method'] = 'std_dev'
            self.processing_info['wet_dry_window_length'] = window_length
            self.processing_info['wet_dry_threshold'] = threshold
        else:
            ValueError('Wet/dry classification method not supported')
        
    def do_baseline_determination(self, method='constant',print_info=False):
        if method == 'constant':
            baseline_func = baseline.baseline_constant
            if print_info:
                print 'Setting RSL baseline'
                print ' Method = constant'
        elif method == 'linear':
            baseline_func = baseline.baseline_linear
            if print_info:
                print 'Setting RSL baseline'
                print ' Method = linear'
        else:
            ValueError('Wrong baseline method')
        for pair_name in self.processing_info['tx_rx_pairs']:
            self.data['baseline_' + pair_name] = \
                                baseline_func(self.data['txrx_' + pair_name], 
                                              self.data['wet_' + pair_name])
        self.processing_info['baseline_method'] = method

    def calc_A(self, remove_negative_A=True):
        for pair_name in self.processing_info['tx_rx_pairs']:
            self.data['A_' + pair_name] = self.data['txrx_' + pair_name] \
                                        - self.data['baseline_' + pair_name]
            if remove_negative_A:
                self.data['A_' + pair_name][self.data['A_' + pair_name]<0] = 0
                
    def calc_R_from_A(self, a=None, b=None, approx_type='ITU'):
        if a==None or b==None:
            a, b = A_R_relation.a_b(f_GHz=self.metadata['f_GHz'], 
                                    pol=self.metadata['pol'],
                                    approx_type=approx_type)
        for pair_name in self.processing_info['tx_rx_pairs']:
            self.data['R_' + pair_name] = \
                A_R_relation.calc_R_from_A(self.data['A_' + pair_name], 
                                           a, b,
                                           self.metadata['length_km'])
                                          
                