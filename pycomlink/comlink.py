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

import numpy as np
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

        tx_rx_pairs = {'fn': {'name': 'far-near', 
                              'tx': 'tx_far',
                              'rx': 'rx_near',
                              'color': 'r'},
                       'nf': {'name': 'near-far',
                              'tx': 'tx_near',
                              'rx': 'rx_far',
                              'color': 'b'}}

        # Calculate TX-RX
        for pair_id, column_names in tx_rx_pairs.iteritems():
            self.data['txrx_' + pair_id] = self.data[column_names['tx']] \
                                         - self.data[column_names['rx']]
        self.processing_info['tx_rx_pairs'] = tx_rx_pairs
    
    def info(self):
        """Print MW link info 
        
        WIP: Print rudimentary MW link information
        
        """
        
        # TODO: Deal with protection links or links for which only
        #       unidirectional data is available
        
        print '============================================================='
        print 'ID: ' + self.metadata['link_id']
        print '-------------------------------------------------------------'
        print '     Site A                       Site B'
        print ' IP: ' + self.metadata['ip_a'] + '                  '  \
                      + self.metadata['ip_b']
        print '  f:   --------- ' + str(self.metadata['f_GHz_nf']) \
                                    + ' GHz ----------> '
        print '      <--------- ' + str(self.metadata['f_GHz_fn']) \
                                    + ' GHz ---------- ' 
        print '  L: ' + str(self.metadata['length_km']) + ' km'
        print '============================================================='
    
    def plot(self, 
             param_list=['txrx'], 
             resampling_time=None, 
             add_raw_data=False,
             figsize=(6,4),
             **kwargs):
        """ WIP for generic plotting function
        
        Parameters
        ----------
        
        param_list : str, list of str, tuple of str, list of tuple of str
            List of parameters to plot.....
            ....
            ...bla bla
        ...
        ..
        .
        
        """

        if resampling_time is not None:
            df_temp = self.data.resample(resampling_time)
        else:
            df_temp = self.data

        if type(param_list) is str:
            param_list = [param_list,]

        fig, ax = plt.subplots(len(param_list), 1, 
                               squeeze=False,
                               figsize=figsize)

        for i, param in enumerate(param_list):
            if type(param) is str:
                if param in self.data.columns:
                    df_temp[param].plot(ax=ax[i][0], 
                                        label=param,
                                        **kwargs)
                else:
                    # Try parameter + tx_rx_pair_identifier
                    for txrx_pair_id in self.processing_info['tx_rx_pairs']:
                        if param == 'tx' or param == 'rx':
                            param_temp = self.processing_info['tx_rx_pairs']\
                                                        [txrx_pair_id]\
                                                        [param]
                        else:
                            param_temp = param + '_' + txrx_pair_id
                            
                        color = self.processing_info['tx_rx_pairs']\
                                                    [txrx_pair_id]\
                                                    ['color']
                        name = self.processing_info['tx_rx_pairs']\
                                                   [txrx_pair_id]\
                                                   ['name']
                        df_temp[param_temp].plot(ax=ax[i][0], 
                                                 label=name,
                                                 color=color,
                                                 **kwargs)
            elif type(param) is tuple:
                for param_item in param:
                    df_temp[param_item].plot(ax=ax[i][0], 
                                             label=param_item, 
                                             **kwargs)
            ax[i][0].legend(loc='best')
            ax[i][0].set_ylabel(param)
        return ax
                    
            
    
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
        df_temp.txrx_fn.plot(label='far-near', color='r', **kwargs)        
        df_temp.txrx_nf.plot(label='near-far', color='b', **kwargs)        
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
            for pair_id in self.processing_info['tx_rx_pairs']:
                (self.data['wet_' + pair_id], 
                 roll_std_dev) = wet_dry.wet_dry_std_dev(
                                    self.data['txrx_' + pair_id].values, 
                                    window_length, 
                                    threshold)
                self.processing_info['wet_dry_roll_std_dev_' + pair_id] \
                                  = roll_std_dev
            self.processing_info['wet_dry_method'] = 'std_dev'
            self.processing_info['wet_dry_window_length'] = window_length
            self.processing_info['wet_dry_threshold'] = threshold
        else:
            ValueError('Wet/dry classification method not supported')
        
    def do_baseline_determination(self, 
                                  method='constant',
                                  wet_external=None,
                                  print_info=False):
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
        for pair_id in self.processing_info['tx_rx_pairs']:
            if wet_external is None:            
                wet = self.data['wet_' + pair_id]
            else:
                wet = wet_external
            self.data['baseline_' + pair_id] = \
                                baseline_func(self.data['txrx_' + pair_id], 
                                              wet)
        self.processing_info['baseline_method'] = method

    def calc_A(self, remove_negative_A=True):
        for pair_id in self.processing_info['tx_rx_pairs']:
            self.data['A_' + pair_id] = self.data['txrx_' + pair_id] \
                                      - self.data['baseline_' + pair_id]
            if remove_negative_A:
                self.data['A_' + pair_id][self.data['A_' + pair_id]<0] = 0
                
    def calc_R_from_A(self, a=None, b=None, approx_type='ITU'):
        if a==None or b==None:
            calc_a_b = True
        else:
            calc_a_b = False
        for pair_id in self.processing_info['tx_rx_pairs']:
            if calc_a_b:
                a, b = A_R_relation.a_b(f_GHz=self.metadata['f_GHz_' \
                                                            + pair_id], 
                                        pol=self.metadata['pol_' \
                                                          + pair_id],
                                        approx_type=approx_type)
                self.processing_info['a_' + pair_id] = a
                self.processing_info['b_' + pair_id] = b

            self.data['R_' + pair_id] = \
                A_R_relation.calc_R_from_A(self.data['A_' + pair_id], 
                                           a, b,
                                           self.metadata['length_km'])
                                          
                