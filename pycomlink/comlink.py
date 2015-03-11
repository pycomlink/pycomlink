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
from . import wet_antenna

class Comlink():
    """
    Commercial microwave link (CML) class for all data processing 
    
    Attributes
    ----------

    data : pandas.DataFrame
        OUTDATED!!!! Will be rewritten soon...    
    
        DataFrame which holds at minimum one time series of  RX-levels. Then,
        TX power is assumed to be constant at 20 dBmFor each,
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
        
    tx_rx_pairs : dict, optional
        Dictonary that defines which TX and RX values belong together and
        which frequency and polarization are used. Example:
            tx_rx_pairs =  {'fn': {'name': 'far-near', 
                                   'tx': 'tx_far',
                                   'rx': 'rx_near',
                                   'tx_site': 'site_B',
                                   'rx_site': 'site_A',
                                   'f_GHz': 17.8,
                                   'pol': 'V',
                                   'linecolor': 'r'},
                            'nf': {'name': 'near-far',
                                   'tx': 'tx_near',
                                   'rx': 'rx_far',
                                   'tx_site': 'site_A',
                                   'rx_site': 'site_B',
                                   'f_GHz': 18.8,
                                   'pol': 'V',
                                   'linecolor': 'b'}}
    
   metadata : dict, opional
        Dictonary with two keys for the two MW link sites. Each item holds
        another dict with at least 'lat' and 'lon' values in ???? units...
        Further keys, like 'ID' or 'site_name' are possible but not mandatory.
        If the 'lat' and 'lon' values are not supplied, the geolocating 
        functions do not work of course.
        Example site info dict:
            metadata = {'site_A': {'lat': 2123,
                                   'lon': 324,
                                   'id': 'MY1231',
                                   'ip': '127.0.0.1',
                                   'slot': 2},
                        'site_B': {'lat': 23123,
                                   'lon': 1231,
                                   'id': 'MY1231',
                                   'ip': '127.0.0.2',
                                   'slot': 3},
                        'link_id': 'MY2345_3_MY2345_2',
                        'length_km': 23.4}
    
    """
    def __init__(self, 
                 data, 
                 tx_rx_pairs=None, 
                 metadata=None,
                 const_TX_power=False):
        self.data = data
        self.tx_rx_pairs = tx_rx_pairs
        self.metadata = metadata
        self.processing_info = {}

        if const_TX_power is not False:
            # If the constant TX power is supplied as value, just add 
            # it with a default column name
            if type(const_TX_power) == int or type(const_TX_power) == float:
                self.data['tx'] = const_TX_power
            # If a tuple is supplied, the first value is the TX column name
            # an the second value is the TX power
            elif type(const_TX_power) == tuple:
                self.data[const_TX_power[0]] = const_TX_power[1]
            else:
                raise TypeError('const_TX_power must be int, float or tuple')

        # If no tx_rx_pairs are supplied, try to be smart and figure
        # them out by analysing the column names of the TXRX_df
        if tx_rx_pairs is None:
            tx_rx_pairs = derive_tx_rx_pairs(self.data.columns)
            self.tx_rx_pairs = tx_rx_pairs
        
        # TODO resolve protection link data in DataFrame

        # Calculate TX-RX
        # TODO change name of `column_names` since this does not make
        #      sense anymore.
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
        if 'ip' in self.metadata['site_A'].keys() and \
           'ip' in self.metadata['site_B'].keys():
            print ' IP: ' + self.metadata['site_A']['ip'] + '                 '  \
                          + self.metadata['site_B']['ip']
        for key, tx_rx_pair in self.tx_rx_pairs.iteritems():
            print '  f:   --------- ' + str(tx_rx_pair['f_GHz']) \
                                      + ' GHz ---------- '
#        print '      <--------- ' + str(self.metadata['f_GHz_fn']) \
#                                    + ' GHz ---------- ' 
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
            String or ensemble of sting which give the parameters that 
            will be plotted.
        resampling_time: 
            resampling the raw data
        add_raw_data:
            
        figsize:
            size of output figure
        kwargs:
            matplotlib parameters
        """

        if resampling_time is not None:
            df_temp = self.data.resample(resampling_time)
        else:
            df_temp = self.data

        if type(param_list) is str:
            param_list = [param_list,]

        fig, ax = plt.subplots(len(param_list), 1, 
                               squeeze=False,
                               figsize=figsize,
                               sharex=True)

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
                                                    ['linecolor']
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
            
            # Remove xticklabels for all but the bottom most plot
            if i < len(param_list):
                #plt.setp(ax[i][0].get_xticklabels(), visible=False)
                ax[i][0].xaxis.set_ticklabels([])
        return ax
        
        
    def do_wet_dry_classification(self, method='std_dev', 
                                        window_length=128,
                                        threshold=1,
                                        dry_window_length=600,
                                        f_divide=1e-3,
                                        reuse_last_Pxx=False,
                                        print_info=False):
        """
        Perform wet/dry classification for CML time series
        
     
        Attributes:
        ---------------------------------------------
        method: str
            WIP: currently two methods are supported:
                - std_dev: Rolling standard deviation method (Schleiss & Berne, 2010)
                - stft: Rolling Fourier-transform method (Chwala et al, 2012)
        window_length: int
            length of the sliding window        
        threshold: int
            threshold which has to be surpassed to classifiy a period as 'wet'
        .....................
        Only for method stft:
        
        dry_window_length: int
            length of window for identifying dry period used for calibration purpose
        f_divide: flt
        
        reuse_last_Pxx: bool
        
        
        .....................
        print_info: bool
            print information about executed method
        -------------------------------------------    
            
        """
        
        # Standard deviation method
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
        # Shor-term Fourier transformation method
        elif method == 'stft':
            if print_info:
                print 'Performing wet/dry classification'
                print ' Method = stft'
                print ' dry_window_length = ' + str(dry_window_length)
                print ' window_length = ' + str(window_length)
                print ' threshold = ' + str(threshold)
                print ' f_divide = ' + str(f_divide)
            
            for pair_id in self.processing_info['tx_rx_pairs']:
                txrx = self.data['txrx_' + pair_id].values
            
                # Find dry period (wit lowest fluctuation = lowest std_dev)
                t_dry_start, \
                t_dry_stop = wet_dry.find_lowest_std_dev_period(
                                txrx,
                                window_length=dry_window_length)
                self.processing_info['wet_dry_t_dry_start'] = t_dry_start
                self.processing_info['wet_dry_t_dry_stop'] = t_dry_stop
            
                if reuse_last_Pxx is False:
                    self.data['wet_' + pair_id], info = wet_dry.wet_dry_stft(
                                                            txrx,
                                                            window_length,
                                                            threshold,
                                                            f_divide,
                                                            t_dry_start,
                                                            t_dry_stop)
                elif reuse_last_Pxx is True:
                    Pxx=self.processing_info['wet_dry_Pxx_' + pair_id]
                    f=self.processing_info['wet_dry_f']
                    self.data['wet_' + pair_id], info = wet_dry.wet_dry_stft(
                                                            txrx,
                                                            window_length,
                                                            threshold,
                                                            f_divide,
                                                            t_dry_start,
                                                            t_dry_stop,
                                                            Pxx=Pxx,
                                                            f=f)
                else:
                    raise ValueError('reuse_last_Pxx can only by True or False')
                self.processing_info['wet_dry_Pxx_' + pair_id] = \
                                                        info['Pxx']
                self.processing_info['wet_dry_P_norm_' + pair_id] = \
                                                        info['P_norm']
                self.processing_info['wet_dry_P_sum_diff_' + pair_id] = \
                                                        info['P_sum_diff']
                self.processing_info['wet_dry_P_dry_mean_' + pair_id] = \
                                                        info['P_dry_mean']
            
            self.processing_info['wet_dry_f'] = info['f']                
            self.processing_info['wet_dry_method'] = 'stft'
            self.processing_info['wet_dry_window_length'] = window_length
            self.processing_info['dry_window_length'] = window_length
            self.processing_info['f_divide'] = f_divide
            self.processing_info['wet_dry_threshold'] = threshold
        else:
            ValueError('Wet/dry classification method not supported')
        
    def do_baseline_determination(self, 
                                  method='constant',
                                  wet_external=None,
                                  print_info=False):
                                      
        """
        Perform baseline determination for CML time series
        
        Attributes:
        ---------------------------------------------
        method: str
            supported methods:
                constant
                linear
        wet_external:      
        
        print_info: bool
            print information about executed method        
        """   
                                      
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
        
    def do_wet_antenna_baseline_adjust(self,
                                       waa_max, 
                                       delta_t, 
                                       tau,
                                       wet_external=None):

        """
        Perform baseline adjustion due to wet antenna for CML time series
        
        Attributes:
        ---------------------------------------------
        waa_max: 

        delta_t:    
        
        tau:
        
        wet_external:
               
        """ 
                                           
        for pair_id in self.processing_info['tx_rx_pairs']:
            txrx = self.data['txrx_' + pair_id].values
            baseline = self.data['baseline_' + pair_id].values
            if wet_external is None:            
                wet = self.data['wet_' + pair_id]
            else:
                wet = wet_external
            baseline_waa, waa = wet_antenna.waa_adjust_baseline(rsl=txrx,
                                                           baseline=baseline,
                                                           waa_max=waa_max,
                                                           delta_t=delta_t,
                                                           tau=tau,
                                                           wet=wet)
            self.data['baseline_' + pair_id] = baseline_waa
            self.data['waa_' + pair_id] = waa
            #return baseline_waa

    def calc_A(self, remove_negative_A=True):
        """
        Perform calculation of attenuation for CML time series
        
        Attributes:
        ---------------------------------------------
        remove_negative_A: bool
            assignment: negative values of Attenuation = 0
               
        """           
        for pair_id in self.processing_info['tx_rx_pairs']:
            self.data['A_' + pair_id] = self.data['txrx_' + pair_id] \
                                      - self.data['baseline_' + pair_id]
            if remove_negative_A:
                self.data['A_' + pair_id][self.data['A_' + pair_id]<0] = 0
                
    def calc_R_from_A(self, a=None, b=None, approx_type='ITU'):
        """
        Perform calculation of rain rate from attenuation for CML time series
        
        Attributes:
        ---------------------------------------------
        a: flt
            Parameter of A-R relationship
            If not given: Approximation considering polarization and frequency of link
        b: flt
            Parameter of A-R relationship
            If not given: Approximation considering polarization and frequency of link        
        approx_type: str
            Type used for approximate a and b
            Supported type : ITU
            
               
        """         
        if a==None or b==None:
            calc_a_b = True
        else:
            calc_a_b = False
        for pair_id in self.processing_info['tx_rx_pairs']:
            if calc_a_b:
                a, b = A_R_relation.a_b(f_GHz=self.tx_rx_pairs[pair_id]['f_GHz'], 
                                        pol=self.tx_rx_pairs[pair_id]['pol'],
                                        approx_type=approx_type)
                self.processing_info['a_' + pair_id] = a
                self.processing_info['b_' + pair_id] = b

            self.data['R_' + pair_id] = \
                A_R_relation.calc_R_from_A(self.data['A_' + pair_id], 
                                           a, b,
                                           self.metadata['length_km'])
                  
####################                        
# Helper functions #
####################
                  
def derive_tx_rx_pairs(columns_names):
    """ 
    Derive the TX-RX pairs from the MW link data columns names
    
    Right now, this only works for the following cases:
    1. A duplex link with TX and RX columns for each direction. The
       Naming convention is 'TX_something', or 'tx_something',
       or 'RX_...', 'rx_...', where `something` is most commonly `far`/`near`
       or `A`/`B`.
    2. A simplex link with one TX and RX columns which also carry the site
       name after a '_' like in case 1.
    3. A simplex link with one TX and RX columns which are named 'TX' 
       and 'RX', or 'tx' and 'rx'.
    
    Parameters
    ==========
    
    column_names : list
        List of columns names from the MW link DataFrame
        
    Returns
    =======
    tx_rx_pairs : dict of dicts
        Dict of dicts of the TX-RX pairs
    
    """
    
    tx_patterns = ['tx', 'TX']
    rx_patterns = ['rx', 'RX']
    patterns = tx_patterns + rx_patterns

    # Find the columns for each of the pattern
    pattern_found_dict = {}
    for pattern in patterns:
        pattern_found_dict[pattern] = []
        for name in columns_names:
            if pattern in name:
                pattern_found_dict[pattern].append(name)
        if pattern_found_dict[pattern] == []:
            # Remove those keys where no match was found
            pattern_found_dict.pop(pattern, None)
    
    # Do different things, depending on how many column names 
    # did fit to one of the patterns
    #
    # If both, TX and RX were found, the dict length should be 2
    if len(pattern_found_dict) == 2:
        # Derive site name guesses from column names, i.e. the 'a' from 'tx_a'
        site_name_guesses = []
        for pattern, column_name_list in pattern_found_dict.iteritems():
            for col_name in column_name_list:
                if '_' in col_name:
                    site_name_guesses.append(col_name.split('_')[1])
                    no_site_name_in_tx_rx_column = False
                elif col_name in tx_patterns:
                    site_name_guesses.append('B')
                    no_site_name_in_tx_rx_column = True
                elif col_name in rx_patterns:
                    site_name_guesses.append('A')
                    no_site_name_in_tx_rx_column = True
                else:
                    raise ValueError('Column name can not be recognized')
        # Check that there are two columns for each site name guess
        unique_site_names = list(set(site_name_guesses))
        if len(unique_site_names) != 2:
            raise ValueError('There should be exactly two site names')
        for site_name in unique_site_names:
            if site_name_guesses.count(site_name) != len(site_name_guesses)/ \
                                                     len(unique_site_names):
                raise ValueError('Site names were found but they must always be two correspoding columns for each name')
        # Build tx_rx_dict
        site_a_key = unique_site_names[0]
        site_b_key = unique_site_names[1]
        for tx_pattern in tx_patterns:
            if tx_pattern in pattern_found_dict.keys():
                break
        else:
            raise ValueError('There must be a match between the tx_patterns and the patterns already found')
        for rx_pattern in rx_patterns:
            if rx_pattern in pattern_found_dict.keys():
                break
        else:
            raise ValueError('There must be a match between the rx_patterns and the patterns already found')
        # If we have two directions, each with a TX and RX signal
        if len(site_name_guesses)/len(unique_site_names) == 2:
            tx_rx_pairs = {site_a_key + site_b_key: {'name': site_a_key + '-' + site_b_key,
                                                    'tx': tx_pattern + '_' + site_a_key,
                                                    'rx': rx_pattern + '_' + site_b_key,
                                                    'linecolor': 'b'},
                          site_b_key + site_a_key: {'name': site_b_key + '-' + site_a_key,
                                                    'tx': tx_pattern + '_' + site_b_key,
                                                    'rx': rx_pattern + '_' + site_a_key,
                                                    'linecolor': 'r'}}
        # If we have only one direction witch a TX and RX signal
        if len(site_name_guesses)/len(unique_site_names) == 1:
            if no_site_name_in_tx_rx_column:
                tx_rx_pairs = {site_b_key + site_a_key: {'name': site_b_key + '-' + site_a_key,
                                                         'tx': tx_pattern,
                                                         'rx': rx_pattern,
                                                         'linecolor': 'b'}}
                                
    # TODO Try to deal with the case were only a RX column is present            
    elif len(pattern_found_dict) == 1:
        raise ValueError('Only one column was found to fit the TX- and RX-naming convetion')
    elif len(pattern_found_dict) == 0:
        raise ValueError('None of the columns was found to fit the TX- and RX-naming convention')
    else:
        raise ValueError('There seem to be too many columns that fit the TX- and RX-naming convention')
    
    return tx_rx_pairs
    
    
    
        
        