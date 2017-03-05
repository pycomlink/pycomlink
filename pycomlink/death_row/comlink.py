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
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
from cartopy.io import img_tiles

from pycomlink import wet_dry
from pycomlink.processing.baseline import baseline
from pycomlink.processing.A_R_relation import A_R_relation
from pycomlink.processing.wet_antenna import wet_antenna
from pycomlink.death_row import mapping


class Comlink():
    """Commercial microwave link (CML) class for all data processing 
    
    Attributes
    ----------
    data : pandas.DataFrame
           Dataframe that holds the data of one link 
        
    tx_rx_pairs : dict, optional
           Dictonary that holds the configuration of the link including the
           transmission direction, the frequencies and the polarizations 
        
    metadata : dict, optional
        Dictonary that holds metadata about the link and the two associated sites
        
        
    """
    
    def __init__(self, 
                 data, 
                 tx_rx_pairs=None, 
                 metadata=None,
                 const_TX_power=False):
                     
        """Initialisation of commercial microwave link (CML) class
    
        Parameters
        ----------            
        data : pandas.DataFrame
               Dataframe that holds the data of the link. 
               The type of the link (Simplex/Duplex) and the associated transmission
               directions are automatically extracted from the dataframe's column names.
               The name should contain 'RX' or 'rx' for the received signal level
               and 'TX' or 'tx' for tranmitted signal level. For each site (far end,
               near end) entries can exists. The naming should be clear without
               ambiguity and conform to one of the conventions 
               
               - 'tx_far'          = TX level far end 
               - 'TX_near'         = TX level near end
               - 'rx_A'            = RX level far end
               - 'RX_B'            = RX level near end
               -  ...
               -  ...                  
               
               If the automatic extraction fails because the collerations can not 
               be recognized from the column names, the configuration can be specified 
               with the tx_rx_pairs dictionary.
               Further columns can be present in the DataFrame, 
               e.g. RTT (the round trip time of a SNMP data acquisition request).
            
        tx_rx_pairs : dict, optional
               Dictonary that holds the configuration of the link including the
               transmission direction, the frequencies and the polarizations.
               The Dictonary defines which TX and RX values belong together, if the
               automatic recognition fails, and which frequency and polarization are used.
                             
               Example
               -------
               >>> tx_rx_pairs =  {'fn': {'name': 'far-near', 
                                          'tx': 'tx_far'       # Column name in DataFrame
                                          'rx': 'rx_near',     # Column name in DataFrame
                                          'tx_site': 'site_B',
                                          'rx_site': 'site_A',
                                          'f_GHz': 17.8,
                                          'pol': 'V',
                                          'linecolor': 'r'},
                                    'nf': {'name': 'near-far',
                                          'tx': 'tx_near',     # Column name in DataFrame
                                          'rx': 'rx_far',      # Column name in DataFrame
                                          'tx_site': 'site_A',
                                          'rx_site': 'site_B',
                                          'f_GHz': 18.8,
                                          'pol': 'V',
                                          'linecolor': 'b'}}
                                     
               The keys, here 'fn' and 'nf' can be named arbitrarily but should be 
               clear without ambiguity. 
               
               Here 'fn' is short for far-near, but the use of AB to indicate the 
               connection from site A to site B would be appropriate, too.                                     
        
        metadata : dict, optional
            Dictonary that holds metadata about the link and the two associated sites.
            The dictonary must have two keys for the two MW link sites. Each item holds
            another dict with at least 'lat' and 'lon' values in decimal format.
            Further keys, like 'ID' or 'site_name' are possible but not mandatory.
            If the 'lat' and 'lon' values are not supplied, the geolocating 
            functions do not work of course.
            Additionally the key 'length_km', that holds the link's length in kilometer,
            should be appended.
            
            Example
            -------            
            >>> metadata = {'site_A': {'lat': 21.23,
                                       'lon': 3.24,
                                       'id': 'MY1231',
                                       'ip': '127.0.0.1',
                                       'slot': 2},
                            'site_B': {'lat': -2.123,
                                       'lon': -12.31,
                                       'id': 'MY1232',
                                       'ip': '127.0.0.2',
                                       'slot': 3},
                            'link_id': 'MY1231_2_MY1232_3',
                            'length_km': 23.4}
                         
        const_TX_power : int or float
            Constant transmittion power level. Should be provided if the link is
            operated this way and the dataframe only holds one column with 
            received power level.                  
                         
        """             
        
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
        for pair_id, column_names in tx_rx_pairs.iteritems():
            for key in column_names:
                
                if key == 'TX' or key == 'tx':
                    tx_series = self.data[tx_rx_pairs[pair_id][key]]
                elif key == 'RX' or key == 'rx':
                    rx_series = self.data[tx_rx_pairs[pair_id][key]]
                
            self.data['txrx_' + pair_id] = tx_series - rx_series                                          
                                         
        self.processing_info['tx_rx_pairs'] = tx_rx_pairs

    def info(self):
        """Print information about the microwave link
       
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


    def info_plot(self,out_file=None):
        """Print information about the link and plot it on a low resolution map 
                
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
        print '  L: ' + str(self.metadata['length_km']) + ' km'

        if 'site_A' in self.metadata and 'site_B' in self.metadata:
            if 'lat' in self.metadata['site_A'] and \
               'lon' in self.metadata['site_A'] and \
               'lat' in self.metadata['site_B'] and \
               'lon' in self.metadata['site_B']:                
                   print ' Lat: ' + str(self.metadata['site_A']['lat'])+ '                      '  \
                          + str(self.metadata['site_B']['lat'])   
                   print ' Lon: ' + str(self.metadata['site_A']['lon']) + '                     '  \
                          + str(self.metadata['site_B']['lon']) 
                          
                   area=[min(self.metadata['site_A']['lon'],self.metadata['site_B']['lon'])-.15,
                         max(self.metadata['site_A']['lon'],self.metadata['site_B']['lon'])+.15,
                         min(self.metadata['site_A']['lat'],self.metadata['site_B']['lat'])-.1,
                         max(self.metadata['site_A']['lat'],self.metadata['site_B']['lat'])+.1]
                         
                   plt.figure(figsize=(12, 8))
                   ax = plt.axes(projection=cartopy.crs.PlateCarree())
                   ax.set_extent((area[0], area[1], area[2], area[3]), crs=cartopy.crs.PlateCarree())
                   gg_tiles=img_tiles.OSM(style='satellite')
                   ax.add_image(gg_tiles, 11)

                   plt.plot([self.metadata['site_A']['lon'],self.metadata['site_B']['lon']],
                            [self.metadata['site_A']['lat'],self.metadata['site_B']['lat']],
                            linewidth=3,color='k',
                            transform=cartopy.crs.Geodetic())
                   xy= mapping.label_loc(self.metadata['site_A']['lon'],
                                        self.metadata['site_A']['lat'],
                                        self.metadata['site_B']['lon'],
                                        self.metadata['site_B']['lat'])

                   plt.text(xy[0],xy[1],'A',fontsize=15,transform=cartopy.crs.Geodetic()) 
                   plt.text(xy[2],xy[3],'B',fontsize=15,transform=cartopy.crs.Geodetic())  
                   #plt.tight_layout()
                   #plt.show() 
                   
        print '============================================================='                   

        if out_file is not None:
            plt.savefig(out_file,format='png',bbox_inches='tight', dpi=300) 
   
    def quality_test(self,rx_range=[-85,-10],tx_range=[-6,35],figsize=(6,4)):
        """Perform quality tests of TX, RX time series and print information
        
        Parameters
        ----------
        rx_range : list, optional
            List of lower and upper limit of plausible RX values in dBm. 
            Default is [-75,-10]
        tx_range : list, optional
            List of lower and upper limit of plausible TX values in dBm. 
            Default is [-6,35]    
        figsize : matplotlib parameter, optional 
            Size of output figure in inches (default is (6,4))   
            
        Note
        ----
        WIP : Currently three simple tests are implemented:
                - Outlier test: Test for TX/RX values outside of a specified range
                - Plausability test: Test if hourly rolling mean is within the 
                                     specified range.
                - Completeness test: Counts the number of data gaps                     
                   
        """
        
        for pair_id, column_names in self.processing_info['tx_rx_pairs'].iteritems():
            
            for key in column_names:
                
                if key == 'rx' or key == 'RX':
                    limit = rx_range
                elif key == 'tx' or key == 'TX':
                    limit = tx_range
                    
                if key == 'rx' or key == 'tx' or key == 'RX' or key == 'TX':
                    
                    column = self.processing_info['tx_rx_pairs'][pair_id][key]
                    
                    print '------'
                    print ' - ' +column+ ':'
                    
                    #Outlier test
                    outlier = self.data[column][(self.data[column] < limit[0]) | 
                                                (self.data[column] > limit[1])]
                                             
                    if len(outlier) == 0:            
                        print '    Outlier test passed.'
                        self.processing_info['outlier_test'+str(column)] = 'passed'   
                    else:
                        self.processing_info['outlier_test'+str(column)] = 'failed'
                        self.processing_info['outlier'+str(column)] = outlier
                        print '    Outlier test failed. Consider method remove_bad_values.'
                        plt.figure(figsize=figsize)
                        plt.ylabel(column)
                        if key == 'rx' or key == 'RX':                        
                            plt.fill_between(self.data.index,rx_range[0], 
                                             rx_range[1],color='g', alpha=0.2)
                            plt.ylim((-300,100))                  
                        elif key == 'tx' or key == 'TX':                        
                            plt.fill_between(self.data.index,tx_range[0], 
                                             tx_range[1],color='g', alpha=0.2)                                             
                            plt.ylim((-100,300))                  
                        plt.plot(self.data.index,self.data[column],'k')
                        plt.plot(outlier.index,outlier,'r.')        
                        plt.show()  
                        
                    #Plausability test
                    rol_mean = pd.rolling_mean(self.data[column],
                                                window=60, center=True)    
                    rol_mean_in=rol_mean                                         
                    rol_mean_out=rol_mean  
                                                        
                    rol_mean_out[(rol_mean > limit[0]) & 
                                 (rol_mean < limit[1])] = None
                    rol_mean_in[(rol_mean < limit[0]) | 
                                (rol_mean > limit[1])] = None   
                                                 
                    if ((rol_mean < limit[0]) | (rol_mean > limit[1])).any(): 
                        self.processing_info['plausability_test'+str(column)] = 'failed'
                        print "    Plausability test failed. Consider method remove_bad_values or check for protection link."
                        plt.figure(figsize=figsize)
                        plt.ylabel(column)
                        if key == 'rx' or key == 'RX':                        
                            plt.fill_between(self.data.index,rx_range[0], 
                                             rx_range[1],color='g', alpha=0.2)
                            plt.ylim((-300,100))                  
                        elif key == 'tx' or key == 'TX':                        
                            plt.fill_between(self.data.index,tx_range[0], 
                                             tx_range[1],color='g', alpha=0.2)                                             
                            plt.ylim((-100,300))                  
                        plt.plot(self.data.index,self.data[column],'k')
                        plt.plot(rol_mean_in.index,rol_mean_in,'g',linewidth=2)   
                        plt.plot(rol_mean_out.index,rol_mean_out,'r',linewidth=2)
                        plt.show()                                                  
                        
                    else:
                        print '    Plausability test passed.'
                        self.processing_info['plausability_test'+str(column)] = 'passed'        
     
        print '-------------------' 
        print 'Completeness test:'
        print 'Number of data gaps'
        count_nan = self.data.isnull().sum()
        print count_nan
        self.processing_info['data_gap_number'] = count_nan
        
    def remove_bad_values(self,bad_value=-99.9):
        """Detect bad values and convert to NaN
        
        Parameters
        ----------
        bad_value : int or float
                Bad value to be removed
        
        """

        for pair_id, column_names in self.processing_info['tx_rx_pairs'].iteritems():
            for key in column_names:
                
                if key == 'rx' or key == 'tx' or key == 'RX' or key == 'TX':
                    self.data[self.processing_info['tx_rx_pairs']
                                [pair_id][key]].replace(bad_value,np.nan,inplace=True)                
                
                if key == 'TX' or key == 'tx':
                    tx_series = self.data[self.processing_info['tx_rx_pairs']
                                            [pair_id][key]]
                elif key == 'RX' or key == 'rx':
                    rx_series = self.data[self.processing_info['tx_rx_pairs']
                                            [pair_id][key]]
                
            #Recalculate TXRX
            self.data['txrx_' + pair_id] = tx_series - rx_series     
        
   
   
   
    def plot(self, 
             param_list=['txrx'], 
             resampling_time=None, 
             add_raw_data=False,
             figsize=(6,4),
             **kwargs):
                 
        """Generic function to plot times series of data
        
        Parameters
        ----------        
        param_list : str, list of str, tuple of str, list of tuple of str, optional
            String or ensemble of string which give the parameters that 
            will be plotted (Default is 'txrx', which will plot transmitted power 
            minus received power).
        resampling_time : pandas parameter, optional
            resampling the raw data
        add_raw_data : bool, optional
            Not used at the moment (Default False)
        figsize : matplotlib parameter, optional 
            Size of output figure in inches (default is (6,4))
        kwargs :  matplotlib parameters, optional
        
        """

        if resampling_time is not None:
            df_temp = self.data.resample(resampling_time, label='right').mean()
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

                        # if there is no name supplied, usde the tx_rx_pair id
                        try:
                            name = self.processing_info['tx_rx_pairs']\
                                                   [txrx_pair_id]\
                                                   ['name']
                        except KeyError:
                            name = txrx_pair_id

                        # if there is no color supplied, let matplotlib decide
                        try:
                            color = self.processing_info['tx_rx_pairs']\
                                                    [txrx_pair_id]\
                                                    ['linecolor']
                            df_temp[param_temp].plot(ax=ax[i][0],
                                                     label=name,
                                                     color=color,
                                                     **kwargs)
                        except KeyError:
                            df_temp[param_temp].plot(ax=ax[i][0],
                                                     label=name,
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
                                            
        """Perform wet/dry classification for CML time series

        Parameters
        ----------
        method : str, optional
                 String which indicates the classification method (see Notes)
                 Default is 'std_dev'
        window_length : int, optional
                 Length of the sliding window (Default is 128)        
        threshold : int, optional
                 Threshold which has to be surpassed to classifiy a period as 'wet'
                 (Default is 1)        
        dry_window_length : int, optional
                 Length of window for identifying dry period used for calibration
                 purpose. Only for method stft. (Default is 600)
        f_divide : float
                 Parameter for classification with method Fourier transformation
                 (Default is 1e-3)    
        reuse_last_Pxx : bool
                 Parameter for classification with method Fourier transformation
                 (Default is false)  
        print_info : bool
                  Print information about executed method (Default is False)
        
        Note
        ----        
        WIP : Currently two classification methods are supported:
                - std_dev: Rolling standard deviation method [1]_
                - stft: Rolling Fourier-transform method [2]_      
                
        References
        ----------
        .. [1] Schleiss, M. and Berne, A.: "Identification of dry and rainy periods 
                using telecommunication microwave links", IEEE Geoscience and 
                Remote Sensing, 7, 611-615, 2010
        .. [2] Chwala, C., Gmeiner, A., Qiu, W., Hipp, S., Nienaber, D., Siart, U.,
              Eibert, T., Pohl, M., Seltmann, J., Fritz, J. and Kunstmann, H.:
              "Precipitation observation using microwave backhaul links in the 
              alpine and pre-alpine region of Southern Germany", Hydrology
              and Earth System Sciences, 16, 2647-2661, 2012        
        
        """
        
        # Standard deviation method
        if method == 'std_dev':
            if print_info:
                print 'Performing wet/dry classification for link '+ self.metadata['link_id']
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
                print 'Performing wet/dry classification for link '+ self.metadata['link_id']
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
                                      
        """Perform baseline determination for CML time series
        
        Parameters
        ----------
        method : str, optional
                 String which indicates the baseline determination method (see Notes)
                 Default is 'constant'
        wet_external : iterable of int or iterable of float, optional
                 External wet/dry classification information. Has to be specified
                 for each classified index of times series (Default is None)        
        print_info : bool
                 Print information about executed method (Default is False)
                 
        Note
        ----
        WIP: Currently two methods are supported:
             - constant: Keeping the RSL level constant at the level of the preceding dry period
             - linear: interpolating the RSL level linearly between two enframing dry periods
             
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

        """Perform baseline adjustion due to wet antenna for CML time series
        
        Parameters
        ----------
        waa_max : float
                  Maximum value of wet antenna attenuation   
        delta_t : float
                  Parameter for wet antnenna attenation model    
        tau : float
              Parameter for wet antnenna attenation model 
        
        wet_external : iterable of int or iterable of float, optional
                 External wet/dry classification information. Has to be specified
                 for each classified index of times series (Default is None)
        
        Note
        ----        
        The wet antenna adjusting is based on a peer-reviewed publication [3]_
                
        References
        ----------
        .. [3] Schleiss, M., Rieckermann, J. and Berne, A.: "Quantification and
                modeling of wet-antenna attenuation for commercial microwave
                links", IEEE Geoscience and Remote Sensing Letters, 10, 2013        
        
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
        
        """Perform calculation of attenuation for CML time series
        
        Parameters
        ----------
        remove_negative_A : bool
                Negative attenuation values are assigned as zero (Default is True)
               
        """           
        
        for pair_id in self.processing_info['tx_rx_pairs']:
            A = self.data['txrx_' + pair_id] - self.data['baseline_' + pair_id]
            if remove_negative_A:
                A[A < 0] = 0
            self.data['A_' + pair_id] = A
                
    def calc_R_from_A(self, a=None, b=None, approx_type='ITU'):
        
        """Perform calculation of rain rate from attenuation for CML time series
        
        Parameters
        ----------
        a : float, optional
            Parameter of A-R relationship (Default is None)
            If not given: Approximation considering polarization and frequency of link
        b : float, optional
            Parameter of A-R relationship  (Default is None)
            If not given: Approximation considering polarization and frequency of link    
        approx_type : str, optional
            Approximation type (the default is 'ITU', which implies parameter
            approximation using a table recommanded by ITU)     
               
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
    
    """Derive the TX-RX pairs from the MW link data columns names
    
    Right now, this only works for the following cases:
    
    1. A duplex link with TX and RX columns for each direction. The
       naming convention is 'TX_something', or 'tx_something',
       or 'RX_something', 'rx_something', where `something` is most commonly `far`/`near`
       or `A`/`B`.
    2. A simplex link with one TX and RX columns which also carry the site
       name after a '_' like in case 1.
    3. A simplex link with one TX and RX columns which are named 'TX' 
       and 'RX', or 'tx' and 'rx'.
    
    Parameters
    ----------    
    column_names : list
        List of columns names from the MW link DataFrame
        
    Returns
    -------
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
    
    
    
        
        