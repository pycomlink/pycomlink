#----------------------------------------------------------------------------
# Name:         comlinkset
# Purpose:      Commercial MW link Class to handle all processing steps for 
#                   a set of links
#
# Authors:      Christian Chwala, Felix Keis
#
# Created:      25.02.2015
# Copyright:    (c) Christian Chwala, Felix Keis 2015
# Licence:      The MIT License
#----------------------------------------------------------------------------

from __future__ import division
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import cartopy
from cartopy.io import img_tiles
import pandas as pd
from scipy.io import netcdf
<<<<<<< HEAD:pycomlink/death_row/comlinkset.py
=======
from collections import OrderedDict
import random
>>>>>>> master:pycomlink/comlinkset.py
import matplotlib

from pycomlink.death_row import mapping


class ComlinkSet():
    """Commercial microwave link set (CMLS) class for data processing of multiple links
    
    Attributes
    ----------
    set : list of pycomlink class Comlink objects
    set_info : dict
          Dictionary that holds information about the CMLS object
        
    """  
    
    def __init__(self, cml_list,area,start,stop):
        """Initialisation of commercial microwave link set (CMLS) class 
           
            Parameters
            ----------
            cml_list : list of pycomlink class Comlink objects
                List that holds the links located in a specified area. Each list item
                has to be a Comlink object.
            area : list of floats
                List that holds the geographic coordinates of the area borders in
                decimal format. The order is mandatory.
                
                Example
                -------
                >>> area = [lower_longitude,upper_longitude,lower_latitude, upper_latitude]                                
            start : str
                Start Timestamp of requested period
            stop : str
                Stop Timestamp of requested period                      
        """
        
        
        self.set = cml_list
        self.set_info = {'area':area, 'start':start,'stop':stop}
        
    def info(self):
        """Print information about associated microwave links 
        
        """        
        
        print '============================================================='
        print 'Number of Comlink Objects in ComlinkSet: ' + str(len(self.set))
        print '  ----- '+"{:.2f}".format(self.set_info['area'][3])+' -----'
        print '  |               |'
        print "{:.2f}".format(self.set_info['area'][0])+'   area    ' \
                +"{:.2f}".format(self.set_info['area'][1])
        print '  |               |'        
        print '  ----- '+"{:.2f}".format(self.set_info['area'][2])+' -----'        

        print 'IDs: ' 
        for cml in self.set:
            print '     ' + str(cml.metadata['link_id'])
        print '============================================================='
        

    def info_plot(self,out_file=None,figsize=(12,8), add_labels=False):
        """Plot associated links on a map 
                
        """
        plt.figure(figsize=figsize)
        ax = plt.axes(projection=cartopy.crs.PlateCarree())

        lons=[]
        lats=[]
        for cml in self.set:
            lons.append(cml.metadata['site_A']['lon'])
            lons.append(cml.metadata['site_B']['lon'])
            lats.append(cml.metadata['site_A']['lat'])
            lats.append(cml.metadata['site_B']['lat'])   
            
        area=[min(lons)-.05,
              max(lons)+.05,
              min(lats)-.05,
              max(lats)+.05]           
        
        ax.set_extent((area[0], area[1], area[2], area[3]), crs=cartopy.crs.PlateCarree())
        gg_tiles=img_tiles.OSM()
        ax.add_image(gg_tiles, 11)        
        
        for cml in self.set:
                   plt.plot([cml.metadata['site_A']['lon'],cml.metadata['site_B']['lon']],
                            [cml.metadata['site_A']['lat'],cml.metadata['site_B']['lat']],
                            linewidth=2,color='k',
                            transform=cartopy.crs.Geodetic())
                   if add_labels:
                        plt.text(cml.metadata['site_A']['lon'] - (cml.metadata['site_A']['lon'] - cml.metadata['site_B']['lon'])/2,
                             cml.metadata['site_A']['lat'] - (cml.metadata['site_A']['lat'] - cml.metadata['site_B']['lat'])/2,
                             cml.metadata['link_id'],
                             transform=cartopy.crs.Geodetic(),
                             fontsize=15,
                             color='r')

        if out_file is not None:
            plt.savefig(out_file,format='png',bbox_inches='tight', dpi=300) 


    def quality_test(self,rx_range=[-80,-10],tx_range=[-6,35],figsize=(6,4)):
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
                   
        """
        for cml in self.set:
            cml.quality_test(rx_range,tx_range,figsize)


    def remove_bad_values(self,bad_value=-99.9):
        """Detect bad values and convert to NaN
        
        Parameters
        ----------
        bad_value : int or float
                Bad value to be removed
        
        """
        for cml in self.set:
            cml.remove_bad_values(bad_value)


    def find_neighboring_links(self,crit_dis=10.,min_link_length=0.7):
        """Identify neighboring links for which both ends are within a critical distance from either end of a link (computation is done for all links).
        
        Parameters
        ----------
        crit_dis : float
                critical distance
        min_link_length : float
                Only links with a length larger than min_link_length are considered 
                
        Note
        ----
        Requires Metadata dictionaries for all involved links with at 
        least minimum information (see example for key naming)     
        
        Example
        -------            
        >>> metadata = {'site_A': {'lat': 21.23,
                                       'lon': 3.24},
                        'site_B': {'lat': -2.123,
                                       'lon': -12.31},
                        'link_id': 'MY1231_2_MY1232_3'}
        """
        for cml in self.set:
            id_list = []
            cml_list = []    
            if cml.metadata['length_km'] > min_link_length:
                id_list.append(cml.metadata['link_id'])
                cml_list.append(cml)
                for cml2 in self.set:
                    if cml.metadata['link_id'] != cml2.metadata['link_id'] and \
                       mapping.distance((cml.metadata['site_A']['lat'],
                                 cml.metadata['site_A']['lon']),
                                (cml2.metadata['site_A']['lat'],
                                 cml2.metadata['site_A']['lon'])) < crit_dis and \
                       mapping.distance((cml.metadata['site_A']['lat'],
                                 cml.metadata['site_A']['lon']),
                                (cml2.metadata['site_B']['lat'],
                                 cml2.metadata['site_B']['lon'])) < crit_dis and \
                       mapping.distance((cml.metadata['site_B']['lat'],
                                 cml.metadata['site_B']['lon']),
                                (cml2.metadata['site_A']['lat'],
                                 cml2.metadata['site_A']['lon'])) < crit_dis and \
                       mapping.distance((cml.metadata['site_B']['lat'],
                                 cml.metadata['site_B']['lon']),
                                (cml2.metadata['site_B']['lat'],
                                 cml2.metadata['site_B']['lon'])) < crit_dis and \
                       cml2.metadata['length_km'] > min_link_length: 
                                     id_list.append(cml2.metadata['link_id'])
                                     cml_list.append(cml2)
        
            cml.processing_info['neighbors'] = id_list
            cml.processing_info['neighbors_cml'] = cml_list
            cml.processing_info['crit_dis'] = crit_dis    
    
    
    def do_wet_dry_classification(self, method='std_dev', 
                                        window_length=128,
                                        threshold=1,
                                        dry_window_length=600,
                                        f_divide=1e-3,
                                        reuse_last_Pxx=False,
                                        number_neighbors=2,
                                        deltaP=-1.4,
                                        deltaPL=-0.7,
                                        deltaP_max=-2.0, 
                                        print_info=False):
        """Perform wet/dry classification for all CML time series in CMLS

        Parameters
        ----------
        method : str, optional
                 String which indicates the classification method (see Notes)
                 Default is 'std_dev'
        window_length : int, optional
                 Length of the sliding window (Default is 128)        
        threshold : int, optional
                 threshold which has to be surpassed to classifiy a period as 'wet'
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
        number_neighbors : int, optional
                 Minimum number of neighboring links for method link approach       
        deltaP : float, optional
                 Threshold value (in dB) for mutual decrease in minimum RSL
                 of neighboring links
        deltaPL : float, optional
                 Threshold value (in dB/km) for mutual decrease in RSL           
                 of neighboring links  
        deltaP_max : float, optional
                 Big threshold value (in dB) for mutual decrease in minimum RSL
                 of neighboring links                                   
        print_info : bool
                  Print information about executed method (Default is False)
        
        Note
        ----        
        WIP: Currently two classification methods are supported:
            - std_dev: Rolling standard deviation method [1]_
            - stft: Rolling Fourier-transform method [2]_   
            - link_appr: Link approach [5]_. Temporal resolution is set to 15 Minutes. Links without neighbors are dismissed.
                                             
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
        .. [5] Overeem, A. and Leijnse, H. and Uijlenhoet R.: "Measuring urban  
                rainfall using microwave links from commercial cellular 
                communication networks", Water Resources Research, 47, 2011                      
        
        """
        
        
        if method == 'std_dev' or method == 'stft':
            for cml in self.set:    
                cml.do_wet_dry_classification(method, 
                                                  window_length,
                                                  threshold,
                                                  dry_window_length,
                                                  f_divide,
                                                  reuse_last_Pxx,
                                                  print_info)
            
        elif method == 'link_appr':
            if print_info:
                print 'Performing wet/dry classification with link approach'
                print 'Method = link_appr'
                print 'Critical distance = ' + str(self.set[0].processing_info['crit_dis'])
                print '-----------------------------------------'
                print 'Hint:'
                print 'Temporal resolution is set to 15 Minutes'
                print 'Links without neighbors are dismissed'

            
            cml_list_update = []  
            for cml in self.set:  
                                  
                if len(cml.processing_info['neighbors']) >= number_neighbors+1: 
                    data_min = pd.DataFrame()
                    dp = pd.DataFrame()
                    dpl = pd.DataFrame()
                    for pair_id in cml.processing_info['tx_rx_pairs']:
                        for cml_nb in cml.processing_info['neighbors_cml']:
                            if pair_id in cml_nb.processing_info['tx_rx_pairs']: 
                                data_min['txrx_'+pair_id] = cml_nb.data['txrx_'+pair_id].resample('15min').min()
                                dp[cml_nb.metadata['link_id']] = data_min['txrx_'+pair_id] - \
                                                                 pd.rolling_max(data_min['txrx_'+pair_id],96)
                                dpl[cml_nb.metadata['link_id']] = (data_min['txrx_'+pair_id] - \
                                                                   pd.rolling_max(data_min['txrx_'+pair_id],96))/ \
                                                                   cml.metadata['length_km']            
                        
                        data_min['wet_' + pair_id] = ((dp.median(axis=1) < deltaP) & \
                                                         (dpl.median(axis=1) < deltaPL))
                                                         
                        cond = (data_min['wet_' + pair_id]) & (dp[cml.metadata['link_id']] < deltaP_max)
                        cond_shifted_plus1 = cond.shift(1)
                        cond_shifted_minus1= cond.shift(-1)
                        cond_shifted_minus2= cond.shift(-2)
        
                        cond_shifted_plus1[(cond == True) & (cond.shift(1) == False)] = True
                        cond_shifted_minus1[(cond == True) & (cond.shift(-1) == False)] = True
                        cond_shifted_minus2[(cond == True) & (cond.shift(-2) == False)] = True
                        data_min['wet_' + pair_id][(cond_shifted_plus1 == True) | \
                                                    (cond_shifted_minus1 == True) | \
                                                    (cond_shifted_minus2 == True)] = True       


                              
                    cml.data = data_min 
            
                    cml_list_update.append(cml) 
                            
            self.set = cml_list_update          
  
        else:
                ValueError('Wet/dry classification method not supported')            
                
                

    def do_baseline_determination(self, 
                                  method='constant',
                                  wet_external=None,
                                  print_info=False):
                                      
        """Perform baseline determination for all CML time series in CMLS
        
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

                              
        for cml in self.set:   
            cml.do_baseline_determination(method,
                                              wet_external,
                                              print_info)
  


    def do_wet_antenna_baseline_adjust(self,
                                       waa_max, 
                                       delta_t, 
                                       tau,
                                       wet_external=None):
                                           
        """Perform baseline adjustion due to wet antenna for all CML time series in CMLS
        
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
                                           
        for cml in self.set:    
            cml.do_wet_antenna_baseline_adjust(waa_max,
                                                   delta_t,
                                                   tau,
                                                   wet_external)
                
 
    def calc_A(self, remove_negative_A=True):
        
        """Perform calculation of attenuation for all CML time series in CMLS
        
        Parameters
        ----------
        remove_negative_A : bool
                Negative attenuation values are assigned as zero (Default is True)
               
        """         
        
        for cml in self.set:
            cml.calc_A(remove_negative_A)
 
                    
                    
    def calc_R_from_A(self, a=None, b=None, approx_type='ITU'):
        
        """Perform calculation of rain rate from attenuation for all CML time series in CMLS . 
        
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
            
        Note
        ----
        cml without length in metadata and frequency information in processing 
        info are removed from cmls.set
        """   

        for cml in self.set:
            remove=False
            if not cml.metadata['length_km']:
                remove=True
            for pair_id in cml.processing_info['tx_rx_pairs']:
                if not cml.processing_info['tx_rx_pairs'][pair_id]['f_GHz']:
                    remove=True
                elif np.isnan(cml.processing_info['tx_rx_pairs'][pair_id]['f_GHz']):
                    remove=True
            if remove:
                self.set.remove(cml)        
        
        for cml in self.set:
                cml.calc_R_from_A(a,b,approx_type)     
                                                                    
               
    def spat_interpol(self,
                      int_type,
                      grid_res=None,
                      longrid=None,
                      latgrid=None,
                      figsize=(15,15),
                      resampling_time='H',
                      start_time=None,
                      stop_time=None,
                      method='mean',
                      power=2,smoothing=0,nn=None,
                      print_info=False,
                      **kwargs):
                     
        """ Perform spatial interpolation of rain rate grid
        
        Parameters
        ----------   
        int_type : str
            interpolation method
            'IDW' Inverse Distance Interpolation
            'Kriging' Kriging Interpolation with pyKrige         
        grid_res : int,optional
            Number of bins of output grid in area 
            Default is None
        longrid : iterable of floats
            Grid (2D Array) for longitude values. Default is None
        latgrid : iterable of floats
            Grid (2D Array) for latitude values. Default is None.            
        figsize : matplotlib parameter, optional 
            Size of output figure in inches (default is (15,15))      
        resampling_time : pandas parameter, optional
            resampling the raw data of each cml   
        method : str, optional
            Indicates how to claculate rainrate/sum of bidirectional link
            (Default is 'mean')
            'mean' average of near-far and far-near
            'max' maximum of near-far and far-near
            'min' minimum of near-far and far-near
            'nf', 'fn', 'fnp' etc. use direction from tx_rx_pairs                           
        power : flt, optional
               Power of distance decay for IDW interpolation. Only used if 
               int_type is 'IDW' (Default is 2)   
        smoothing : flt, optional
               Power of smoothing factor for IDW interpolation. Only used if 
               int_type is 'IDW' (Default is 0) 
        nn : int,optional
                Number of neighbors considered for interpolation. If None all
                neighbors are used
                Default is None
        kwargs : kriging parameters, optional
                See https://github.com/bsmurphy/PyKrige for details          
                
        """
        
        if longrid is None or latgrid is None:
            if not grid_res is None:
                #Definition of output grid
                gridx = np.linspace(self.set_info['area'][0],self.set_info['area'][1],grid_res)
                gridy = np.linspace(self.set_info['area'][2],self.set_info['area'][3],grid_res)   
                longrid,latgrid = np.meshgrid(gridx, gridy)
            else:
                ValueError('Either longrid & latgrid or grid_res have to be provided')

        
        self.set_info['interpol_longrid'] = longrid
        self.set_info['interpol_latgrid'] = latgrid
        
        if start_time is None or stop_time is None:
            times = pd.date_range(self.set_info['start'],self.set_info['stop'],
                                  freq=resampling_time)[0:-1]
        else:
            times = pd.date_range(start_time,stop_time,
                                  freq=resampling_time)

        self.set_info['interpol_time_array'] = times
        self.set_info['interpol'] = OrderedDict()

        temp_df_list = []
        for cml in self.set:
            temp_df_list.append(cml.data.resample(resampling_time, label='right').mean())

        meas_points_old = np.empty(0)

        for i_time, time in enumerate(times):
            if print_info:
                print "Interpolating for UTC time",time
            lons_mw=[]
            lats_mw=[]
            values_mw=[]     
            for cml, temp_df in zip(self.set, temp_df_list):
                if 'site_A' in cml.metadata and 'site_B' in cml.metadata:
                    if 'lat' in cml.metadata['site_A'] and \
                       'lon' in cml.metadata['site_A'] and \
                       'lat' in cml.metadata['site_B'] and \
                       'lon' in cml.metadata['site_B']:
                             
                           lat_center = (cml.metadata['site_A']['lat']
                                           +cml.metadata['site_B']['lat'])/2.
                           lon_center = (cml.metadata['site_A']['lon']
                                           +cml.metadata['site_B']['lon'])/2. 
    
    
                           start = pd.Timestamp(time) - pd.Timedelta('10s')
                           stop = pd.Timestamp(time) + pd.Timedelta('10s')
                           plist = []
                           if method in ['mean','max','min']:
                               for pair_id in cml.processing_info['tx_rx_pairs']:
                                   # TODO: Get rid of the [0] indexing and the start stop index thing
                                   R_temp = (temp_df['R_'+pair_id][start:stop])
                                   if len(R_temp) > 0:
                                       plist.append(R_temp.values[0])
                                   else:
                                       plist.append(np.nan)
                                 
                               if method == 'mean':
                                   precip = np.mean(plist)                     
                               elif method == 'max':
                                   precip = np.max(plist)                        
                               elif method == 'min':
                                   precip = np.min(plist)                          
                           elif method in cml.processing_info['tx_rx_pairs']:
                               if 'R_' + method in temp_df.keys():
                                   precip = (temp_df['R_'+method][start:stop]).values[0]
                               else:
                                   print 'Warning: Pair ID '+method+' not available for link '+\
                                             cml.metadata['link_id'] + ' (link is ignored)'
                                   precip = None
                           else:
                               print 'Error: '+method+' not available for link '+\
                                         cml.metadata['link_id'] + ' (link is ignored)'
                               print '      Use "mean","max","min" or pair_id'        
                               precip = None
                           
                          
                           lons_mw.append(lon_center)
                           lats_mw.append(lat_center)
                           if precip >= 0.:
                               values_mw.append(precip)  
                           else:
                               values_mw.append(np.nan)    

             
            val_mw=np.ma.compressed(np.ma.masked_where(np.isnan(values_mw),values_mw))
            lon_mw=np.ma.compressed(np.ma.masked_where(np.isnan(values_mw),lons_mw))
            lat_mw=np.ma.compressed(np.ma.masked_where(np.isnan(values_mw),lats_mw))

            meas_points=np.vstack((lon_mw,lat_mw)).T
            xi, yi = longrid.flatten(), latgrid.flatten()
            grid = np.vstack((xi, yi)).T
            
            if int_type == 'IDW':
                # Check if IDW weights have to be calculated or if
                # they can be reused
                must_calc_new_weights = False
                if i_time == 0:
                    must_calc_new_weights = True
                if not np.array_equal(meas_points_old, meas_points):
                    if print_info:
                        print 'meas_points not equal to meas_points_old at %s UTC' % str(time)
                    must_calc_new_weights = True
                    meas_points_old = meas_points

                if must_calc_new_weights:
                    idw_weights = mapping._get_idw_weights(meas_points,grid,
                                       power,smoothing, nn)
                interpol = mapping.inv_dist(meas_points,val_mw,grid,
                                       power,smoothing,nn, idw_weights).reshape(longrid.shape)
                                      
            elif int_type == 'Kriging':
                interpol= mapping.kriging(meas_points,val_mw,grid,
                                         nn,**kwargs).reshape(longrid.shape)   
                                         
            else:
                ValueError('Interpolation method not supported')                             
            

            self.set_info['interpol'][(pd.Timestamp(time)).strftime('%Y-%m-%d %H:%M')] = interpol
        

    def plot_spat_interpol(self,time,
                      figsize=(15,15),
                      OSMtile=False,
                      out_file=None):
        """Plot spatial interpolation of rain rate 
        
        Parameters
        ----------
        time : str, optional
            Datetime string of desired timestamp (for example 'yyyy-mm-dd HH:MM')
        figsize : matplotlib parameter, optional 
            Size of output figure in inches (default is (15,15)) 
        OSMtile : bool, optional
                 Use OpenStreetMap tile as background (slows down the plotting)            
        out_file : str, optional
                file path of output image file
                (Default is None)
        """                          
                          
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=cartopy.crs.PlateCarree())
        gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')        
        gl.xlabels_top = False
        ax.set_extent((self.set_info['area'][0]-.05, self.set_info['area'][1]+.05,
                       self.set_info['area'][2]-.05, self.set_info['area'][3]+.05),
                         crs=cartopy.crs.PlateCarree())
        if OSMtile:                 
            gg_tiles = img_tiles.OSM()      
            ax.add_image(gg_tiles, 11)          
        
         
        nws_precip_colors = [
        "#04e9e7",  # 0.01 - 0.10 inches
        "#019ff4",  # 0.10 - 0.25 inches
        "#0300f4",  # 0.25 - 0.50 inches
        "#02fd02",  # 0.50 - 0.75 inches
        "#01c501",  # 0.75 - 1.00 inches
        "#008e00",  # 1.00 - 1.50 inches
        "#fdf802",  # 1.50 - 2.00 inches
        "#e5bc00",  # 2.00 - 2.50 inches
        "#fd9500",  # 2.50 - 3.00 inches
        "#fd0000",  # 3.00 - 4.00 inches
        "#d40000",  # 4.00 - 5.00 inches
        "#bc0000",  # 5.00 - 6.00 inches
        "#f800fd",  # 6.00 - 8.00 inches
        "#9854c6"  # 10.00+
        ]
    
        precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors) 
        levels_rr = [0.01,0.1,0.25,0.5, 1.0, 1.5,2.0, 2.5,3.0, 5.0, 7.5, 10.0,12.5,15.0,20.0] 
        
        for cml in self.set:
            if 'site_A' in cml.metadata and 'site_B' in cml.metadata:
                if 'lat' in cml.metadata['site_A'] and \
                   'lon' in cml.metadata['site_A'] and \
                   'lat' in cml.metadata['site_B'] and \
                   'lon' in cml.metadata['site_B']:
                       plt.plot([cml.metadata['site_A']['lon'],cml.metadata['site_B']['lon']],
                         [cml.metadata['site_A']['lat'],cml.metadata['site_B']['lat']],
                         linewidth=1,color='k',
                         transform=cartopy.crs.Geodetic()) 
                                               

        interpolm = np.ma.masked_less(self.set_info['interpol'][(pd.Timestamp(time)).strftime('%Y-%m-%d %H:%M')],0.01)
        norm = matplotlib.colors.BoundaryNorm(levels_rr, 14)
        cs = plt.pcolormesh(self.set_info['interpol_longrid'],self.set_info['interpol_latgrid'],
                            interpolm, norm=norm, cmap=precip_colormap,alpha=0.4)              
        cbar = plt.colorbar(cs,orientation='vertical', shrink=0.4)
        cbar.set_label('mm/h')            
        plt.title((pd.Timestamp(time)).strftime('%Y-%m-%d %H:%M')+'UTC',loc='right')
    
    
        plt.plot([self.set_info['area'][0],self.set_info['area'][0]],
                 [self.set_info['area'][2],self.set_info['area'][3]],linewidth=2,
                    color='k',alpha=0.6, transform=cartopy.crs.Geodetic())
        plt.plot([self.set_info['area'][0],self.set_info['area'][1]],
                 [self.set_info['area'][2],self.set_info['area'][2]],linewidth=2,
                    color='k',alpha=0.6, transform=cartopy.crs.Geodetic())
        plt.plot([self.set_info['area'][1],self.set_info['area'][1]],
                 [self.set_info['area'][2],self.set_info['area'][3]],linewidth=2,
                    color='k',alpha=0.6, transform=cartopy.crs.Geodetic())                    
        plt.plot([self.set_info['area'][0],self.set_info['area'][1]],
                 [self.set_info['area'][3],self.set_info['area'][3]],linewidth=2,
                    color='k',alpha=0.6, transform=cartopy.crs.Geodetic())
                    
        if out_file is not None:
            plt.savefig(out_file,bbox_inches='tight',format='png')


    def write_netcdf_wrf(self,nc_file):
        """Write spatial interpolated Rainrate to netcdf file (WRF-Hydro readable)

        Parameters
        ----------
        nc_file : string
            Name for output netcdf file
        """             
        
        f = netcdf.netcdf_file(nc_file, 'w')
        
        f.createDimension('Time', len(self.set_info['interpol_time_array']))
        f.createDimension('DateStrLen', 
                          len(self.set_info['interpol_time_array'][0].strftime('%Y-%m-%d_%H:%M:%S')))
        f.createDimension('west_east', self.set_info['interpol_longrid'].shape[0])
        f.createDimension('south_north', self.set_info['interpol_longrid'].shape[1])
        
        time_list=[]
        for time in self.set_info['interpol_time_array']:
            time_list.append(time.strftime('%Y-%m-%d_%H:%M:%S'))     
        TIME = f.createVariable('TIME', 'c', ('Time','DateStrLen',))
        TIME[:] = time_list
        
        XLAT_M = f.createVariable('XLAT_M', 'd', ( 'west_east','south_north',))
        XLAT_M[:] = self.set_info['interpol_latgrid']
        #XLAT_M[:] = np.stack([self.set_info['interpol_latgrid']]*len(self.set_info['interpol_time_array']))
        
        XLON_M = f.createVariable('XLON_M', 'd', ('west_east','south_north',))
        XLON_M[:] = self.set_info['interpol_longrid']
        #XLON_M[:] = np.stack([self.set_info['interpol_longrid']]*len(self.set_info['interpol_time_array']))
        
        rr_list=[]
        for time in self.set_info['interpol']:
            rr_list.append(self.set_info['interpol'][time])     
     
        RAINRATE = f.createVariable('RAINRATE', 'f', ('Time','west_east','south_north',))
        RAINRATE[:] = np.stack(rr_list)
        
        
        f.close()        

