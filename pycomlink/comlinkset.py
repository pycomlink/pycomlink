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
import numpy as np
import matplotlib.pyplot as plt
import cartopy
from cartopy.io import img_tiles
import pandas as pd


import matplotlib

from . import wet_dry
from . import baseline
from . import A_R_relation
from . import wet_antenna
from . import mapping


class ComlinkSet():
    """Commercial microwave link set (CMLS) class for data processing of multiple links
    
    Attributes
    ----------
    set : list of pycomlink class Comlink objects
    set_info : dict
          Dictionary that holds information about the CMLS object
        
    """  
    
    def __init__(self, cml_list,area):
        """Initialisation of commercial microwave link set (CMLS) class 
           
            Parameters
            ----------
            cml_list : list of pycomlink class Comlink objects
                List that holds the links located in a specified area. Each list item
                has to be a Comlink object.
            area : list of floats
                List that holds the geographic coordinates of the area borders in
                decimal format. 
                The order is mandatory.
                
                Example
                -------
                >>> area = [lower_longitude,upper_longitude,lower_latitude, upper_latitude]
                
        """
        
        
        self.set = cml_list
        self.set_info = {'area':area}
        
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
        

    def info_plot(self):
        """Plot associated links a map 
                
        """
        plt.figure(figsize=(10,10))
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


    def find_neighboring_links(self,crit_dis=10.):
        """Identify neighboring links for which both ends are within a critical
           distance from either end of a link (computation is done for all links)
        
        Parameters
        ----------
        crit_dis : float
                critical distance
                
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
                             cml2.metadata['site_B']['lon'])) < crit_dis:
                                 id_list.append(cml2.metadata['link_id'])
                                                
            cml.processing_info['neighbors'] = id_list
            cml.processing_info['crit_dis'] = crit_dis    
    
    
    def do_wet_dry_classification(self, method='std_dev', 
                                        window_length=128,
                                        threshold=1,
                                        dry_window_length=600,
                                        f_divide=1e-3,
                                        reuse_last_Pxx=False,
                                        number_neighbors=2,
                                        min_link_length=0.7,
                                        deltaP=-1.4,
                                        deltaPL=-0.7,
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
        min_link_length : float, optional
                 Minimum link length (in km) for method link approach        
        deltaP : float, optional
                 Threshold value (in dB) for mutual decrease in minimum RSL
                 of neighboring links
        deltaPL : float, optional
                 Threshold value (in dB/km) for mutual decrease in RSL           
                 of neighboring links                 
        print_info : bool
                  Print information about executed method (Default is False)
        
        Note
        ----        
        WIP: Currently two classification methods are supported:
                - std_dev: Rolling standard deviation method [1]_
                - stft: Rolling Fourier-transform method [2]_   
                - link_appr: Link approach [5]_. 
                             Temporal resolution is set to 15 Minutes.
                             Links without neighbors are dismissed.
                                             
        References
        ----------
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
                print 'Critical distance = ' + str(cml.processing_info['crit_dis'])
                print '-----------------------------------------'
                print 'Hint:'
                print 'Temporal resolution is set to 15 Minutes'
                print 'Links without neighbors are dismissed'
            for cml in self.set:
            # Needs minimum power over 15min intervals
                cml.data_mean = cml.data.resample('15min',how='mean')
                cml.data_min = pd.DataFrame()
                for pair_id in cml.processing_info['tx_rx_pairs']:
                    cml.data_min['txrx_'+pair_id] = cml.data['txrx_'+pair_id].resample('15min',how='min') 
            
            cml_list_update = []            
            for cml in self.set:                        
                if len(cml.processing_info['neighbors']) >= number_neighbors and \
                   cml.metadata['length_km'] > min_link_length: 
                       cml_list_update.append(cml)
                       for pair_id in cml.processing_info['tx_rx_pairs']:                           
                           temp_dp = cml.data_min['txrx_'+pair_id] - \
                                      pd.rolling_max(cml.data_min['txrx_'+pair_id],96)
                           dp = temp_dp.to_frame(cml.metadata['link_id'])       
                           temp_dpl = (cml.data_min['txrx_'+pair_id] - \
                                       pd.rolling_max(cml.data_min['txrx_'+pair_id],96))/ \
                                       cml.metadata['length_km']
                           dpl = temp_dpl.to_frame(cml.metadata['link_id'])  
                            
                           for cml2 in self.set:
                               if cml2.metadata['link_id'] in cml.processing_info['neighbors'] and \
                                   pair_id in cml2.processing_info['tx_rx_pairs'] and \
                                   cml2.metadata['length_km'] > min_link_length:    
                                       dp[cml2.metadata['link_id']] = \
                                        cml2.data_min['txrx_'+pair_id] - \
                                        pd.rolling_max(cml2.data_min['txrx_'+pair_id],96)   
                                       dpl[cml2.metadata['link_id']] = \
                                        cml2.data_min['txrx_'+pair_id] - \
                                        pd.rolling_max(cml2.data_min['txrx_'+pair_id],96)/ \
                                        cml2.metadata['length_km']   
          
                           cml.data_min['wet_' + pair_id] = ((dp.median(axis=1) < deltaP) & \
                                                         (dpl.median(axis=1) < deltaPL))
                           for i in range(2,len(cml.data_min)-1):  
                               if cml.data_min['wet_' + pair_id][i] and temp_dp[i] < -2.:
                                    cml.data_min['wet_' + pair_id][i-2] = True
                                    cml.data_min['wet_' + pair_id][i-1] = True
                                    cml.data_min['wet_' + pair_id][i+1] = True
                                    
                           cml.data_mean['wet_' + pair_id] = cml.data_min['wet_' + pair_id]   
                           
                cml.data = cml.data_mean                             
            self =  ComlinkSet(cml_list_update, self.set_info['area'],
                               self.set_info['start'],self.set_info['stop'])   
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
        
        """Perform calculation of rain rate from attenuation for all CML time series in CMLS
        
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
        for cml in self.set:
            for pair_id in cml.processing_info['tx_rx_pairs']:
                if cml.processing_info['tx_rx_pairs'][pair_id]['f_GHz'] is not None: 
                   cml.calc_R_from_A(a,b,approx_type)     
                else:
                   cml.data['R_'+pair_id] = None 
                
                                                           
               
    def spat_interpol(self, grid_res,
                 int_type,
                 figsize=(15,10),
                 time=None,
                 method='mean',
                 time_resolution=15,
                 power=2,smoothing=0,
                 krig_type='ordinary',
                 variogram_model='linear',
                 drift_terms=['regional_linear'],
                 out_file=None):
                     
        """ Perform and plot spatial interpolation of rain rate or rain sum on regular grid
        
        Parameters
        ----------   
        grid_res : int
            Number of bins of output grid in area            
        int_type : str
            interpolation method
            'IDW' Inverse Distance Interpolation
            'Kriging' Kriging Interpolation with pyKrige            
        figsize : matplotlib parameter, optional 
            Size of output figure in inches (default is (15,10))      
        time : str, optional
            Datetime string of desired timestamp (for example 'yyyy-mm-dd HH:MM')
            Default is None.
            If given the rain rate for this timestamp is plotted.
            If not given the accumulation of the whole time series is plotted.         
        method : str, optional
            Indicates how to claculate rainrate/sum of bidirectional link
            (Default is 'mean')
            'mean' average of near-far and far-near
            'max' maximum of near-far and far-near
            'min' minimum of near-far and far-near
            'nf', 'fn', 'fnp' etc. use direction from tx_rx_pairs               
        time_resolution : int, optional
                Resampling time for rain rate calculation. Only used if time
                is not None (Default is 15)               
        power : flt, optional
               Power of distance decay for IDW interpolation. Only used if 
               int_type is 'IDW' (Default is 2)   
        smoothing : flt, optional
               Power of smoothing factor for IDW interpolation. Only used if 
               int_type is 'IDW' (Default is 0)        
        krig_type : str, optional
                Parameters for Kriging interpolation. See pykrige documentation
                for information. Only used if int_type is 'Kriging' 
                (Default is 'Ordinary')
        variogram_model : str, optional
                Parameters for Kriging interpolation. See pykrige documentation
                for information. Only used if int_type is 'Kriging' 
                (Default is 'linear')        
        drift_terms : str, optional
                Parameters for Kriging interpolation. See pykrige documentation
                for information. Only used if int_type is 'Kriging' 
                (Default is ['regional_linear'])
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
        gg_tiles = img_tiles.OSM()

        ax.add_image(gg_tiles, 11)                 
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


        levels_rr = [0.5, 1.0, 1.5,2.0, 2.5, 5.0, 7.5, 10.0,12.5,15.0,20.0]
        levels_sum = [5.0,10.0,15.0,20.0,25.0,50.0,75.0,100.0,150.0,200.0,250.0]          

        
        #Definition of output grid
        gridx = np.linspace(self.set_info['area'][0],self.set_info['area'][1],grid_res)
        gridy = np.linspace(self.set_info['area'][2],self.set_info['area'][3],grid_res)   
       
        # MW data and metadata      
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
                       cml.metadata['lat_center'] = (cml.metadata['site_A']['lat']
                                                           +cml.metadata['site_B']['lat'])/2.
                       cml.metadata['lon_center'] = (cml.metadata['site_A']['lon']
                                                           +cml.metadata['site_B']['lon'])/2.                                                           
                           
                       if time is None:
                           for pair_id in cml.processing_info['tx_rx_pairs']:
                               if 'R_' + pair_id in cml.data.columns:
                                   cml.processing_info['accR_' + pair_id] = cml.data['R_' + pair_id].resample('H',how='mean').cumsum()[-1]  

        lons_mw=[]
        lats_mw=[]
        values_mw=[]        
 
        if time is None:
             for cml in self.set:
                 plist = []
                 for pair_id in cml.processing_info['tx_rx_pairs']:
                     try:
                         plist.append(cml.processing_info['accR_'+pair_id])
                     except:
                         pass       
                 if method == 'mean':
                     try:
                         precip = np.mean(plist)
                     except ValueError:
                         pass
                     except TypeError:
                         pass                     
                 elif method == 'max':    
                     try:
                         precip = np.max(plist)
                     except ValueError:
                         pass
                     except TypeError:
                         pass                      
                 elif method == 'min':
                     try:
                         precip = np.min(plist)  
                     except ValueError:
                         pass
                     except TypeError:
                         pass                      
                 elif method in cml.processing_info['tx_rx_pairs']:
                     if 'accR_' + pair_id in cml.processing_info.keys():
                         precip = cml.processing_info['accR_'+method]
                     else:
                         print 'Pair ID '+method+' not available for link '+cml.metadata['link_id']
                         precip = None                         
                 else:
                     print method+' not available for link '+cml.metadata['link_id']
                     precip = None     

                 if precip >= 0.:                                     
                     lons_mw.append(cml.metadata['lon_center'])
                     lats_mw.append(cml.metadata['lat_center'])
                     values_mw.append(precip)     
  
              
        else:
             start = pd.Timestamp(time) - pd.Timedelta('30s')
             stop = pd.Timestamp(time) + pd.Timedelta('30s')
             for cml in self.set:
                 plist = []
                 for pair_id in cml.processing_info['tx_rx_pairs']:
                     try:
                         plist.append((cml.data['R_'+pair_id].resample(str(time_resolution)+'Min',how='mean')[start:stop]).values[0])
                     except:
                         pass
 
                 if method == 'mean':
                     try:
                         precip = np.mean(plist)
                     except ValueError:
                         pass
                     except TypeError:
                         pass                        
                 elif method == 'max':
                     try:
                         precip = np.max(plist)
                     except ValueError:
                         pass
                     except TypeError:
                         pass                        
                 elif method == 'min':
                     try:
                         precip = np.min(plist)  
                     except ValueError:
                         pass
                     except TypeError:
                         pass                        
                 elif method in cml.processing_info['tx_rx_pairs']:
                     if 'R_' + pair_id in cml.data.keys():
                         precip = (cml.data['R_'+method].resample(str(time_resolution)+'Min',how='mean')[start:stop]).values[0]
                     else:
                         print 'Pair ID '+method+' not available for link '+cml.metadata['link_id']
                         precip = None
                 else:
                     print method+' not available for link '+cml.metadata['link_id']
                     precip = None
     
                 if precip >= 0.:    
                     lons_mw.append(cml.metadata['lon_center'])
                     lats_mw.append(cml.metadata['lat_center'])
                     values_mw.append(precip)                                                                    
                    
        if not all(v==0.0 for v in values_mw):                         
            if int_type == 'IDW':       
                interpol=mapping.inv_dist(lons_mw,lats_mw,values_mw,
                                              gridx,gridy,power,smoothing)
                                  
            elif int_type == 'Kriging':
                interpol=mapping.kriging(lons_mw,lats_mw,values_mw,gridx,gridy, 
                                         krig_type,variogram_model,drift_terms)                                      
            else:
                ValueError('Interpolation method not supported')                             
                                                                            
        if time is None:    
            if not all(v==0.0 for v in values_mw):                                                    
                cs = plt.contourf(gridx,gridy,interpol,levels=levels_sum,cmap=plt.cm.winter_r,transform=cartopy.crs.PlateCarree())
                cbar = plt.colorbar(cs,orientation='vertical', shrink=0.4)
                cbar.set_label('mm')
            plt.title('accumulated rainfall from time period: '+(self.set[0].data.index[0]).strftime('%Y-%m-%d %H:%M')+'UTC - '+
                        (self.set[0].data.index[-1]).strftime('%Y-%m-%d %H:%M')+'UTC',loc='right')

        else:
            if not all(v==0.0 for v in values_mw):
                cs = plt.contourf(gridx,gridy,interpol,levels=levels_rr,cmap=plt.cm.winter_r,alpha=0.6,transform=cartopy.crs.PlateCarree())
                cbar = plt.colorbar(cs,orientation='vertical', shrink=0.4)
                cbar.set_label('mm/h')            
            plt.title((pd.Timestamp(time)).strftime('%Y-%m-%d %H:%M')+'UTC',loc='right')

       
        if out_file is not None:
            plt.savefig(out_file,bbox_inches='tight',pad_inches=0)








                 