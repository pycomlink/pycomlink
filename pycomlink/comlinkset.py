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
import cartopy.crs as ccrs    
from cartopy.io.img_tiles import GoogleTiles
import pandas as pd

import math
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
                The order has to be:
                area = [lower_longitude,upper_longitude,
                        lower_latitude, upper_latitude]
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
        ax = plt.axes(projection=ccrs.PlateCarree())

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
        
        ax.set_extent((area[0], area[1], area[2], area[3]), crs=ccrs.PlateCarree())
        gg_tiles = GoogleTiles()
        ax.add_image(gg_tiles, 11)        
        
        for cml in self.set:
                   plt.plot([cml.metadata['site_A']['lon'],cml.metadata['site_B']['lon']],
                            [cml.metadata['site_A']['lat'],cml.metadata['site_B']['lat']],
                            linewidth=2,color='k',
                            transform=ccrs.Geodetic())        
        
        
    def do_wet_dry_classification(self, method='std_dev', 
                                        window_length=128,
                                        threshold=1,
                                        dry_window_length=600,
                                        f_divide=1e-3,
                                        reuse_last_Pxx=False,
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
        print_info : bool
                  Print information about executed method (Default is False)
        
        Note
        ----        
        WIP: Currently two classification methods are supported:
                - std_dev: Rolling standard deviation method [1]_
                - stft: Rolling Fourier-transform method [2]_                
                      
        
        """
        
        for cml in self.set:
            cml.do_wet_dry_classification(method, 
                                              window_length,
                                              threshold,
                                              dry_window_length,
                                              f_divide,
                                              reuse_last_Pxx,
                                              print_info)
            
    
                

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
        ------        
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
            if (cml.processing_info['tx_rx_pairs']['fn']['f_GHz'] is not None 
               and cml.processing_info['tx_rx_pairs']['nf']['f_GHz'] is not None):
                   cml.calc_R_from_A(a,b,approx_type)                
                
                                                           
               
    def spat_interpol(self, grid_res,
                 int_type,
                 figsize=(15,10),
                 time=None,
                 method='mean',
                 time_resolution=15,
                 power=2,smoothing=0,
                 krig_type='ordinary',
                 variogram_model='linear',
                 drift_terms=['regional_linear']):
                     
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
            'dir1' use only first direction from tx_rx_pairs
            'dir2' use only second direction from tx_rx_pairs                
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
                
        """
        
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')        
        gl.xlabels_top = False
        ax.set_extent((self.set_info['area'][0]-.05, self.set_info['area'][1]+.05,
                       self.set_info['area'][2]-.05, self.set_info['area'][3]+.05),
                         crs=ccrs.PlateCarree())
        gg_tiles = GoogleTiles()

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
                         transform=ccrs.Geodetic()) 


        levels_rr = [0.5, 1.0, 1.5,2.0, 2.5, 5.0, 7.5, 10.0,12.5,15.0,20.0]
        levels_sum = [5.0,10.0,15.0,20.0,25.0,50.0,75.0,100.0,150.0,200.0,250.0]          

        
        #Definition of output grid
        gridx = np.linspace(self.set_info['area'][0],self.set_info['area'][1],grid_res)
        gridy = np.linspace(self.set_info['area'][2],self.set_info['area'][3],grid_res)   
       
        # MW data and metadata
        ACC = {}       
        for cml in self.set:
            cumsu = {}
            if 'site_A' in cml.metadata and 'site_B' in cml.metadata:
                if 'lat' in cml.metadata['site_A'] and \
                   'lon' in cml.metadata['site_A'] and \
                   'lat' in cml.metadata['site_B'] and \
                   'lon' in cml.metadata['site_B']:
                       plt.plot([cml.metadata['site_A']['lon'],cml.metadata['site_B']['lon']],
                             [cml.metadata['site_A']['lat'],cml.metadata['site_B']['lat']],
                             linewidth=1,color='k',
                             transform=ccrs.Geodetic())  
                       cml.metadata['lat_center'] = (cml.metadata['site_A']['lat']
                                                           +cml.metadata['site_B']['lat'])/2.
                       cml.metadata['lon_center'] = (cml.metadata['site_A']['lon']
                                                           +cml.metadata['site_B']['lon'])/2.                                                           
                           
                       if time is None:
                           for pair_id in cml.processing_info['tx_rx_pairs']:
                               if 'R_' + pair_id in cml.data.columns:
                                   cumsu[pair_id]=cml.data['R_' + pair_id].resample('H',how='mean').cumsum()
                           if cumsu:        
                               ACC[cml.metadata['link_id']]=cumsu    

        lons_mw=[]
        lats_mw=[]
        values_mw=[]        
 
        if time is None:
             for cml in self.set:
                 if len(cml.processing_info['tx_rx_pairs']) == 2:
                     if cml.metadata['link_id'] in ACC:
                         if method == 'mean':
                             prep_sum = (ACC[cml.metadata['link_id']][cml.processing_info['tx_rx_pairs'].keys()[0]][-1]+
                                        ACC[cml.metadata['link_id']][cml.processing_info['tx_rx_pairs'].keys()[1]][-1])/2. 
                         elif method == 'max':
                             prep_sum = max(ACC[cml.metadata['link_id']][cml.processing_info['tx_rx_pairs'].keys()[0]][-1],
                                        ACC[cml.metadata['link_id']][cml.processing_info['tx_rx_pairs'].keys()[1]][-1])
                         elif method == 'min':
                             prep_sum = min(ACC[cml.metadata['link_id']][cml.processing_info['tx_rx_pairs'].keys()[0]][-1],
                                        ACC[cml.metadata['link_id']][cml.processing_info['tx_rx_pairs'].keys()[1]][-1])
                         elif method == 'dir1':
                             prep_sum = ACC[cml.metadata['link_id']][cml.processing_info['tx_rx_pairs'].keys()[0]][-1]
                         elif method == 'dir2':
                             prep_sum = ACC[cml.metadata['link_id']][cml.processing_info['tx_rx_pairs'].keys()[1]][-1]
                                    
                 else:
                     prep_sum = ACC[cml.metadata['link_id']][cml.processing_info['tx_rx_pairs'].keys()[0]][-1]
                 
                 if not math.isnan(prep_sum):  
                     lons_mw.append(cml.metadata['lon_center'])
                     lats_mw.append(cml.metadata['lat_center'])
                     values_mw.append(prep_sum)     

              
        else:
             start = pd.Timestamp(time) - pd.Timedelta('30s')
             stop = pd.Timestamp(time) + pd.Timedelta('30s')
             for cml in self.set:
                 if len(cml.processing_info['tx_rx_pairs']) == 2:
                     if 'R_'+cml.processing_info['tx_rx_pairs'].keys()[0] in cml.data.columns and 'R_'+cml.processing_info['tx_rx_pairs'].keys()[1] in cml.data.columns:
                         if method == 'mean':
                             prep_sum = ((cml.data['R_'+cml.processing_info['tx_rx_pairs'].keys()[0]].resample(str(time_resolution)+'Min',how='mean')[start:stop]+
                                         cml.data['R_'+cml.processing_info['tx_rx_pairs'].keys()[1]].resample(str(time_resolution)+'Min',how='mean')[start:stop])/2.)
                         elif method == 'max':
                             prep_sum = pd.DataFrame({'a':cml.data['R_'+cml.processing_info['tx_rx_pairs'].keys()[0]].resample(str(time_resolution)+'Min',how='mean')[start:stop],
                                                      'b':cml.data['R_'+cml.processing_info['tx_rx_pairs'].keys()[1]].resample(str(time_resolution)+'Min',how='mean')[start:stop]}).max(1)
                         elif method == 'min':
                              prep_sum = pd.DataFrame({'a':cml.data['R_'+cml.processing_info['tx_rx_pairs'].keys()[0]].resample(str(time_resolution)+'Min',how='mean')[start:stop],
                                                      'b':cml.data['R_'+cml.processing_info['tx_rx_pairs'].keys()[1]].resample(str(time_resolution)+'Min',how='mean')[start:stop]}).min(1)
                         elif method == 'dir1':
                             prep_sum = cml.data['R_'+cml.processing_info['tx_rx_pairs'].keys()[0]].resample(str(time_resolution)+'Min',how='mean')[start:stop]
                         elif method == 'dir2':
                             prep_sum = cml.data['R_'+cml.processing_info['tx_rx_pairs'].keys()[1]].resample(str(time_resolution)+'Min',how='mean')[start:stop]      
                 else:
                     prep_sum = cml.data['R_'+cml.processing_info['tx_rx_pairs'].keys()[0]].resample(str(time_resolution)+'Min',how='mean')[start:stop]

                 if not math.isnan(prep_sum):  
                     lons_mw.append(cml.metadata['lon_center'])
                     lats_mw.append(cml.metadata['lat_center'])
                     values_mw.append(prep_sum.values)
                     
                    
                        
                        
        if int_type == 'IDW':          
            interpol=mapping.inv_dist(lons_mw,lats_mw,values_mw,
                                          gridx,gridy,power,smoothing)
        elif int_type == 'Kriging':
            interpol=mapping.kriging(lons_mw,lats_mw,values_mw,gridx,gridy, 
                                     krig_type,variogram_model,drift_terms)  
        else:
            ValueError('Interpolation method not supported')                             
                                          
  
        if time is None:                                                        
            cs = plt.contourf(gridx,gridy,interpol,levels=levels_sum,cmap=plt.cm.winter_r,transform=ccrs.PlateCarree())
            plt.title('accumulated rainfall from time period: '+(self.set[0].data.index[0]).strftime('%Y-%m-%d %H:%M')+'UTC - '+
                        (self.set[0].data.index[-1]).strftime('%Y-%m-%d %H:%M')+'UTC',loc='right')
            cbar = plt.colorbar(cs,orientation='vertical')
            cbar.set_label('mm')
        else:

            cs = plt.contourf(gridx,gridy,interpol,levels=levels_rr,cmap=plt.cm.winter_r,alpha=0.6,transform=ccrs.PlateCarree())
            plt.title((pd.Timestamp(time)).strftime('%Y-%m-%d %H:%M')+'UTC',loc='right')
            cbar = plt.colorbar(cs,orientation='vertical')
            cbar.set_label('mm/h')
       

                 