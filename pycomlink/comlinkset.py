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
    """
    Commercial microwave link set (CMLS) class 
        - for all data processing 
        - for multiple links
        - contains a list of Comlink Objects defined by Comlink()   
    
    Attributes
    ----------
    cml : list of pycomlink class Comlink

    
    """    
    def __init__(self, cml_list,area):
        self.set = cml_list
        self.set_info = {'area':area}
        
    def info(self):
        """
        Print ComlinkSet info 
        
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
        """Show ComlinkSet locations on map 
                
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
        """
        Perform wet/dry classification for all CML time series in CMLS
        
     
        Attributes:
        ---------------------------------------------
        method: str
            WIP: currently two methods are supported:
                - std_dev: Rolling standard deviation method (Schleiss & Berne, 2010)
                - stft: Rolling Fourier-transform method (Chwala et al, 2012)
        window_length: int
            length of the sliding window        
        threshold: 
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
                                      
        """
        Perform baseline determination for all CML time series in CMLS
        
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

                              
        for cml in self.set:   
            cml.do_baseline_determination(method,
                                              wet_external,
                                              print_info)
  


    def do_wet_antenna_baseline_adjust(self,
                                       waa_max, 
                                       delta_t, 
                                       tau,
                                       wet_external=None):
                                           
        """
        Perform baseline adjustion due to wet antenna for all CML time series in CMLS
        
        Attributes:
        ---------------------------------------------
        waa_max: 

        delta_t:    
        
        tau:
        
        wet_external:
               
        """ 
                                           
        for cml in self.set:    
            cml.do_wet_antenna_baseline_adjust(waa_max,
                                                   delta_t,
                                                   tau,
                                                   wet_external)
                
 
    def calc_A(self, remove_negative_A=True):
        
        """
        Perform calculation of attenuation for all CML time series in CMLS
        
        Attributes:
        ---------------------------------------------
        remove_negative_A: bool
            assignment: negative values of Attenuation = 0
               
        """         
        
        for cml in self.set:
            cml.calc_A(remove_negative_A)
 
                    
                    
    def calc_R_from_A(self, a=None, b=None, approx_type='ITU'):
        
        """
        Perform calculation of rain rate from attenuation for all CML time series in CMLS
        
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
        """
        Plotting Inverse Distance Interpolation of Rain Sums or Rain Rate 
            on regular grid
        
        Parameter:
        ---------------
        area: list
            [lower_lon,upper_lon,lower_lat,upper_lat]
            
        grid_res: int
            number of bins of output grid in area
            
        int_type:str
            interpolation method
            'IDW' Inverse Distance Interpolation
            'Kriging' Kriging Interpolation with pyKrige
            
        figsize:
            size of output figure  
            
        time: string
            datetime string of desired timestamp (for example 'yyyy-mm-dd HH:MM')
            If given the rainrate for this timestamp is plotted.
            If not given the accumulation of the whole time series is plotted
            
               
        method: str
            how to claculate rainrate/sum of duplex link
                'mean' average of near-far and far-near
                'max' maximum of near-far and far-near
                'min' minimum of near-far and far-near
                'dir1' use only first direction from tx_rx_pairs
                'dir2' use only second direction from tx_rx_pairs
                
        time_resolution: int
                resampling time for rain rate calculation
                only used for type 'rr'
                
        power, smoothing: flt
                only if int_type 'IDW'
                 power of distance decay and smoothing factor for 
                 IDW interpolation    
                 
        krig_type, variogram_model, drift_terms: str
                only if int_type 'Kriging'     
                Parameters for Kriging interpolation
                (see pykrige documentation for information)
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
                               cumsu[pair_id]=cml.data['R_' + pair_id].resample('H',how='mean').cumsum()
                           ACC[cml.metadata['link_id']]=cumsu    

        lons_mw=[]
        lats_mw=[]
        values_mw=[]        

        if time is None:
             for cml in self.set:
                 if len(cml.processing_info['tx_rx_pairs']) == 2:
                     
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
            cbar = plt.colorbar(cs,orientation='vertical')
            cbar.set_label('mm')
        else:

            cs = plt.contourf(gridx,gridy,interpol,levels=levels_rr,cmap=plt.cm.winter_r,alpha=0.6,transform=ccrs.PlateCarree())
            cbar = plt.colorbar(cs,orientation='vertical')
            cbar.set_label('mm/h')
       

                 